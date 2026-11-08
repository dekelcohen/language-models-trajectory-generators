# N-gram memory lookup (Qwen3.8-Next / DeepSeek Engram)

How a huge n-gram embedding table adds capacity with almost no extra compute or VRAM.

**Sources**
- Qwen3.8-Next paper, §2.3 "N-gram Embedding": [arXiv 2608.30320](https://arxiv.org/abs/2608.30320).
  - 125B MoE with 6B active parameters, plus 51B of n-gram tables kept off the GPU.
  - One n-gram layer at layer 2.
  - No n-gram code released.
- DeepSeek Engram, "Conditional Memory via Scalable Lookup": [arXiv 2601.07372](https://arxiv.org/abs/2601.07372) and [github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) (`engram_demo_v1.py`).
  - Qwen cites Engram for its context-aware gating design.
  - Code references below point to this demo file.

## Legend (tensor dims)
| Symbol | Meaning | Demo value |
|---|---|---|
| **B** | Batch: how many sequences are processed together | 1 in the example |
| **L** | Sequence length in tokens | 7 in the example |
| **HC** | Residual lanes (hyper-connection branches) | 4 |
| **D** | Model hidden width, the size of one lane's vector | 1024 |
| **O** | N-gram orders used (2-gram, 3-gram) | 2 |
| **K** | Hash heads per order | 8 |
| **d_h** | Width of one hash-head row | 64 |
| **d_mem** | Memory vector width = O·K·d_h | 1024 |

## Why 4 residual lanes ("hyper-connections", HC)
- **A normal transformer has one residual stream.** Each position carries a single vector of size D. Every layer reads it, computes, and adds its result back.
- **That lane is a bottleneck.** Every layer writes into the same D numbers, so token identity, syntax, facts and reasoning all compete for space and can overwrite each other.
- **Hyper-connections widen it to HC = 4 parallel lanes per token** (shape `[B, L, 4, D]`).
- **Each layer reads a learned mix** of the 4 lanes as its input. It still computes at width D, so its compute cost is unchanged.
- **Each layer writes back with learned per-lane weights.** A cheap learned 4×4 matrix also shuffles information between lanes.
- **Result: more storage room in the residual path for very little compute.** Lanes can specialize, e.g. one keeps "what token is this", another accumulates "what's going on". DeepSeek uses this (mHC), and Qwen's Gated Residual also uses 4 branches.
- **What it means for Engram:** each lane gets its own gate (`key_projs[hc]`, `norm1[hc]`, `norm2[hc]`). The same memory can be written strongly into a lane that tracks facts or entities and barely into a lane that tracks grammar.
- **In the demo, HC is mocked.** The embedding is copied into 4 identical lanes (`unsqueeze(2).expand(...,4,...)`), and lane 0 is taken at the end.

## The story
Example: the model reads **"Yesterday Diana, Princess of Wales"** (B = 1, L = 7 tokens, e.g. `Yesterday | Diana | , | Princess | of | Wales | ...`).

**[S1] Normalize tokens** (`CompressedTokenizer`)
- Token ids are mapped to "canonical" ids: lowercase, accents removed, whitespace unified. So "Wales", " wales" and "WALES" become one id.
- This makes the phrase table smaller and denser.

**[S2] Build phrases and hash them** (`NgramHashMapping._get_ngram_hashes`)
- At every position, take the 2-gram and 3-gram ending there. At "wales" these are `(of, wales)` and `(princess, of, wales)`.
- Each phrase is hashed K = 8 ways (multiply-XOR, then `% prime`). That gives 16 row numbers per token.
- This runs on the CPU and needs only token ids, so it can happen **before** the GPU forward pass.

**[S3] Prefetch rows** (`MultiHeadEmbedding`, rows in CPU RAM in production)
- Only those 16 rows per token (64 numbers each) are copied over PCIe.
- This runs while the GPU computes layer 1 (`TransformerBlock` 0).
- The 51B-parameter table never sits in VRAM.

**[S4] Layer 1 runs normally**
- Attention lets "wales" see "Diana, Princess of". MoE/FFN adds features.
- The lanes `hidden_states` (h_t) now roughly say "royal title, linked to Diana".

**[S5] Assemble the memory vector `e`**
- The 16 rows are concatenated into `e`, with 1024 numbers per token.
- Example content of `e` for "wales": "Princess of Wales → Diana, British royal…".
- It has no context: it is identical every time this phrase appears.

**[S6] Gate per lane: does this memory fit?** (`key_projs[hc]`, `norm1/norm2`, sigmoid)
- `key` = what the memory is about.
- `query` = what this lane currently thinks (h_t).
- Similar → gate ≈ 1; unrelated → ≈ 0.5; conflicting → ≈ 0.
- Examples:
  - **Royal context:** royalty vs royalty, so the gate is about 0.98.
  - **"…the coast of Wales":** the `(of, wales)` memory leans royal, the context is geography, so the gate is about 0.05.
  - **Hash collision:** one of the 8 heads hit a "stock market crash" row. The other 7 dilute it, and if the overall memory still mismatches, the gate closes.

**[S7] Write the value, scaled by each lane's gate** (`value_proj`, `gates * value`)
- One value vector per token, copied to 4 lanes and multiplied by each lane's gate.

**[S8] Smooth over nearby tokens** (`ShortConv`)
- A tiny causal convolution blends this token's memory with the memories at t−3, t−6 and t−9.
- So "princess", "of" and "wales" reinforce each other.

**[S9] Residual add, then continue** (`TransformerBlock.forward`)
- `hidden += engram_out`, then layer 2 attention and MoE run as usual.
- The entity "Diana, Princess of Wales" is known by layer 2 instead of being assembled over layers 2–6.
- The spare depth goes to reasoning and long context.

**[S10] Generate the next token**
- Only the new token's phrases are hashed and prefetched (16 rows), so cost per generated token stays flat.

### Gate values (illustrative)
Gate ≈ sigmoid(sign·√(32·cos)), where cos is the cosine between key and query:

| cos(key, query) | raw | after signed √ | gate |
|---|---|---|---|
| +0.5 (fits) | 16 | 4.0 | **0.98** open |
| 0 (unrelated) | 0 | 0 | 0.5 half |
| −0.3 (conflicts) | −9.6 | −3.1 | **0.04** closed |

Unrelated gives 0.5, not 0. To close the gate, training must learn to point the key against the query, i.e. a negative cosine.

## Annotated code (faithful to the demo, simplified)
```python
# ---------- CPU side: before the GPU forward pass ----------
def hash_ngrams(token_ids):                          # token_ids: [B, L] int
    ids = compressed_tokenizer(token_ids)            # [S1] [B, L] canonical ids (lowercased etc.)
    prev1 = shift_right(ids, 1, pad=PAD)             # [B, L] token at t-1
    prev2 = shift_right(ids, 2, pad=PAD)             # [B, L] token at t-2
    rows = []
    for n, toks in [(2, [ids, prev1]), (3, [ids, prev1, prev2])]:   # [S2] O = 2 orders
        mix = toks[0] * m[0]
        for k in range(1, n):
            mix ^= toks[k] * m[k]                    # multiply-XOR → one 64-bit fingerprint per phrase
        for head in range(8):                        # K = 8 hash heads, each its own prime-sized table
            rows.append(mix % prime[n][head])        # [B, L] row index into that head's table
    return stack(rows, dim=-1)                       # [B, L, O*K=16] row ids — depends only on text

# ---------- GPU side: Engram module at layer 2 ----------
def engram_forward(lanes, row_ids):
    # lanes   : [B, L, HC=4, D=1024]  residual lanes after layer 1 = h_t (has seen context via attention)
    # row_ids : [B, L, 16]            from hash_ngrams, prefetched while layer 1 ran

    rows = table[row_ids + head_offsets]           # [S3] [B, L, 16, d_h=64]  gathered memory rows
    e    = rows.flatten(-2)                        # [S5] [B, L, d_mem=1024] phrase memory, context-free

    v = W_V(e)                                     # [B, L, D]  "what to write" (one Linear 1024→1024, shared by lanes)

    gates = []
    for hc in range(4):                            # [S6] one gate per lane
        key   = RMSNorm_k[hc](W_K[hc](e))          # [B, L, D]  "what this memory is about", lane-specific view
        query = RMSNorm_q[hc](lanes[:, :, hc])     # [B, L, D]  "what this lane currently thinks" (h_t)
        score = (key * query).sum(-1) / sqrt(D)    # [B, L]     ≈ 32·cos(key, query), range ~[-32, 32]
        score = sign(score) * sqrt(abs(score))     # [B, L]     squash to ~[-5.7, 5.7] → sigmoid stays sensitive
        gates.append(sigmoid(score))               # [B, L]     0 = reject memory, 1 = accept
    gates = stack(gates, dim=2)[..., None]         # [B, L, HC, 1]

    value = gates * v[:, :, None, :]               # [S7] [B, L, HC, D]  same content, per-lane volume
    out   = value + ShortConv(value)               # [S8] [B, L, HC, D]  blend with memories at t-3, t-6, t-9
    return out

# ---------- Transformer block ----------
def block_forward(lanes, row_ids, layer_id):
    if layer_id in ENGRAM_LAYERS:                  # Qwen: one layer (2); Engram demo: [1, 15]
        lanes = lanes + engram_forward(lanes, row_ids)   # [S9] residual add
    lanes = lanes + attention(lanes)               # attention mixes tokens (reads a learned mix of lanes)
    lanes = lanes + moe(lanes)                     # experts add features
    return lanes                                   # [B, L, HC, D]
```

## Where the weights come from
- `table`, `W_K[hc]`, `W_V`, the norms and `ShortConv` are all ordinary learned parameters.
- They are trained by backprop from the next-token loss, together with the whole model. Only the rows actually used in a batch get gradient updates.
- In Qwen, W_K/W_V are trained with Muon and the tables with AdamW.

## Why it saves compute and VRAM
- **Compute:** per token, about 16 row gathers plus two D×D matrix multiplies and a small convolution. This is independent of table size, so Qwen adds 51B parameters at near-zero extra compute.
- **VRAM:**
  - Row ids depend only on token ids, so they are known before the forward pass. MoE routing, by contrast, depends on hidden states.
  - The table lives in CPU RAM. Only the rows in use (a few KB per token) cross PCIe, overlapped with layer-1 compute. That is why the layer sits at layer 2.
  - Frequent n-grams can be cached in VRAM (phrase use is Zipf-distributed).
- **Measured (Engram):** a 100B-parameter table fully in CPU RAM cost under 3% throughput on an H800. The 8B dense model went from 6316 to 6140 tok/s.

## How it differs from a token embedding
- **The key is a phrase** (2–3 tokens), not a single token. Effectively a vocabulary of millions of phrases, with no change to the tokenizer or the output softmax.
- **Injected at layer 2 and gated by h_t,** which has already seen the context, so mismatched or colliding memories are suppressed.
- **Attention still mixes tokens afterward.** But early layers no longer have to assemble multi-token entities. LogitLens/CKA analysis shows earlier "prediction-ready" layers, i.e. more effective depth. Attention is freed for long-range context (multi-query needle-in-a-haystack 97 vs 84).

## Multiple meanings ("Wales" = royal title vs country)
- **Today:** one memory vector holds several meanings at once, like a normal word embedding.
  - The gate is one number per lane for the whole memory. It turns the memory up or down; it cannot choose among meanings.
  - Longer n-grams (`(coast, of, wales)`) are less ambiguous.
  - Later layers pick the relevant meaning from context.
- **Possible extension:** store S meaning vectors per phrase, fetch all of them (still prefetchable), and let `softmax(h_t·W_K·candidate_i)` pick among them on the GPU.
  - Cost: S× storage and transfer, and each phrase's training data is split S ways.
  - Related work: Product-Key Memory and Meta's Memory Layers. Those choose addresses from h_t, which breaks prefetching.
  - A cheaper middle ground: one gate per hash-head slice (16 instead of 1).
  - Neither the Engram nor the Qwen paper reports testing this.

## Qwen-specific findings (§2.3)
- **One layer is enough.** A single n-gram layer at layer 2 matched multi-layer setups; layer 2 also gives time to prefetch.
- **Loss vs benchmarks:** a larger n-gram vocabulary (20× to 200× the 250K base vocab) lowers loss steadily, but downstream benchmarks saturate. Chinese benchmarks keep improving.
- **Swapping experts for n-gram slots at a fixed total size gave no gain.** Memory and experts play different roles.
