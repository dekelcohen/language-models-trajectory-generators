#!/usr/bin/env node
// Paper-research helpers (no API keys, Node >= 18, no deps). Output: JSON on stdout.
//   node paper_research.js yt-search <@channelHandle> <query> [maxPages]
//       Videos from a channel search (follows continuations). Fields: videoId, title, published, desc.
//   node paper_research.js yt-watch <videoId...>
//       Full title / publish date / description per video (descriptions usually hold the arXiv link).
//   node paper_research.js arxiv <id...>
//       arXiv metadata (title, published, summary) via the export API.
//   node paper_research.js arxiv-ids <file|->
//       Extracts unique arXiv IDs from text/JSON (e.g. pipe yt-watch output in).

const fs = require('fs');
const UA = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Accept-Language': 'en-US,en' };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function collectVideos(obj, out, conts) {
  if (Array.isArray(obj)) { obj.forEach((x) => collectVideos(x, out, conts)); return; }
  if (!obj || typeof obj !== 'object') return;
  if (obj.videoRenderer) {
    const v = obj.videoRenderer;
    out.push({
      videoId: v.videoId,
      title: v.title?.runs?.map((r) => r.text).join('') ?? null,
      published: v.publishedTimeText?.simpleText ?? null,
      desc: v.descriptionSnippet?.runs?.map((r) => r.text).join('') ?? null,
    });
  }
  if (obj.continuationCommand?.token) conts.push(obj.continuationCommand.token);
  for (const k in obj) collectVideos(obj[k], out, conts);
}

async function ytSearch(handle, query, maxPages = 5) {
  const url = `https://www.youtube.com/${handle}/search?query=${encodeURIComponent(query)}`;
  const html = await (await fetch(url, { headers: UA })).text();
  const init = html.match(/var ytInitialData = (\{.*?\});<\/script>/s);
  if (!init) throw new Error('ytInitialData not found (blocked or layout changed)');
  const key = html.match(/"INNERTUBE_API_KEY":"([^"]+)"/)?.[1];
  const ctx = html.match(/"INNERTUBE_CONTEXT":(\{.*?\}),"INNERTUBE_CONTEXT_CLIENT_NAME"/s)?.[1];
  const videos = [];
  let conts = [];
  collectVideos(JSON.parse(init[1]), videos, conts);
  for (let page = 1; page < maxPages && conts.length && key && ctx; page++) {
    const res = await fetch(`https://www.youtube.com/youtubei/v1/browse?key=${key}`, {
      method: 'POST',
      headers: { ...UA, 'Content-Type': 'application/json' },
      body: JSON.stringify({ context: JSON.parse(ctx), continuation: conts[0] }),
    });
    const next = [];
    const before = videos.length;
    collectVideos(await res.json(), videos, next);
    console.error(`page ${page + 1}: +${videos.length - before} videos`);
    conts = next;
    await sleep(500);
  }
  const seen = new Set();
  return videos.filter((v) => !seen.has(v.videoId) && seen.add(v.videoId));
}

async function ytWatch(ids) {
  const out = [];
  for (const id of ids) {
    try {
      const html = await (await fetch(`https://www.youtube.com/watch?v=${id}`, { headers: UA })).text();
      const unq = (s) => (s == null ? null : JSON.parse(`"${s}"`));
      out.push({
        id,
        title: unq(html.match(/"title":"((?:[^"\\]|\\.)*)","lengthSeconds"/)?.[1]),
        date: html.match(/"publishDate":"([^"]+)"/)?.[1] ?? null,
        desc: unq(html.match(/"shortDescription":"((?:[^"\\]|\\.)*)"/)?.[1]),
      });
    } catch (e) { out.push({ id, error: e.message }); }
    await sleep(300);
  }
  return out;
}

async function arxiv(ids) {
  const out = {};
  for (let i = 0; i < ids.length; i += 50) {
    const batch = ids.slice(i, i + 50).join(',');
    const xml = await (await fetch(`https://export.arxiv.org/api/query?id_list=${batch}&max_results=50`)).text();
    for (const e of xml.split('<entry>').slice(1)) {
      const id = e.match(/<id>https?:\/\/arxiv\.org\/abs\/([0-9.]+)/)?.[1];
      if (!id) continue;
      const clean = (s) => s?.replace(/\s+/g, ' ').trim() ?? null;
      out[id] = {
        title: clean(e.match(/<title>([\s\S]*?)<\/title>/)?.[1]),
        published: e.match(/<published>([^<]+)<\/published>/)?.[1] ?? null,
        summary: clean(e.match(/<summary>([\s\S]*?)<\/summary>/)?.[1]),
      };
    }
    await sleep(3000); // arXiv API etiquette
  }
  return out;
}

function arxivIds(text) {
  return [...new Set([...text.matchAll(/(?:arxiv\.org\/(?:abs|pdf|html)\/|arXiv:\s?)(\d{4}\.\d{4,5})/gi)].map((m) => m[1]))];
}

(async () => {
  const [cmd, ...args] = process.argv.slice(2);
  let result;
  if (cmd === 'yt-search') result = await ytSearch(args[0], args[1], Number(args[2] ?? 5));
  else if (cmd === 'yt-watch') result = await ytWatch(args);
  else if (cmd === 'arxiv') result = await arxiv(args);
  else if (cmd === 'arxiv-ids') result = arxivIds(fs.readFileSync(args[0] === '-' || !args[0] ? 0 : args[0], 'utf8'));
  else { console.error('usage: yt-search <@handle> <query> [maxPages] | yt-watch <ids..> | arxiv <ids..> | arxiv-ids <file|->'); process.exit(2); }
  console.log(JSON.stringify(result, null, 2));
})().catch((e) => { console.error(e.message); process.exit(1); });
