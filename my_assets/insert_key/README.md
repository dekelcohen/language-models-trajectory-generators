# insert_key assets

- `key.obj` mesh + `meshes/key_basecolor.jpg`: derived from the RoboDojo benchmark
  (https://github.com/RoboDojo-Benchmark/RoboDojo, Apache-2.0) `insert_key` key asset,
  decimated with UVs transferred.
- Everything else (table, key stand, lock, textures) is procedural.

Regenerate: `tools/extract_robodojo_usdz.py` (needs `usd-core`; writes gitignored
`cache/robodojo/extracted/`), then `tools/build_assets.py`.
