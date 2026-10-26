# Research tools

Small helpers used while researching papers. They need no API keys and no dependencies (Node 18 or later).

`paper_research.js` (prints JSON to stdout):

| Command | Purpose |
|---|---|
| `node paper_research.js yt-search @AIPaperSlop "self distillation" 5` | Searches a channel's videos, up to N pages |
| `node paper_research.js yt-watch <videoId...>` | Gets the full title, date and description of each video; descriptions usually contain the arXiv link |
| `node paper_research.js arxiv-ids <file or ->` | Extracts unique arXiv IDs from any text or JSON |
| `node paper_research.js arxiv <id...>` | Gets arXiv title, date and abstract from the export API, 50 IDs per request with a 3 s delay |

Typical pipeline (PowerShell):

```powershell
$ids = (node paper_research.js yt-search '@AIPaperSlop' 'on-policy distillation' 5 | ConvertFrom-Json).videoId
$arx = node paper_research.js yt-watch @ids | node paper_research.js arxiv-ids - | ConvertFrom-Json
node paper_research.js arxiv @arx > papers.json
```

Write outputs outside the repo or under `outputs/`; don't commit them.

`yt-search` relies on YouTube's page layout (`ytInitialData`), so it may break if YouTube changes it.
