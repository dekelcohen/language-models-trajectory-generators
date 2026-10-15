# Copy uncommitted changes (modified + untracked .py) of this repo into an experiment
# worktree, e.g. `git worktree add ..\lmtg_exp HEAD` (subprocess runs import live files,
# so never run experiments from the tree being edited).
param([string]$dst=(Join-Path $PSScriptRoot "..\..\..\lmtg_exp"))
$src=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $src
git -C $dst checkout -- . 2>$null
$files = @(git diff --name-only) + @(git ls-files --others --exclude-standard | Where-Object { $_ -like "*.py" })
foreach ($p in $files) { $d = Join-Path $dst $p; New-Item -ItemType Directory -Force (Split-Path $d) | Out-Null; Copy-Item (Join-Path $src $p) $d -Force }
Pop-Location
"synced: $($files -join ', ')"
