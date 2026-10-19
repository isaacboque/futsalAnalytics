# futsalAnalytics (GitHub repo `isaacboque/futsalAnalytics`)

Public GitHub counterpart of the local `IAFS` folder (whose remote is named
`futsalAnalytics`). Python 3.10+, YOLO (ultralytics) + OpenCV, `src/futsal_analytics/`.
Branch: `main`. This is the **closed thesis baseline** (TFG): no behaviour changes to the
pipeline — it must keep reproducing the thesis results. Unlike IAFS/pivotiq, this repo has
no `src/pivotiq/` package.

## Commands

- `pip install -e ".[dev,viewer]"`
- `pytest` (config in `pyproject.toml`: imports this checkout's `src/`, not another editable install)
- `ruff check src tests web` / `ruff format .` (line-length 110)

## Notes

- See IAFS/CLAUDE.md for torch/CUDA, `supervision<0.30` pin and model-weights gotchas.
- Don't push to `main` without asking Isaac (root CLAUDE.md rule).
- Public repo: keep unrelated or private details out of code, docs and commits.
