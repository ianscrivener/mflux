# Issue Triage

### (1) Get GitHub Issues
`scripts/issue_triage/get_issue_data.py` fetches all open and closed GitHub issues (not PRs) from mflux-community/mflux using the authenticated gh CLI, then writes one JSON record per issue to `scripts/issue_triage/issues.jsonl`. Each record includes the issue number, title, author, creation date, open/closed state, and body.


```sh
uv run --group issue-triage python scripts/issue_triage/get_issue_data.py
```


### (2) Classify Issues

`scripts/issue_triage/classify_1.py` reads GitHub issue records from `issues.jsonl` and adds category labels with a local ONNX model.

Run the classifier with its optional dependency group:

```sh
uv run --group issue-triage python scripts/issue_triage/classify_1.py
```

Use `--overwrite` to classify issues that already have labels.

```sh
uv run --group issue-triage python scripts/issue_triage/classify_1.py --overwrite
```



Use `--totals` to count saved labels without model inference, optionally outputting in JSON format:

```sh
uv run --group issue-triage python scripts/issue_triage/classify_1.py --totals
uv run --group issue-triage python scripts/issue_triage/classify_1.py --totals --json
```


---
### To Do
1. Test GitHub Actions configuration and check that auto-sending model comments works correctly. 