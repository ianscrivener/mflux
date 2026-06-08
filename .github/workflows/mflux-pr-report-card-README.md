# mflux PR Report Card

A validation Github workflow for mflux pull requests. On every non-draft PR it runs the project's own quality tools and posts a structured markdown report as a PR comment.

---

## What it does

For each PR, the workflow:

1. Runs **ruff**, **typos**, and **mypy** against the `main` branch first to establish a baseline — so contributors are not penalised for pre-existing issues
2. Runs the same tools against the PR branch
3. Runs `make test-fast` (pytest fast suite, no image generation)
4. Runs `make check` (the full pre-commit git hook suite)
5. Posts a formatted markdown report as a PR comment
6. Uploads the report as a GitHub Actions artifact

---

## Report format

```
# Mflux PR Report Card

| PR     | #426 — Add FLUX.2-klein-9b-kv KV-cache support |
| Author | @contributor                                   |
| Commit | 1b700a7                                        |
| Base   | main                                           |

| Check                     | Result    |
| ------------------------- | --------- |
| ruff                      | ✅        |
| typos                     | ✅        |
| mypy                      | ⚠️        |
| make test-fast            | ✅ Pass   |
| mflux pre-commit git hook | ✅ Pass   |

### ruff
### typos
### mypy
### make test-fast
### mflux pre-commit git hook
```

Main branch issues (if any) appear as a blockquote note beneath the relevant section — a signal to the maintainer without blaming the contributor.

---

## Tools and versions

All tool versions are pinned to match `.pre-commit-config.yaml`:

| Tool | Version | Install |
|------|---------|---------|
| ruff | 0.12.1 | `uv tool install` |
| typos | 1.33.1 | `brew install typos-cli` |
| mypy | 1.16.1 | `uv tool install` |
| pre-commit | latest | `uv tool install` |

Python 3.13, uv. Runner: `macos-14` (Apple Silicon — required for MLX).

---

## Requirements

- **No secrets required.** The workflow uses the built-in `github.token` for posting PR comments. Nothing to configure.
- **No changes to existing workflows.** This is an additive file only.
- Skips draft PRs automatically.
- Cancels superseded runs when a contributor force-pushes.

---

## Advisory only

This workflow is informational. It does not block merging. All results are the maintainer's own quality bar — the workflow runs the same tools contributors are already asked to run locally (`pre-commit run --all-files`).

---

# About GitHub Actions Setup

The original goal was to only run the GH workflow when PRs were submitted to Mflux main repo. Though it became apparent that there were more than one places that we may wish to use to get a report... so v1.1 axpands the range; 


### Modes

| Mode  | Repo             | Trigger | Explanation                                  |
| :---: | ---------------- | :------ | -------------------------------------------- |
| **1** | mflux            | auto    | auto triggered on PRs                        |
| **2** | mflux            | manual  | manual check for a specific PR - select PR # |
| **3** | contributor fork | auto    | auto triggered on each commit                |
| **4** | contributor fork | manual  | manual check on current branch               |

### Code Shape
```
.github/
  actions/
    mflux-report/
      action.yml                         ← steps 1.2–1.20, takes inputs: head_sha, base_sha, mode, pr_number
  workflows/
    mflux-pr-report-mflux-upstream.yml   ← Mode 1 & 2, posts as PR comment
    mflux-pr-report-contributor-fork.yml ← Mode 3 & 4, posts as job summary / artifact only  
```

**A few things worth knowing:**

**`action.yml`** — posts to `$GITHUB_STEP_SUMMARY` unconditionally, so you always get a rendered report in the Actions UI regardless of mode. PR comment is opt-in via `post_pr_comment: "true"`.

**`mflux-pr-report-mflux-upstream.yml`** — on `workflow_dispatch` it calls `gh pr view` to fetch all metadata via the API, so it works even if no checkout has happened yet. On `pull_request` event it reads directly from the event context.

**`mflux-pr-report-contributor-fork.yml`** — fetches upstream `main` SHA via `gh api repos/filipstrand/mflux/...`. You'll want to confirm that repo slug is correct, or parameterise it if you're validating against a different upstream.


