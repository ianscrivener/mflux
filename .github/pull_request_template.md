## What

<!-- A headline sentence, then the benefits for each group of users. Link issues: Fixes #... -->

## Checklist (definition of done)

- [ ] **Tests added/updated run in CI by default**<br>
Ran `just test`.<br>
Only mark `@pytest.mark.slow` or `@pytest.mark.high_memory_requirement` when a test exceeds CI's time or memory budget (typically weight downloads / image generation).<br><br>
- [ ] **`ruff check` and `ruff format` are clean**<br>
`uv run ruff` uses the version pinned in the dev dependencies of `pyproject.toml`, which is the single source of truth for pre-commit and CI; `pre-commit run -a` covers it locally.<br><br>
- [ ] **Release note block below filled in**<br>
Every PR gets a note. Do not write `none`.<br><br>
- [ ] **Docs updated where behavior changed**<br>
README examples/table rows are part of the API contract (see `.cursor/rules/RULE.md`).<br><br>
- [ ] **New model: shared config wiring**<br>
aliases, default steps, mflux-save dispatch, capabilities, completions, thin CLI entrypoint and `src/mflux/models/<name>/README.md`.<br><br>
- [ ] **New/changed CLI: ignored/rejected options declared**<br>(`IGNORED_OPTIONS`/`REJECTED_OPTIONS`) and `warn_ignored_options` actually called in `main()`<br>`mflux-capabilities` must stay truthful.



## Release note

```release-note
```

<!-- One or two sentences that describe the change, harvested into the release notes at
     release time. Every PR gets a note. Do not write `none`. The PR label selects the
     release section. A change that only contributors see (CI, tests, docs, agent skills)
     gets the `chore` or `ci` label and goes in the Internal section. The block starts
     empty on purpose: CI fails until you write the note. -->

## Verification

<!-- Commands you ran and what you observed. Include generated images/screenshots for model-affecting changes. -->
