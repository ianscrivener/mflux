# `mflux-pr-docs` AI Agent Skill


***Making PRs easier... an AI skill that works across Claude Code, Cursor, and GitHub Co-Pilot.*** 


### Goals
 1. Guide AI to write copy that is more concise and easy to read (using Simple Technical English \*)
 2. AI autofill "What" - see pr-what-prompt.md
 3. AI autofill release note code block
 4. AI autofill release note 


### Process:
1. Read `git diff main...HEAD`, the commit log and the linked issues.
1. Copy `.github/pull_request_template.md` to `tmp-PR-content.md`.
1. Fill in the `What` section (the benefit prompt), the `release-note` block, and the `Verification` section from commands that were actually run.
1. Run the same regex check as CI on `tmp-PR-content.md`.
1. Show `tmp-PR-content.md` to the developer for approval or edits. Only after approval does it run `gh pr create --body-file tmp-PR-content.md`, or `gh pr edit` if the PR already exists.


\* Simple Technical English Skill - [asd-ste100-skill](https://github.com/danyuchn/asd-ste100-skill)
