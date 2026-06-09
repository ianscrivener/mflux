"""Overwrite-confirmation helper for mflux-save-advanced.

Used by both --bf16_model and --nvfp4_model flows. Refuses to silently overwrite
an existing directory: prompts y/N on a TTY, hard-errors in a non-TTY environment
(unless --force is set).
"""
import sys
from pathlib import Path


def confirm_overwrite(path: Path, force: bool = False) -> None:
    """Confirm before overwriting an existing directory.

    - If `force` is True, return immediately without prompting.
    - If `path` does not exist, return immediately.
    - If stdin is not a TTY, abort with SystemExit (so a piped/redirected
      invocation cannot accidentally clobber data).
    - Otherwise, prompt y/N; anything other than 'y' aborts with SystemExit.
    """
    if force:
        return
    if not Path(path).exists():
        return
    if not sys.stdin.isatty():
        print(
            f"Refusing to overwrite {path} in non-interactive mode. "
            "Use --force to bypass.",
            flush=True,
        )
        sys.exit(2)
    answer = input(f"{path} already exists. Overwrite? [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.", flush=True)
        sys.exit(1)
