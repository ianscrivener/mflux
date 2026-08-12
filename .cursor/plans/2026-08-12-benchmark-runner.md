# Benchmark runner

## Goals

- Move the Z-Image benchmark prompts to `BENCHTEST/prompts.txt`.
- Add `BENCHTEST/test.py` to run each prompt and quantization combination.
- Save generated images under `BENCHTEST/results/images`.
- Append model size, peak process memory, and elapsed inference time to separate CSV logs.

## Constraints

- Preserve the existing model root, CLI command, dimensions, seed, steps, and quantizations.
- Use the Python standard library so no project dependency changes are needed.
- Keep existing unrelated workspace changes intact.

## Verification

- Compile the new script with `uv run python -m py_compile BENCHTEST/test.py`.
- Run `--help` to verify the executable interface without starting inference.
