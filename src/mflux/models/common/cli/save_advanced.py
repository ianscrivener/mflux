"""mflux-save-advanced console script entry point.

A new CLI for converting pytorch/HuggingFace models to the mflux MLX
safetensors format with **per-component quantization**. Components without a
--quantize_XXX flag are skipped entirely (the resulting directory has no
subdir for them).

Usage:
  mflux-save-advanced \\
      --bf16_model /path/to/flux-dev-bf16 \\
      --quantize_vae 4 \\
      --quantize_transformer 8 \\
      --quantize_text_encoder bf16 \\
      --path /tmp/flux-dev-mixed \\
      --force
"""
from mflux.cli.parser.parsers import CommandLineParser
from mflux.utils.save_advanced_runner import main as runner_main


def main() -> None:
    parser = CommandLineParser(
        description=(
            "Convert a pytorch/HuggingFace model to mflux MLX safetensors "
            "with per-component quantization."
        )
    )
    parser.add_general_arguments()
    parser.add_save_advanced_arguments()
    args = parser.parse_args()
    runner_main(args)


if __name__ == "__main__":
    main()
