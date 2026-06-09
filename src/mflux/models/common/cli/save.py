from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image import ZImage, ZImageTurbo


def resolve_model_class(args):
    """Pick a model class from CLI args.

    Reused by mflux-save and mflux-save-advanced.
    """
    model_name_lower = args.model.lower()
    base_model_lower = (args.base_model or "").lower()
    if "ernie" in model_name_lower:
        return ErnieImage
    if "qwen" in model_name_lower and "edit" in model_name_lower:
        return QwenImageEdit
    if "qwen" in model_name_lower:
        return QwenImage
    if "fibo" in model_name_lower:
        return FIBO
    if "z-image-turbo" in model_name_lower or "zimage-turbo" in model_name_lower:
        return ZImageTurbo
    if "z-image" in model_name_lower or "zimage" in model_name_lower:
        return ZImage
    if "flux2" in model_name_lower or "flux.2" in model_name_lower:
        return Flux2Klein
    if "ideogram" in model_name_lower or "ideogram" in base_model_lower:
        return Ideogram4
    return Flux1


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a quantized version of a model to disk.")  # fmt: off
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    args = parser.parse_args()

    # 1. Determine model class based on model name
    model_class = resolve_model_class(args)

    if model_class is Ideogram4:
        model_config = Ideogram4WeightDefinition.resolve_inference_config(
            args.model,
            args.base_model,
            args.model_path,
        )
        model_path = None if Ideogram4WeightDefinition.is_builtin_name(args.model) else args.model_path
    else:
        model_config = ModelConfig.from_name(args.model, base_model=args.base_model)
        model_path = args.model_path

    # 2. Load, quantize and save the model
    model = model_class(
        quantize=args.quantize,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        model_path=model_path,
        model_config=model_config,
    )
    model.save_model(args.path)


if __name__ == "__main__":
    main()
