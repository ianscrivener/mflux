from dataclasses import dataclass, field


@dataclass
class MetaData:
    quantization_level: int | str | None = None
    mflux_version: str | None = None
    # Per-component quantization level, populated by the loader.
    # Keyed by component name (e.g. "text_encoder", "transformer", "vae").
    # If a component is missing from this dict, fall back to quantization_level
    # for backward compatibility.
    per_component_quantization_level: dict[str, int | str | None] = field(default_factory=dict)


@dataclass
class LoadedWeights:
    components: dict[str, dict]
    meta_data: MetaData

    def __getattr__(self, name: str) -> dict | None:
        if name in ("components", "meta_data"):
            return object.__getattribute__(self, name)
        return self.components.get(name)

    def num_transformer_blocks(self, component_name: str = "transformer") -> int:
        transformer = self.components.get(component_name)
        if transformer is None:
            for comp in self.components.values():
                if isinstance(comp, dict) and "transformer_blocks" in comp:
                    transformer = comp
                    break
        if transformer and "transformer_blocks" in transformer:
            return len(transformer["transformer_blocks"])
        return 0

    def num_single_transformer_blocks(self, component_name: str = "transformer") -> int:
        transformer = self.components.get(component_name)
        if transformer is None:
            for comp in self.components.values():
                if isinstance(comp, dict) and "single_transformer_blocks" in comp:
                    transformer = comp
                    break
        if transformer and "single_transformer_blocks" in transformer:
            return len(transformer["single_transformer_blocks"])
        return 0
