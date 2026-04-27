from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathsConfig:
    templates_dir: Path = Path("templates")
    new_templates_dir: Path = Path("data/new_templates")
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")


@dataclass
class BinarizationConfig:
    use_hsv_v: bool = True
    blur_ksize: int = 5
    threshold_mode: str = "otsu"
    fixed_threshold: int = 30
    open_kernel_size: int = 5
    close_kernel_size: int = 5
    min_component_area_px: int = 300
    crop_margin_px: int = 10


@dataclass
class FeatureExtractionConfig:
    mm_per_px: float = 0.105
    min_hole_area_px: int = 20
    camera_id: str = "cam_01"
    material: Optional[str] = "steel"
    thickness_mm: Optional[float] = 5.0


@dataclass
class ClassificationConfig:
    metric: str = "manhattan"
    normalize: bool = True
    top_k: int = 3
    cache_dir: Path = Path("templates/_cache")
    template_cache_meta: Path = Path("templates/_cache/template_cache_meta.json")
    cache_npz_path: Path = Path("templates/_cache/template_cache.npz")
    template_features_config: Path = Path("app/configs/template_features_config.json")


@dataclass
class UncertaintyConfig:
    gap_threshold_1_2: float = 0.1
    gap_threshold_2_3: float = 0.1


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    binarization: BinarizationConfig = field(default_factory=BinarizationConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)