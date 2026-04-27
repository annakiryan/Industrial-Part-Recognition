from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil

from app.configs.config import AppConfig
from app.core.cad_converter import convert_single_dwg_to_dxf
from app.core.factories import build_template_classifier
from app.core.template_mask_builder import (
    build_mask_from_dxf,
    extract_features_from_dxf_and_mask,
)
from app.core.utils import save_json


@dataclass
class AddTemplatePaths:
    template_name: str
    template_dir: Path
    saved_dwg_path: Path
    saved_dxf_path: Path
    saved_mask_path: Path
    saved_features_path: Path


@dataclass
class AddTemplateResult:
    template_name: str
    template_dir: str
    dwg_path: str
    dxf_path: str
    mask_path: str
    features_path: str
    added_to_matrix: bool
    added_count: int


class TemplateAddUseCase:
    def __init__(self, config: AppConfig):
        self.config = config
        self.templates_dir = config.paths.templates_dir

        self.oda_exe = Path(
            r"C:\Program Files\ODA\ODAFileConverter 27.1.0\ODAFileConverter.exe"
        )
        self.oda_output_version = "ACAD2018"

        self.canvas_size = 5000
        self.padding = 40
        self.line_thickness = 3

    def _validate_input_dwg(self, dwg_path: Path) -> None:
        if not dwg_path.exists():
            raise FileNotFoundError(f"DWG file not found: {dwg_path}")

        if dwg_path.suffix.lower() != ".dwg":
            raise ValueError(f"Expected .dwg file, got: {dwg_path.suffix}")

    def _prepare_template_paths(
        self,
        dwg_path: Path,
        template_name: Optional[str],
        overwrite: bool,
    ) -> AddTemplatePaths:
        if template_name is None:
            template_name = dwg_path.stem

        template_dir = self.templates_dir / template_name

        if template_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Template directory already exists: {template_dir}"
                )
            shutil.rmtree(template_dir)

        template_dir.mkdir(parents=True, exist_ok=True)

        return AddTemplatePaths(
            template_name=template_name,
            template_dir=template_dir,
            saved_dwg_path=template_dir / "part.dwg",
            saved_dxf_path=template_dir / "part.dxf",
            saved_mask_path=template_dir / "mask.png",
            saved_features_path=template_dir / "features.json",
        )

    def _build_template_artifacts(
        self,
        source_dwg_path: Path,
        paths: AddTemplatePaths,
    ) -> None:
        shutil.copy2(source_dwg_path, paths.saved_dwg_path)

        convert_single_dwg_to_dxf(
            oda_exe=self.oda_exe,
            input_dwg_path=paths.saved_dwg_path,
            output_dxf_path=paths.saved_dxf_path,
            version=self.oda_output_version,
        )

        mask, dxf_paths = build_mask_from_dxf(
            dxf_path=paths.saved_dxf_path,
            output_mask_path=paths.saved_mask_path,
            canvas_size=self.canvas_size,
            padding=self.padding,
            line_thickness=self.line_thickness,
        )

        features = extract_features_from_dxf_and_mask(
            paths=dxf_paths,
            mask=mask,
            min_hole_area_px=self.config.features.min_hole_area_px,
            camera_id=self.config.features.camera_id,
            material=self.config.features.material,
            thickness_mm=self.config.features.thickness_mm,
        )

        save_json(paths.saved_features_path, features)

    def _update_template_matrix(self) -> int:
        classifier = build_template_classifier(self.config)
        return classifier.update_template_matrix()

    def add_from_dwg(
        self,
        dwg_path: str | Path,
        template_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> AddTemplateResult:
        dwg_path = Path(dwg_path)

        self._validate_input_dwg(dwg_path)
        paths = self._prepare_template_paths(
            dwg_path=dwg_path,
            template_name=template_name,
            overwrite=overwrite,
        )

        self._build_template_artifacts(
            source_dwg_path=dwg_path,
            paths=paths,
        )

        added_count = self._update_template_matrix()

        return AddTemplateResult(
            template_name=paths.template_name,
            template_dir=str(paths.template_dir),
            dwg_path=str(paths.saved_dwg_path),
            dxf_path=str(paths.saved_dxf_path),
            mask_path=str(paths.saved_mask_path),
            features_path=str(paths.saved_features_path),
            added_to_matrix=added_count > 0,
            added_count=added_count,
        )