from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.configs.config import AppConfig
from app.core.binarization import (
    binarize_detail,
    binarize_details_mask,
    crop_mask_by_bbox,
)
from app.core.feature_extraction import extract_features_from_mask
from app.core.factories import build_template_classifier
from app.core.utils import save_json


class RecognitionPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.templates_dir = config.paths.templates_dir
        self.input_dir = config.paths.input_dir
        self.output_dir = config.paths.output_dir

        self.classifier = build_template_classifier(config)
        self.classifier.load_cache()

        self.allowed_exts = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

    def _prepare_output_dir(self, image_path: Path) -> Path:
        sample_dir = self.output_dir / image_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    def _prepare_output_dir_by_name(self, sample_name: str) -> Path:
        sample_dir = self.output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    def _make_timestamp_name(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _extract_features(self, mask: np.ndarray) -> Dict[str, Any]:
        return extract_features_from_mask(
            mask=mask,
            mm_per_px=self.config.features.mm_per_px,
            min_hole_area_px=self.config.features.min_hole_area_px,
            camera_id=self.config.features.camera_id,
            material=self.config.features.material,
            thickness_mm=self.config.features.thickness_mm,
        )

    def _select_candidates(
        self,
        top_matches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates = list(top_matches[:3])

        if len(candidates) == 3:
            d2 = candidates[1]["distance"]
            d3 = candidates[2]["distance"]
            if (d3 - d2) > self.config.uncertainty.gap_threshold_2_3:
                candidates = candidates[:2]

        if len(candidates) == 2:
            d1 = candidates[0]["distance"]
            d2 = candidates[1]["distance"]
            if (d2 - d1) > self.config.uncertainty.gap_threshold_1_2:
                candidates = candidates[:1]

        return candidates

    def _classify_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        classification_result = self.classifier.classify_features_json(
            features_json=features,
            top_k=self.config.classification.top_k,
        )

        candidates = self._select_candidates(classification_result["top_matches"])

        predicted_label = None
        best_dwg_path = None

        if len(candidates) == 1:
            predicted_label = candidates[0]["label"]
            best_dwg_path = candidates[0]["dwg_path"]

        return {
            "classification_result": classification_result,
            "predicted_label": predicted_label,
            "best_dwg_path": best_dwg_path,
            "candidates": candidates,
        }

    def _classify_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        features = self._extract_features(mask)
        classified = self._classify_features(features)

        return {
            "features": features,
            "classification_result": classified["classification_result"],
            "predicted_label": classified["predicted_label"],
            "best_dwg_path": classified["best_dwg_path"],
            "candidates": classified["candidates"],
        }

    def _build_single_prediction(self, classified: Dict[str, Any]) -> Dict[str, Any]:
        classification_result = classified["classification_result"]

        return {
            "predicted_label": classified["predicted_label"],
            "best_dwg_path": classified["best_dwg_path"],
            "top_predictions": classification_result["top_predictions"],
            "top_matches": classification_result["top_matches"],
            "predictions": [item["label"] for item in classified["candidates"]],
            "candidates": classified["candidates"],
        }

    def _save_single_result(
        self,
        sample_dir: Path,
        image: np.ndarray,
        mask: np.ndarray,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        original_suffix: str,
    ) -> None:
        original_output_path = sample_dir / f"original{original_suffix.lower()}"
        mask_output_path = sample_dir / "mask.png"
        features_output_path = sample_dir / "features.json"
        prediction_output_path = sample_dir / "prediction.json"

        cv2.imwrite(str(original_output_path), image)
        cv2.imwrite(str(mask_output_path), mask)
        save_json(features_output_path, features)
        save_json(prediction_output_path, prediction)

    def _process_image_data(
        self,
        image: np.ndarray,
        sample_dir: Path,
        original_suffix: str = ".png",
    ) -> Dict[str, Any]:
        mask = binarize_detail(
            image=image,
            use_hsv_v=self.config.binarization.use_hsv_v,
            blur_ksize=self.config.binarization.blur_ksize,
            threshold_mode=self.config.binarization.threshold_mode,
            fixed_threshold=self.config.binarization.fixed_threshold,
            open_kernel_size=self.config.binarization.open_kernel_size,
            close_kernel_size=self.config.binarization.close_kernel_size,
        )

        classified = self._classify_mask(mask)
        prediction = self._build_single_prediction(classified)

        self._save_single_result(
            sample_dir=sample_dir,
            image=image,
            mask=mask,
            features=classified["features"],
            prediction=prediction,
            original_suffix=original_suffix,
        )

        return prediction

    def _build_object_from_component(
        self,
        component: Dict[str, Any],
        object_id: int,
    ) -> Optional[Dict[str, Any]]:
        cropped_mask, bbox_xyxy = crop_mask_by_bbox(
            mask=component["mask"],
            bbox_xywh=component["bbox_xywh"],
            margin_px=self.config.binarization.crop_margin_px,
        )

        if cv2.countNonZero(cropped_mask) == 0:
            return None

        classified = self._classify_mask(cropped_mask)
        classification_result = classified["classification_result"]

        return {
            "object_id": object_id,
            "bbox_xyxy": list(bbox_xyxy),
            "component_area_px": component["area_px"],
            "mask": cropped_mask,
            "features": classified["features"],
            "predicted_label": classified["predicted_label"],
            "best_dwg_path": classified["best_dwg_path"],
            "candidate_labels": [item["label"] for item in classified["candidates"]],
            "candidates": classified["candidates"],
            "top_predictions": classification_result["top_predictions"],
            "top_matches": classification_result["top_matches"],
        }

    def _build_objects_from_components(
        self,
        components: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []

        for object_id, component in enumerate(components, start=1):
            obj = self._build_object_from_component(
                component=component,
                object_id=object_id,
            )
            if obj is not None:
                objects.append(obj)

        return objects

    def _draw_multi_predictions(
        self,
        image: np.ndarray,
        objects: List[Dict[str, Any]],
    ) -> np.ndarray:
        annotated = image.copy()

        font_scale = image.shape[0] * 0.002
        font_thickness = round(image.shape[0] * 0.003)
        box_thickness = round(image.shape[0] * 0.003)

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            label = obj["predicted_label"] or "uncertain"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

            text = label
            if obj["predicted_label"] is None and obj["candidate_labels"]:
                text = " / ".join(obj["candidate_labels"])

            text_y = y1 - 12
            if text_y < 35:
                text_y = y1 + 35

            cv2.putText(
                annotated,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA,
            )

        return annotated

    def _build_multi_summary(
        self,
        objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        counts = Counter()

        for obj in objects:
            if obj["predicted_label"] is not None:
                counts[obj["predicted_label"]] += 1

        return {
            "num_objects_found": len(objects),
            "counts_by_label": dict(sorted(counts.items(), key=lambda item: item[0])),
            "objects": [
                {
                    "object_id": obj["object_id"],
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "component_area_px": obj["component_area_px"],
                    "predicted_label": obj["predicted_label"],
                    "best_dwg_path": obj["best_dwg_path"],
                    "candidate_labels": obj["candidate_labels"],
                }
                for obj in objects
            ],
        }

    def _save_multi_objects(
        self,
        sample_dir: Path,
        objects: List[Dict[str, Any]],
    ) -> None:
        objects_dir = sample_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)

        for obj in objects:
            object_dir = objects_dir / f"object_{obj['object_id']:02d}"
            object_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(object_dir / "mask.png"), obj["mask"])

            save_json(object_dir / "features.json", obj["features"])
            save_json(
                object_dir / "prediction.json",
                {
                    "object_id": obj["object_id"],
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "component_area_px": obj["component_area_px"],
                    "predicted_label": obj["predicted_label"],
                    "best_dwg_path": obj["best_dwg_path"],
                    "candidate_labels": obj["candidate_labels"],
                    "candidates": obj["candidates"],
                    "top_predictions": obj["top_predictions"],
                    "top_matches": obj["top_matches"],
                },
            )

    def _save_multi_result(
        self,
        sample_dir: Path,
        image: np.ndarray,
        full_mask: np.ndarray,
        annotated: np.ndarray,
        objects: List[Dict[str, Any]],
        summary: Dict[str, Any],
        original_suffix: str,
    ) -> None:
        original_output_path = sample_dir / f"original{original_suffix.lower()}"
        mask_output_path = sample_dir / "mask_all.png"
        annotated_output_path = sample_dir / "annotated.png"
        summary_output_path = sample_dir / "summary.json"

        cv2.imwrite(str(original_output_path), image)
        cv2.imwrite(str(mask_output_path), full_mask)
        cv2.imwrite(str(annotated_output_path), annotated)

        self._save_multi_objects(sample_dir, objects)
        save_json(summary_output_path, summary)

    def _process_image_data_multi(
        self,
        image: np.ndarray,
        sample_dir: Path,
        original_suffix: str = ".png",
    ) -> Dict[str, Any]:
        full_mask, components = binarize_details_mask(
            image=image,
            use_hsv_v=self.config.binarization.use_hsv_v,
            blur_ksize=self.config.binarization.blur_ksize,
            threshold_mode=self.config.binarization.threshold_mode,
            fixed_threshold=self.config.binarization.fixed_threshold,
            open_kernel_size=self.config.binarization.open_kernel_size,
            close_kernel_size=self.config.binarization.close_kernel_size,
            min_component_area_px=self.config.binarization.min_component_area_px,
        )

        objects = self._build_objects_from_components(components)
        annotated = self._draw_multi_predictions(image, objects)
        summary = self._build_multi_summary(objects)

        self._save_multi_result(
            sample_dir=sample_dir,
            image=image,
            full_mask=full_mask,
            annotated=annotated,
            objects=objects,
            summary=summary,
            original_suffix=original_suffix,
        )

        return summary

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        image_path = Path(image_path)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        sample_dir = self._prepare_output_dir(image_path)
        return self._process_image_data(
            image=image,
            sample_dir=sample_dir,
            original_suffix=image_path.suffix or ".png",
        )

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None:
            raise ValueError("frame is None")

        sample_name = self._make_timestamp_name()
        sample_dir = self._prepare_output_dir_by_name(sample_name)

        return self._process_image_data(
            image=frame,
            sample_dir=sample_dir,
            original_suffix=".png",
        )

    def process_image_multi(self, image_path: Path) -> Dict[str, Any]:
        image_path = Path(image_path)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        sample_dir = self._prepare_output_dir(image_path)
        return self._process_image_data_multi(
            image=image,
            sample_dir=sample_dir,
            original_suffix=image_path.suffix or ".png",
        )

    def process_frame_multi(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None:
            raise ValueError("frame is None")

        sample_name = self._make_timestamp_name()
        sample_dir = self._prepare_output_dir_by_name(sample_name)

        return self._process_image_data_multi(
            image=frame,
            sample_dir=sample_dir,
            original_suffix=".png",
        )

    def process_input_dir(self) -> List[Dict[str, Any]]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.input_dir}")

        image_paths = [
            p
            for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.allowed_exts
        ]
        image_paths.sort(key=lambda p: p.name)

        results = []
        for image_path in image_paths:
            results.append(self.process_image(image_path))

        return results

    def process_input_dir_multi(self) -> List[Dict[str, Any]]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.input_dir}")

        image_paths = [
            p
            for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.allowed_exts
        ]
        image_paths.sort(key=lambda p: p.name)

        results = []
        for image_path in image_paths:
            results.append(self.process_image_multi(image_path))

        return results