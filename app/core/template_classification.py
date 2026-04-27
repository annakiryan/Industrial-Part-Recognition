import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class TemplateClassifier:
    def __init__(
        self,
        templates_dir: Path,
        cache_dir: Path,
        template_cache_meta: Path,
        cache_npz_path: Path,
        template_features_config: Path,
        metric: str = "manhattan",
        normalize: bool = True,
    ):
        self.templates_dir = Path(templates_dir)
        self.metric = metric
        self.normalize = normalize

        self.cache_dir = Path(cache_dir)
        self.cache_npz_path = Path(cache_npz_path)
        self.cache_meta_path = Path(template_cache_meta)
        self.template_features_config_path = Path(template_features_config)

        self.labels: List[str] = []
        self.dwg_paths: List[Optional[str]] = []

        self.X_templates: Optional[np.ndarray] = None
        self.X_templates_norm: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

        self.feature_defs = self._load_feature_defs()
        self.feature_order = [item["name"] for item in self.feature_defs]
        self.feature_weights = {
            item["name"]: float(item["weight"]) for item in self.feature_defs
        }
        self.weights_vector = np.array(
            [self.feature_weights[name] for name in self.feature_order],
            dtype=np.float64,
        )

    def _load_feature_defs(self) -> List[Dict[str, Any]]:
        if not self.template_features_config_path.exists():
            raise FileNotFoundError(
                f"Template features config not found: "
                f"{self.template_features_config_path}"
            )

        with open(self.template_features_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        feature_defs = config.get("features")
        if not isinstance(feature_defs, list) or not feature_defs:
            raise ValueError(
                "template_features_config must contain non-empty 'features' list"
            )

        required_keys = {"name", "path", "weight"}
        for item in feature_defs:
            if not required_keys.issubset(item.keys()):
                raise ValueError(
                    "Each feature config item must contain: "
                    "'name', 'path', 'weight'"
                )

        return feature_defs

    def load_cache(self) -> None:
        if not self.cache_npz_path.exists() or not self.cache_meta_path.exists():
            raise FileNotFoundError(
                f"Template cache was not found: {self.cache_npz_path}"
            )
        self._load_cache()

    def _get_by_path(self, data: Any, path: List[Any]) -> Any:
        value = data
        for part in path:
            value = value[part]
        return value

    def _json_to_vector(self, data: Dict[str, Any]) -> np.ndarray:
        values = [
            self._get_by_path(data, feature_def["path"])
            for feature_def in self.feature_defs
        ]
        return np.array(values, dtype=np.float64)

    def _recompute_normalization(self) -> None:
        if self.X_templates is None or len(self.X_templates) == 0:
            self.mean_ = None
            self.std_ = None
            self.X_templates_norm = None
            return

        self.mean_ = self.X_templates.mean(axis=0)
        self.std_ = self.X_templates.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        self.X_templates_norm = (self.X_templates - self.mean_) / self.std_

    def _save_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            self.cache_npz_path,
            X_templates=self.X_templates,
            X_templates_norm=(
                self.X_templates_norm
                if self.X_templates_norm is not None
                else np.array([])
            ),
            mean=self.mean_ if self.mean_ is not None else np.array([]),
            std=self.std_ if self.std_ is not None else np.array([]),
        )

        meta = {
            "labels": self.labels,
            "dwg_paths": self.dwg_paths,
            "metric": self.metric,
            "normalize": self.normalize,
            "template_features_config": str(self.template_features_config_path),
        }

        with open(self.cache_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_cache(self) -> None:
        data = np.load(self.cache_npz_path, allow_pickle=True)

        self.X_templates = data["X_templates"]

        x_norm = data["X_templates_norm"]
        self.X_templates_norm = x_norm if x_norm.size > 0 else None

        mean = data["mean"]
        std = data["std"]
        self.mean_ = mean if mean.size > 0 else None
        self.std_ = std if std.size > 0 else None

        with open(self.cache_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.labels = meta["labels"]
        self.dwg_paths = meta["dwg_paths"]

    def update_template_matrix(self) -> int:
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_npz_path.exists() and self.cache_meta_path.exists():
            self._load_cache()
        else:
            self.X_templates = None
            self.X_templates_norm = None
            self.mean_ = None
            self.std_ = None
            self.labels = []
            self.dwg_paths = []

        existing_labels = set(self.labels)
        added_count = 0

        template_dirs = [p for p in self.templates_dir.iterdir() if p.is_dir()]
        template_dirs.sort(key=lambda p: p.name)

        for template_dir in template_dirs:
            label = template_dir.name

            if label == "_cache":
                continue

            if label in existing_labels:
                continue

            features_path = template_dir / "features.json"
            if not features_path.exists():
                print(f"Skipping {label}: features.json not found")
                continue

            with open(features_path, "r", encoding="utf-8") as f:
                features_json = json.load(f)

            vector = self._json_to_vector(features_json)

            if self.X_templates is None:
                self.X_templates = vector[None, :]
            else:
                self.X_templates = np.vstack([self.X_templates, vector])

            dwg_path = template_dir / "part.dwg"

            self.labels.append(label)
            self.dwg_paths.append(str(dwg_path) if dwg_path.exists() else None)

            existing_labels.add(label)
            added_count += 1

        if self.X_templates is None or len(self.labels) == 0:
            raise FileNotFoundError("No templates were added to the matrix")

        if self.normalize:
            self._recompute_normalization()
        else:
            self.X_templates_norm = None
            self.mean_ = None
            self.std_ = None

        self._save_cache()
        return added_count

    def rebuild_template_matrix(self) -> int:
        if self.cache_npz_path.exists():
            self.cache_npz_path.unlink()

        if self.cache_meta_path.exists():
            self.cache_meta_path.unlink()

        self.X_templates = None
        self.X_templates_norm = None
        self.mean_ = None
        self.std_ = None
        self.labels = []
        self.dwg_paths = []

        return self.update_template_matrix()

    def _compute_distances_to_matrix(self, query_vector: np.ndarray) -> np.ndarray:
        if self.X_templates is None or len(self.X_templates) == 0:
            raise ValueError("Template matrix is empty")

        if self.normalize:
            if self.mean_ is None or self.std_ is None or self.X_templates_norm is None:
                raise ValueError("Normalization data is not initialized")
            query_vector = (query_vector - self.mean_) / self.std_
            X = self.X_templates_norm
        else:
            X = self.X_templates

        if self.metric == "manhattan":
            diff = np.abs(X - query_vector) * self.weights_vector
            distances = diff.sum(axis=1)
            return distances.astype(np.float64)

        if self.metric == "euclidean":
            diff = (X - query_vector) * self.weights_vector
            distances = np.sqrt(np.sum(diff**2, axis=1))
            return distances.astype(np.float64)

        if self.metric == "cosine":
            X_weighted = X * self.weights_vector
            query_weighted = query_vector * self.weights_vector

            query_norm = np.linalg.norm(query_weighted)
            X_norms = np.linalg.norm(X_weighted, axis=1)

            distances = np.ones(X.shape[0], dtype=np.float64)

            valid = (X_norms > 1e-12) & (query_norm > 1e-12)
            if np.any(valid):
                similarities = np.sum(X_weighted[valid] * query_weighted, axis=1) / (
                    X_norms[valid] * query_norm
                )
                distances[valid] = 1.0 - similarities

            return distances

        raise ValueError("metric must be 'manhattan', 'euclidean', or 'cosine'")

    def classify_vector(self, vector: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        distances = self._compute_distances_to_matrix(vector)
        sorted_indices = np.argsort(distances)
        top_indices = sorted_indices[:top_k]

        top_matches = []
        for idx in top_indices:
            top_matches.append(
                {
                    "label": self.labels[idx],
                    "distance": float(distances[idx]),
                    "dwg_path": self.dwg_paths[idx],
                }
            )

        predicted_label = top_matches[0]["label"] if top_matches else None
        top_predictions = [item["label"] for item in top_matches]

        distance_gap_1_2 = None
        if len(top_matches) >= 2:
            distance_gap_1_2 = top_matches[1]["distance"] - top_matches[0]["distance"]

        return {
            "predicted_label": predicted_label,
            "top_predictions": top_predictions,
            "top_matches": top_matches,
            "distance_gap_1_2": distance_gap_1_2,
            "metric": self.metric,
            "normalized": self.normalize,
        }

    def classify_features_json(
        self,
        features_json: Dict[str, Any],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        vector = self._json_to_vector(features_json)
        return self.classify_vector(vector, top_k=top_k)

    def classify_features_file(
        self,
        features_path: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        features_path = Path(features_path)

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        with open(features_path, "r", encoding="utf-8") as f:
            features_json = json.load(f)

        result = self.classify_features_json(features_json, top_k=top_k)
        result["query_features_path"] = str(features_path)
        return result