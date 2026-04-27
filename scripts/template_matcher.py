import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.configs.config import AppConfig
from app.core.factories import build_template_classifier


def load_features_file(sample_dir: Path) -> Optional[Path]:
    features_path = sample_dir / "features.json"
    if features_path.exists():
        return features_path
    return None


def load_true_label(sample_dir: Path) -> Optional[str]:
    label_path = sample_dir / "label.txt"
    if not label_path.exists():
        return None

    label = label_path.read_text(encoding="utf-8").strip()
    return label if label else None


def score_all_templates(
    classifier,
    vector: np.ndarray,
) -> List[Dict[str, Any]]:
    distances = classifier._compute_distances_to_matrix(vector)
    sorted_indices = np.argsort(distances)

    scored: List[Dict[str, Any]] = []
    for idx in sorted_indices:
        scored.append(
            {
                "label": classifier.labels[idx],
                "distance": float(distances[idx]),
                "dwg_path": classifier.dwg_paths[idx],
            }
        )

    return scored


def compute_margin_metrics(
    all_matches: List[Dict[str, Any]],
    true_label: Optional[str],
) -> Dict[str, Optional[float]]:
    if true_label is None:
        return {
            "true_distance": None,
            "nearest_wrong_distance": None,
            "template_margin": None,
            "relative_margin": None,
        }

    true_match = None
    for item in all_matches:
        if item["label"] == true_label:
            true_match = item
            break

    if true_match is None:
        return {
            "true_distance": None,
            "nearest_wrong_distance": None,
            "template_margin": None,
            "relative_margin": None,
        }

    nearest_wrong = None
    for item in all_matches:
        if item["label"] != true_label:
            nearest_wrong = item
            break

    true_distance = float(true_match["distance"])
    nearest_wrong_distance = (
        float(nearest_wrong["distance"]) if nearest_wrong is not None else None
    )

    template_margin = None
    relative_margin = None

    if nearest_wrong_distance is not None:
        template_margin = nearest_wrong_distance - true_distance
        relative_margin = template_margin / (true_distance + 1e-12)

    return {
        "true_distance": true_distance,
        "nearest_wrong_distance": nearest_wrong_distance,
        "template_margin": template_margin,
        "relative_margin": relative_margin,
    }


def evaluate_outputs_with_template_matching(
    outputs_dir: str | Path = "data/output",
    report_path: str | Path = "data/output/template_matching_report_weighted.json",
    top_k: int = 3,
) -> Dict[str, Any]:
    config = AppConfig()

    outputs_dir = Path(outputs_dir)
    report_path = Path(report_path)

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {outputs_dir}")

    classifier = build_template_classifier(config)
    classifier.load_cache()

    sample_dirs = [p for p in outputs_dir.iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: p.name)

    if not sample_dirs:
        raise FileNotFoundError(f"No sample subdirectories found in {outputs_dir}")

    results: List[Dict[str, Any]] = []

    total_processed = 0
    samples_with_label = 0
    top1_correct = 0
    topk_correct = 0

    template_margins: List[float] = []
    relative_margins: List[float] = []
    true_distances: List[float] = []
    nearest_wrong_distances: List[float] = []

    for sample_dir in sample_dirs:
        features_path = load_features_file(sample_dir)
        if features_path is None:
            print(f"Skipping {sample_dir.name}: features.json not found")
            continue

        true_label = load_true_label(sample_dir)

        try:
            with open(features_path, "r", encoding="utf-8") as f:
                features_json = json.load(f)

            vector = classifier._json_to_vector(features_json)
            all_matches = score_all_templates(classifier, vector)
            top_matches = all_matches[:top_k]

        except Exception as e:
            print(f"Matching error for {sample_dir.name}: {e}")
            results.append(
                {
                    "sample": sample_dir.name,
                    "features_path": str(features_path),
                    "true_label": true_label,
                    "error": str(e),
                }
            )
            continue

        total_processed += 1

        top_labels = [item["label"] for item in top_matches]
        best_label = top_labels[0] if top_labels else None

        is_top1_correct = None
        is_topk_correct = None

        if true_label is not None:
            samples_with_label += 1
            is_top1_correct = best_label == true_label
            is_topk_correct = true_label in top_labels[:top_k]

            if is_top1_correct:
                top1_correct += 1
            if is_topk_correct:
                topk_correct += 1

        distance_gap_1_2 = None
        if len(top_matches) >= 2:
            distance_gap_1_2 = top_matches[1]["distance"] - top_matches[0]["distance"]

        margin_info = compute_margin_metrics(
            all_matches=all_matches,
            true_label=true_label,
        )

        if margin_info["template_margin"] is not None:
            template_margins.append(margin_info["template_margin"])

        if margin_info["relative_margin"] is not None:
            relative_margins.append(margin_info["relative_margin"])

        if margin_info["true_distance"] is not None:
            true_distances.append(margin_info["true_distance"])

        if margin_info["nearest_wrong_distance"] is not None:
            nearest_wrong_distances.append(margin_info["nearest_wrong_distance"])

        results.append(
            {
                "sample": sample_dir.name,
                "features_path": str(features_path),
                "true_label": true_label,
                "top1_prediction": best_label,
                "topk_predictions": top_labels[:top_k],
                "is_top1_correct": is_top1_correct,
                "is_topk_correct": is_topk_correct,
                "distance_gap_1_2": distance_gap_1_2,
                "true_distance": margin_info["true_distance"],
                "nearest_wrong_distance": margin_info["nearest_wrong_distance"],
                "template_margin": margin_info["template_margin"],
                "relative_margin": margin_info["relative_margin"],
                "top_matches": top_matches,
            }
        )

    top1_accuracy = (
        top1_correct / samples_with_label if samples_with_label > 0 else None
    )
    topk_accuracy = (
        topk_correct / samples_with_label if samples_with_label > 0 else None
    )

    mean_true_distance = float(np.mean(true_distances)) if true_distances else None
    mean_nearest_wrong_distance = (
        float(np.mean(nearest_wrong_distances)) if nearest_wrong_distances else None
    )

    mean_template_margin = (
        float(np.mean(template_margins)) if template_margins else None
    )
    median_template_margin = (
        float(np.median(template_margins)) if template_margins else None
    )
    min_template_margin = float(np.min(template_margins)) if template_margins else None

    mean_relative_margin = (
        float(np.mean(relative_margins)) if relative_margins else None
    )
    median_relative_margin = (
        float(np.median(relative_margins)) if relative_margins else None
    )
    min_relative_margin = float(np.min(relative_margins)) if relative_margins else None

    summary = {
        "outputs_dir": str(outputs_dir),
        "templates_dir": str(config.paths.templates_dir),
        "metric": config.classification.metric,
        "normalized": config.classification.normalize,
        "weights": classifier.feature_weights,
        "total_processed": total_processed,
        "samples_with_true_label": samples_with_label,
        "top1_correct": top1_correct,
        f"top{top_k}_correct": topk_correct,
        "top1_accuracy": top1_accuracy,
        f"top{top_k}_accuracy": topk_accuracy,
        "mean_true_distance": mean_true_distance,
        "mean_nearest_wrong_distance": mean_nearest_wrong_distance,
        "mean_template_margin": mean_template_margin,
        "median_template_margin": median_template_margin,
        "min_template_margin": min_template_margin,
        "mean_relative_margin": mean_relative_margin,
        "median_relative_margin": median_relative_margin,
        "min_relative_margin": min_relative_margin,
        "results": results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("WEIGHTED TEMPLATE MATCHING RESULTS")
    print("=" * 60)
    print(f"Processed: {total_processed}")
    print(f"With true label: {samples_with_label}")
    print(f"Metric: {config.classification.metric}")
    print(f"Normalization: {config.classification.normalize}")

    if samples_with_label > 0:
        print(f"Top-1 accuracy: {top1_accuracy:.2%}")
        print(f"Top-{top_k} accuracy: {topk_accuracy:.2%}")

    if mean_template_margin is not None:
        print(f"Mean true distance: {mean_true_distance:.6f}")
        print(f"Mean nearest wrong distance: {mean_nearest_wrong_distance:.6f}")
        print(f"Mean template margin: {mean_template_margin:.6f}")
        print(f"Median template margin: {median_template_margin:.6f}")
        print(f"Min template margin: {min_template_margin:.6f}")
        print(f"Mean relative margin: {mean_relative_margin:.6f}")
        print(f"Median relative margin: {median_relative_margin:.6f}")
        print(f"Min relative margin: {min_relative_margin:.6f}")

    print(f"\nReport saved to: {report_path}")

    return summary


if __name__ == "__main__":
    evaluate_outputs_with_template_matching(
        outputs_dir="data/output_for_metric",
        report_path="data/output_for_metric/template_matching_report_weighted.json",
        top_k=3,
    )