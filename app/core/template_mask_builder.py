from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from app.core.dxf_geometry import (
    Path2D,
    compute_bounds,
    compute_dxf_dimensions,
    load_paths_from_dxf,
    world_to_image_transform,
)
from app.core.feature_extraction import (
    extract_features_from_mask,
    find_external_and_holes,
)


def rasterize_paths_to_mask(
    paths: List[Path2D],
    canvas_size: int = 5000,
    padding: int = 40,
    line_thickness: int = 1,
) -> np.ndarray:
    bounds = compute_bounds(paths)
    transform = world_to_image_transform(
        bounds=bounds,
        canvas_size=canvas_size,
        padding=padding,
    )

    contour_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    closed_contours = []
    open_paths = []

    for path in paths:
        if len(path) < 2:
            continue

        pts = [transform(p) for p in path]
        arr = np.array(pts, dtype=np.int32)

        is_closed = np.linalg.norm(arr[0] - arr[-1]) <= 2

        if is_closed and len(arr) >= 3:
            closed_contours.append(arr)
        else:
            open_paths.append(arr)

    for arr in closed_contours:
        cv2.polylines(
            contour_mask,
            [arr],
            isClosed=True,
            color=255,
            thickness=line_thickness,
        )

    for arr in open_paths:
        cv2.polylines(
            contour_mask,
            [arr],
            isClosed=False,
            color=255,
            thickness=line_thickness,
        )

    kernel = np.ones((3, 3), np.uint8)
    contour_mask = cv2.morphologyEx(
        contour_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    inv = cv2.bitwise_not(contour_mask)

    flood = inv.copy()
    h, w = flood.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=100)

    filled_regions = np.zeros_like(flood)
    filled_regions[flood == 255] = 255

    contours, hierarchy = cv2.findContours(
        filled_regions,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours or hierarchy is None:
        raise ValueError("Failed to find filled contours.")

    hierarchy = hierarchy[0]
    final_mask = np.zeros_like(filled_regions)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area <= 1:
            continue

        parent = hierarchy[i][3]

        color = 255 if parent == -1 else 0
        cv2.drawContours(final_mask, [cnt], -1, color, thickness=cv2.FILLED)

    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    return final_mask


def build_mask_from_dxf(
    dxf_path: Path,
    output_mask_path: Path | None = None,
    canvas_size: int = 5000,
    padding: int = 40,
    line_thickness: int = 3,
):
    paths = load_paths_from_dxf(dxf_path)

    mask = rasterize_paths_to_mask(
        paths=paths,
        canvas_size=canvas_size,
        padding=padding,
        line_thickness=line_thickness,
    )

    if output_mask_path is not None:
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_mask_path), mask)

    return mask, paths


def compute_mm_per_px_from_dxf_and_mask(
    mask: np.ndarray,
    dxf_width_mm: float,
    min_hole_area_px: int = 20,
) -> float:
    external_contour, _ = find_external_and_holes(
        mask,
        min_hole_area_px=min_hole_area_px,
    )

    rect = cv2.minAreaRect(external_contour)
    rect_w_px, rect_h_px = rect[1]

    width_px = max(float(rect_w_px), float(rect_h_px))
    if width_px <= 1e-8:
        raise ValueError("Failed to compute part width in pixels from the mask.")

    return dxf_width_mm / width_px


def extract_features_from_dxf_and_mask(
    paths: List[Path2D],
    mask: np.ndarray,
    min_hole_area_px: int = 20,
    camera_id: str = "cam_01",
    material: str | None = None,
    thickness_mm: float | None = None,
) -> Dict[str, Any]:
    dxf_dimensions = compute_dxf_dimensions(paths)

    mm_per_px = compute_mm_per_px_from_dxf_and_mask(
        mask=mask,
        dxf_width_mm=dxf_dimensions["width_mm"],
        min_hole_area_px=min_hole_area_px,
    )

    features = extract_features_from_mask(
        mask=mask,
        mm_per_px=mm_per_px,
        min_hole_area_px=min_hole_area_px,
        camera_id=camera_id,
        material=material,
        thickness_mm=thickness_mm,
    )

    features["geometry"]["dimensions"]["width_mm"] = round(
        dxf_dimensions["width_mm"], 4
    )
    features["geometry"]["dimensions"]["height_mm"] = round(
        dxf_dimensions["height_mm"], 4
    )
    features["geometry"]["dimensions"]["mm_per_px"] = round(mm_per_px, 8)

    return features
