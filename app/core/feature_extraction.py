import cv2
import numpy as np
import math
from datetime import datetime, timezone


def find_external_and_holes(mask, min_hole_area_px=0):
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours or hierarchy is None:
        raise ValueError("Contours were not found")

    hierarchy = hierarchy[0]

    external_idx = None
    for i in range(len(hierarchy)):
        if hierarchy[i][3] == -1:
            external_idx = i
            break

    if external_idx is None:
        raise ValueError("External contour was not found")

    external_contour = contours[external_idx]

    hole_contours = []
    child_idx = hierarchy[external_idx][2]

    while child_idx != -1:
        contour = contours[child_idx]
        if cv2.contourArea(contour) >= min_hole_area_px:
            hole_contours.append(contour)
        child_idx = hierarchy[child_idx][0]

    return external_contour, hole_contours


def get_contour_center(contour):
    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-8:
        points = contour.reshape(-1, 2)
        return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return cx, cy


def compute_equivalent_diameter_mm(area_px, mm_per_px):
    if area_px <= 0:
        return 0.0

    diameter_px = math.sqrt(4.0 * area_px / math.pi)
    return diameter_px * mm_per_px


def compute_hu_moments(contour):
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    result = []
    for value in hu:
        if abs(value) < 1e-30:
            result.append(0.0)
        else:
            result.append(float(-np.sign(value) * np.log10(abs(value))))

    return result


def compute_contour_signature(
    contour,
    center,
    n_samples=36,
):
    cx, cy = center
    points = contour.reshape(-1, 2).astype(np.float32)

    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    radii = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)

    sample_angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    signature = []

    for angle in sample_angles:
        angular_diff = np.abs(angles - angle)
        angular_diff = np.minimum(angular_diff, 2 * np.pi - angular_diff)

        nearest_idx = int(np.argmin(angular_diff))
        signature.append(float(radii[nearest_idx]))

    max_radius = max(signature) if signature else 0.0
    if max_radius > 1e-8:
        signature = [value / max_radius for value in signature]

    return signature


def compute_fourier_descriptors(contour, n_descriptors=10):
    points = contour.reshape(-1, 2).astype(np.float32)
    complex_points = points[:, 0] + 1j * points[:, 1]

    fft_coeffs = np.fft.fft(complex_points)

    if len(fft_coeffs) < 2:
        return [0.0] * n_descriptors

    fft_coeffs = fft_coeffs[1 : 1 + n_descriptors]

    if len(fft_coeffs) == 0:
        return [0.0] * n_descriptors

    norm = np.abs(fft_coeffs[0])
    if norm < 1e-12:
        norm = 1.0

    descriptors = (np.abs(fft_coeffs) / norm).real.tolist()

    if len(descriptors) < n_descriptors:
        descriptors.extend([0.0] * (n_descriptors - len(descriptors)))

    return [float(value) for value in descriptors[:n_descriptors]]


def compute_dimensions(external_contour, hole_contours, mm_per_px):
    external_area_px = float(cv2.contourArea(external_contour))
    holes_areas_px = [float(cv2.contourArea(contour)) for contour in hole_contours]
    holes_area_total_px = float(sum(holes_areas_px))
    material_area_px = max(external_area_px - holes_area_total_px, 0.0)

    perimeter_px = float(cv2.arcLength(external_contour, True))

    rect = cv2.minAreaRect(external_contour)
    rect_w_px, rect_h_px = rect[1]
    rect_w_px = float(rect_w_px)
    rect_h_px = float(rect_h_px)

    width_px = max(rect_w_px, rect_h_px)
    height_px = min(rect_w_px, rect_h_px)

    bbox_ratio = width_px / height_px if height_px > 1e-8 else 0.0
    bbox_area_px = width_px * height_px
    extent = external_area_px / bbox_area_px if bbox_area_px > 1e-8 else 0.0

    return {
        "width_mm": width_px * mm_per_px,
        "height_mm": height_px * mm_per_px,
        "bbox_ratio": bbox_ratio,
        "area_mm2": material_area_px * (mm_per_px**2),
        "perimeter_mm": perimeter_px * mm_per_px,
        "extent": extent,
        "external_area_px": external_area_px,
        "material_area_px": material_area_px,
        "perimeter_px": perimeter_px,
        "width_px": width_px,
        "height_px": height_px,
    }


def compute_topology(external_contour, hole_contours, mm_per_px, external_area_px):
    holes_areas_px = [float(cv2.contourArea(contour)) for contour in hole_contours]
    holes_area_total_px = float(sum(holes_areas_px))

    hole_diameters_mm = [
        compute_equivalent_diameter_mm(area_px, mm_per_px) for area_px in holes_areas_px
    ]
    hole_diameters_sorted_mm = sorted(float(value) for value in hole_diameters_mm)

    detail_cx, detail_cy = get_contour_center(external_contour)

    distances_px = []
    for contour in hole_contours:
        hole_cx, hole_cy = get_contour_center(contour)
        distance_px = math.hypot(hole_cx - detail_cx, hole_cy - detail_cy)
        distances_px.append(distance_px)

    distances_mm = [float(value) * mm_per_px for value in distances_px]
    hole_distances_sorted_mm = sorted(distances_mm)

    holes_area_ratio = (
        holes_area_total_px / external_area_px if external_area_px > 1e-8 else 0.0
    )

    return {
        "holes_count": len(hole_contours),
        "holes_area_total_mm2": holes_area_total_px * (mm_per_px**2),
        "holes_area_ratio": holes_area_ratio,
        "max_hole_diameter_mm": max(hole_diameters_mm) if hole_diameters_mm else 0.0,
        "min_hole_diameter_mm": min(hole_diameters_mm) if hole_diameters_mm else 0.0,
        "mean_hole_diameter_mm": (
            float(np.mean(hole_diameters_mm)) if hole_diameters_mm else 0.0
        ),
        "std_hole_diameter_mm": (
            float(np.std(hole_diameters_mm)) if hole_diameters_mm else 0.0
        ),
        "avg_dist_to_center_mm": float(np.mean(distances_mm)) if distances_mm else 0.0,
        "min_hole_dist_to_center_mm": min(distances_mm) if distances_mm else 0.0,
        "max_hole_dist_to_center_mm": max(distances_mm) if distances_mm else 0.0,
        "std_hole_dist_to_center_mm": (
            float(np.std(distances_mm)) if distances_mm else 0.0
        ),
        "hole_diameters_sorted_mm": hole_diameters_sorted_mm,
        "hole_distances_sorted_mm": hole_distances_sorted_mm,
    }


def compute_morphology(
    external_contour,
    external_area_px,
    material_area_px,
    perimeter_px,
):
    compactness = (
        (perimeter_px**2) / material_area_px if material_area_px > 1e-8 else 0.0
    )

    hull = cv2.convexHull(external_contour)
    hull_area_px = float(cv2.contourArea(hull))
    hull_perimeter_px = float(cv2.arcLength(hull, True))

    solidity = external_area_px / hull_area_px if hull_area_px > 1e-8 else 0.0

    circularity = (
        (4.0 * math.pi * external_area_px) / (perimeter_px**2)
        if perimeter_px > 1e-8
        else 0.0
    )

    convexity = hull_perimeter_px / perimeter_px if perimeter_px > 1e-8 else 0.0

    points = external_contour.reshape(-1, 2).astype(np.float32)
    eccentricity = 0.0
    if len(points) >= 5:
        (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(external_contour)
        major_axis = max(float(major_axis), float(minor_axis))
        minor_axis = (
            min(float(major_axis), float(minor_axis))
            if major_axis != minor_axis
            else min(float(major_axis), float(minor_axis))
        )
        if major_axis > 1e-8:
            ratio = 1.0 - (minor_axis**2) / (major_axis**2)
            ratio = max(0.0, min(1.0, ratio))
            eccentricity = math.sqrt(ratio)

    center = get_contour_center(external_contour)
    contour_signature = compute_contour_signature(
        contour=external_contour,
        center=center,
        n_samples=36,
    )
    fourier_descriptors = compute_fourier_descriptors(
        contour=external_contour,
        n_descriptors=10,
    )
    hu_moments = compute_hu_moments(external_contour)

    return {
        "compactness": compactness,
        "solidity": solidity,
        "circularity": circularity,
        "convexity": convexity,
        "eccentricity": eccentricity,
        "hu_moments": hu_moments,
        "contour_signature": contour_signature,
        "fourier_descriptors": fourier_descriptors,
    }


def build_result(
    dimensions,
    topology,
    morphology,
    camera_id="cam_01",
    timestamp=None,
    material=None,
    thickness_mm=None,
):
    if timestamp is None:
        timestamp = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    return {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "geometry": {
            "dimensions": {
                "width_mm": round(dimensions["width_mm"], 4),
                "height_mm": round(dimensions["height_mm"], 4),
                "bbox_ratio": round(dimensions["bbox_ratio"], 6),
                "area_mm2": round(dimensions["area_mm2"], 4),
                "perimeter_mm": round(dimensions["perimeter_mm"], 4),
                "extent": round(dimensions["extent"], 6),
            },
            "topology": {
                "holes_count": int(topology["holes_count"]),
                "holes_area_total_mm2": round(topology["holes_area_total_mm2"], 4),
                "holes_area_ratio": round(topology["holes_area_ratio"], 6),
                "max_hole_diameter_mm": round(topology["max_hole_diameter_mm"], 4),
                "min_hole_diameter_mm": round(topology["min_hole_diameter_mm"], 4),
                "mean_hole_diameter_mm": round(topology["mean_hole_diameter_mm"], 4),
                "std_hole_diameter_mm": round(topology["std_hole_diameter_mm"], 4),
                "avg_dist_to_center_mm": round(topology["avg_dist_to_center_mm"], 4),
                "min_hole_dist_to_center_mm": round(
                    topology["min_hole_dist_to_center_mm"], 4
                ),
                "max_hole_dist_to_center_mm": round(
                    topology["max_hole_dist_to_center_mm"], 4
                ),
                "std_hole_dist_to_center_mm": round(
                    topology["std_hole_dist_to_center_mm"], 4
                ),
                "hole_diameters_sorted_mm": [
                    round(v, 4) for v in topology["hole_diameters_sorted_mm"]
                ],
                "hole_distances_sorted_mm": [
                    round(v, 4) for v in topology["hole_distances_sorted_mm"]
                ],
            },
            "morphology": {
                "compactness": round(morphology["compactness"], 6),
                "solidity": round(morphology["solidity"], 6),
                "circularity": round(morphology["circularity"], 6),
                "convexity": round(morphology["convexity"], 6),
                "eccentricity": round(morphology["eccentricity"], 6),
                "hu_moments": [round(value, 10) for value in morphology["hu_moments"]],
                "contour_signature": [
                    round(value, 6) for value in morphology["contour_signature"]
                ],
                "fourier_descriptors": [
                    round(value, 6) for value in morphology["fourier_descriptors"]
                ],
            },
        },
        "context": {
            "material": material,
            "thickness_mm": thickness_mm,
        },
    }


def extract_features_from_mask(
    mask,
    mm_per_px,
    min_hole_area_px=20,
    camera_id="cam_01",
    timestamp=None,
    material=None,
    thickness_mm=None,
):
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be greater than 0")

    external_contour, hole_contours = find_external_and_holes(
        mask,
        min_hole_area_px=min_hole_area_px,
    )

    dimensions = compute_dimensions(external_contour, hole_contours, mm_per_px)

    topology = compute_topology(
        external_contour=external_contour,
        hole_contours=hole_contours,
        mm_per_px=mm_per_px,
        external_area_px=dimensions["external_area_px"],
    )

    morphology = compute_morphology(
        external_contour=external_contour,
        external_area_px=dimensions["external_area_px"],
        material_area_px=dimensions["material_area_px"],
        perimeter_px=dimensions["perimeter_px"],
    )

    return build_result(
        dimensions=dimensions,
        topology=topology,
        morphology=morphology,
        camera_id=camera_id,
        timestamp=timestamp,
        material=material,
        thickness_mm=thickness_mm,
    )
