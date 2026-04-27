import math
from pathlib import Path
from typing import Callable, List, Tuple

import ezdxf
import numpy as np


SUPPORTED_TYPES = {
    "LINE",
    "LWPOLYLINE",
    "POLYLINE",
    "CIRCLE",
    "ARC",
    "ELLIPSE",
    "SPLINE",
}


Point2D = Tuple[float, float]
PixelPoint = Tuple[int, int]
Path2D = List[Point2D]


def flatten_points(points_3d) -> Path2D:
    return [(float(p[0]), float(p[1])) for p in points_3d]


def sample_arc(
    center: Point2D,
    radius: float,
    start_angle_deg: float,
    end_angle_deg: float,
    num: int = 180,
) -> Path2D:
    start = math.radians(start_angle_deg)
    end = math.radians(end_angle_deg)

    if end < start:
        end += 2 * math.pi

    angles = np.linspace(start, end, num)
    cx, cy = center

    pts = []
    for angle in angles:
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        pts.append((x, y))

    return pts


def sample_ellipse(
    center: Point2D,
    major_axis: Point2D,
    ratio: float,
    start_param: float,
    end_param: float,
    num: int = 240,
) -> Path2D:
    cx, cy = center
    mx, my = major_axis

    major_len = math.hypot(mx, my)
    if major_len == 0:
        return []

    ux, uy = mx / major_len, my / major_len
    vx, vy = -uy, ux
    minor_len = major_len * ratio

    if end_param < start_param:
        end_param += 2 * math.pi

    params = np.linspace(start_param, end_param, num)
    pts = []

    for t in params:
        x = cx + major_len * math.cos(t) * ux + minor_len * math.sin(t) * vx
        y = cy + major_len * math.cos(t) * uy + minor_len * math.sin(t) * vy
        pts.append((x, y))

    return pts


def sample_spline(entity, num: int = 300) -> Path2D:
    try:
        tool = entity.construction_tool()
        pts = list(tool.approximate(num))
        return flatten_points(pts)
    except Exception:
        return []


def collect_entity_paths(doc) -> List[Path2D]:
    msp = doc.modelspace()
    all_paths: List[Path2D] = []

    for entity in msp:
        dxftype = entity.dxftype()

        if dxftype not in SUPPORTED_TYPES:
            continue

        try:
            if dxftype == "LINE":
                p1 = entity.dxf.start
                p2 = entity.dxf.end
                all_paths.append([(p1.x, p1.y), (p2.x, p2.y)])

            elif dxftype == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in entity.get_points()]
                if len(pts) >= 2:
                    if entity.closed and pts[0] != pts[-1]:
                        pts.append(pts[0])
                    all_paths.append(pts)

            elif dxftype == "POLYLINE":
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                if len(pts) >= 2:
                    if entity.is_closed and pts[0] != pts[-1]:
                        pts.append(pts[0])
                    all_paths.append(pts)

            elif dxftype == "CIRCLE":
                center = entity.dxf.center
                radius = float(entity.dxf.radius)
                pts = sample_arc(
                    center=(center.x, center.y),
                    radius=radius,
                    start_angle_deg=0,
                    end_angle_deg=360,
                    num=360,
                )
                if pts:
                    pts.append(pts[0])
                    all_paths.append(pts)

            elif dxftype == "ARC":
                center = entity.dxf.center
                radius = float(entity.dxf.radius)
                start_angle = float(entity.dxf.start_angle)
                end_angle = float(entity.dxf.end_angle)

                pts = sample_arc(
                    center=(center.x, center.y),
                    radius=radius,
                    start_angle_deg=start_angle,
                    end_angle_deg=end_angle,
                    num=240,
                )
                if pts:
                    all_paths.append(pts)

            elif dxftype == "ELLIPSE":
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = float(entity.dxf.ratio)
                start_param = float(entity.dxf.start_param)
                end_param = float(entity.dxf.end_param)

                pts = sample_ellipse(
                    center=(center.x, center.y),
                    major_axis=(major_axis.x, major_axis.y),
                    ratio=ratio,
                    start_param=start_param,
                    end_param=end_param,
                    num=320,
                )
                if pts:
                    if abs((end_param - start_param) % (2 * math.pi)) < 1e-3:
                        pts.append(pts[0])
                    all_paths.append(pts)

            elif dxftype == "SPLINE":
                pts = sample_spline(entity, num=400)
                if len(pts) >= 2:
                    all_paths.append(pts)

        except Exception as ex:
            print(f"[WARN] Failed to process {dxftype}: {ex}")

    return all_paths


def load_paths_from_dxf(dxf_path: Path) -> List[Path2D]:
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    doc = ezdxf.readfile(dxf_path)
    paths = collect_entity_paths(doc)

    if not paths:
        raise ValueError(f"No supported entities found in {dxf_path.name}.")

    return paths


def compute_bounds(paths: List[Path2D]) -> Tuple[float, float, float, float]:
    xs = []
    ys = []

    for path in paths:
        for x, y in path:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        raise ValueError("Failed to extract geometry from the drawing.")

    return min(xs), min(ys), max(xs), max(ys)


def compute_dxf_dimensions(paths: List[Path2D]) -> dict:
    min_x, min_y, max_x, max_y = compute_bounds(paths)

    width = float(max_x - min_x)
    height = float(max_y - min_y)

    if width <= 0 or height <= 0:
        raise ValueError("Failed to compute part dimensions from DXF.")

    return {
        "width_mm": max(width, height),
        "height_mm": min(width, height),
    }


def world_to_image_transform(
    bounds: Tuple[float, float, float, float],
    canvas_size: int = 2048,
    padding: int = 40,
) -> Callable[[Point2D], PixelPoint]:
    min_x, min_y, max_x, max_y = bounds
    w = max_x - min_x
    h = max_y - min_y

    if w == 0 or h == 0:
        raise ValueError("Invalid geometry bounds.")

    scale = min(
        (canvas_size - 2 * padding) / w,
        (canvas_size - 2 * padding) / h,
    )

    def transform(pt: Point2D) -> PixelPoint:
        x, y = pt

        x = (x - min_x) * scale + padding
        y = (y - min_y) * scale + padding
        y = canvas_size - y

        return int(round(x)), int(round(y))

    return transform
