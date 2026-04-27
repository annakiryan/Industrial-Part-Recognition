import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


def extract_channel(image: np.ndarray, use_hsv_v: bool = True) -> np.ndarray:
    if image is None:
        raise ValueError("Image is not loaded")

    if use_hsv_v:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(gray: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    if blur_ksize <= 1:
        return gray

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    return cv2.medianBlur(gray, blur_ksize)


def threshold_image(
    gray: np.ndarray,
    threshold_mode: str = "otsu",
    fixed_threshold: int = 30,
) -> np.ndarray:
    if threshold_mode == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    if threshold_mode == "fixed":
        _, mask = cv2.threshold(gray, fixed_threshold, 255, cv2.THRESH_BINARY)
        return mask

    raise ValueError("threshold_mode must be 'otsu' or 'fixed'")


def apply_opening(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 0:
        return mask

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def apply_closing(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    if kernel_size <= 0:
        return mask

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(mask)


def find_paper_roi(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    blur = cv2.GaussianBlur(v, (9, 9), 0)
    _, otsu_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = image.shape[:2]
    paper_ratio = cv2.countNonZero(otsu_mask) / (h * w)

    if paper_ratio > 0.90:
        return None

    if paper_ratio < 0.5:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    otsu_mask = cv2.morphologyEx(
        otsu_mask,
        cv2.MORPH_CLOSE,
        np.ones((25, 25), np.uint8),
    )
    otsu_mask = cv2.morphologyEx(
        otsu_mask,
        cv2.MORPH_OPEN,
        np.ones((11, 11), np.uint8),
    )

    contours, _ = cv2.findContours(
        otsu_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    if cv2.contourArea(contours_sorted[0]) > 0.80 * h * w:
        return None

    paper_contour = contours_sorted[0]
    x, y, cw, ch = cv2.boundingRect(paper_contour)

    if cw < 50 or ch < 50:
        return None

    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + cw + margin)
    y2 = min(h, y + ch + margin)

    return (x1, y1, x2, y2)


def binarize_detail(
    image: np.ndarray,
    use_hsv_v: bool = True,
    blur_ksize: int = 5,
    threshold_mode: str = "otsu",
    fixed_threshold: int = 30,
    open_kernel_size: int = 5,
    close_kernel_size: int = 9,
    adaptive: bool = True,
) -> np.ndarray:
    gray = extract_channel(image, use_hsv_v=use_hsv_v)
    gray = apply_blur(gray, blur_ksize=blur_ksize)
    mask = threshold_image(
        gray,
        threshold_mode=threshold_mode,
        fixed_threshold=fixed_threshold,
    )
    mask = invert_mask(mask)
    mask = apply_opening(mask, kernel_size=open_kernel_size)
    mask = apply_closing(mask, kernel_size=close_kernel_size)
    mask = keep_largest_component(mask)

    return mask


def _binarize_raw_details_mask(
    image: np.ndarray,
    use_hsv_v: bool = True,
    blur_ksize: int = 5,
    threshold_mode: str = "otsu",
    fixed_threshold: int = 30,
    open_kernel_size: int = 5,
    close_kernel_size: int = 9,
) -> np.ndarray:
    gray = extract_channel(image, use_hsv_v=use_hsv_v)
    gray = apply_blur(gray, blur_ksize=blur_ksize)
    mask = threshold_image(
        gray,
        threshold_mode=threshold_mode,
        fixed_threshold=fixed_threshold,
    )
    mask = invert_mask(mask)
    mask = apply_opening(mask, kernel_size=open_kernel_size)
    mask = apply_closing(mask, kernel_size=close_kernel_size)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def extract_detail_components(
    mask: np.ndarray,
    min_component_area_px: int = 1500,
) -> List[Dict[str, object]]:
    if mask is None:
        raise ValueError("mask is None")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    components: List[Dict[str, object]] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_component_area_px:
            continue

        component_mask = np.zeros_like(mask)
        component_mask[labels == label] = 255

        components.append(
            {
                "component_id": label,
                "area_px": area,
                "bbox_xywh": [x, y, w, h],
                "mask": component_mask,
            }
        )

    components.sort(key=lambda item: int(item["area_px"]), reverse=True)
    return components


def build_clean_mask_from_components(
    components: List[Dict[str, object]],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    clean_mask = np.zeros(image_shape, dtype=np.uint8)

    for component in components:
        component_mask = component["mask"]
        clean_mask[component_mask > 0] = 255

    return clean_mask


def binarize_details_mask(
    image: np.ndarray,
    use_hsv_v: bool = True,
    blur_ksize: int = 5,
    threshold_mode: str = "otsu",
    fixed_threshold: int = 30,
    open_kernel_size: int = 5,
    close_kernel_size: int = 9,
    min_component_area_px: int = 1500,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:

    raw_mask = _binarize_raw_details_mask(
        image=image,
        use_hsv_v=use_hsv_v,
        blur_ksize=blur_ksize,
        threshold_mode=threshold_mode,
        fixed_threshold=fixed_threshold,
        open_kernel_size=open_kernel_size,
        close_kernel_size=close_kernel_size,
    )

    components = extract_detail_components(
        mask=raw_mask,
        min_component_area_px=min_component_area_px,
    )

    clean_mask = build_clean_mask_from_components(
        components=components,
        image_shape=raw_mask.shape,
    )

    return clean_mask, components


def crop_mask_by_bbox(
    mask: np.ndarray,
    bbox_xywh: List[int],
    margin_px: int = 10,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:

    x, y, w, h = bbox_xywh
    height, width = mask.shape[:2]

    x1 = max(0, x - margin_px)
    y1 = max(0, y - margin_px)
    x2 = min(width, x + w + margin_px)
    y2 = min(height, y + h + margin_px)

    cropped = mask[y1:y2, x1:x2].copy()
    return cropped, (x1, y1, x2, y2)
