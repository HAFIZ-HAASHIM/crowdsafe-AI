from typing import List, Tuple
import time

import cv2
import numpy as np


def to_centroids(boxes_xyxy: np.ndarray) -> List[Tuple[int, int]]:
    centroids: List[Tuple[int, int]] = []
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return centroids
    for (x1, y1, x2, y2) in boxes_xyxy:
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        centroids.append((cx, cy))
    return centroids


def match_and_movements(prev: List[Tuple[int, int]], curr: List[Tuple[int, int]], max_dist: float = 80.0) -> List[float]:
    # Greedy nearest-neighbor matching for MVP
    movements: List[float] = []
    used_prev = set()

    for (cx, cy) in curr:
        best_d, best_idx = 1e9, -1
        for i, (px, py) in enumerate(prev):
            if i in used_prev:
                continue
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d < best_d:
                best_d, best_idx = d, i
        if best_idx != -1 and best_d <= max_dist:
            used_prev.add(best_idx)
            movements.append(float(best_d))
        else:
            movements.append(0.0)
    return movements


def overlay_heatmap(frame: np.ndarray, centroids: List[Tuple[int, int]], grid_rows: int, grid_cols: int, alpha: float = 0.25) -> np.ndarray:
    """Overlay a smooth density heatmap on the frame using JET colormap (blue→green→yellow→red).

    Red indicates the highest density cells.
    """
    H, W = frame.shape[:2]
    cell_h = max(1, H // grid_rows)
    cell_w = max(1, W // grid_cols)

    # Build coarse grid density
    heat = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for (cx, cy) in centroids:
        r = min(int(cy // cell_h), grid_rows - 1)
        c = min(int(cx // cell_w), grid_cols - 1)
        heat[r, c] += 1.0

    if heat.size == 0:
        return frame

    max_cell = float(heat.max())
    if max_cell <= 0:
        return frame

    # Normalize to [0,255] and upsample to frame size
    heat_norm = (heat / max_cell) * 255.0
    heat_norm = heat_norm.astype(np.uint8)
    heat_big = cv2.resize(heat_norm, (W, H), interpolation=cv2.INTER_LINEAR)

    # Smooth to create continuous blobs rather than blocks
    # Kernel sigma scaled with cell size for resolution independence
    sigma = max(cell_h, cell_w) * 0.6
    if sigma > 0:
        heat_big = cv2.GaussianBlur(heat_big, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Apply JET colormap: low=blue, mid=green/yellow, high=red
    colored = cv2.applyColorMap(heat_big, cv2.COLORMAP_JET)

    # Blend only where heat > 0 to keep empty areas untouched
    mask = heat_big > 0
    if not np.any(mask):
        return frame
    blended = cv2.addWeighted(colored, alpha, frame, 1.0 - alpha, 0)

    out = frame.copy()
    out[mask] = blended[mask]
    return out


def draw_boxes(frame: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> None:
    if boxes_xyxy is None:
        return
    for (x1, y1, x2, y2) in boxes_xyxy:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_hud(frame: np.ndarray, crowd_count: int, movements: List[float], panic: bool = False, fps: float = 0.0) -> None:
    y = 28
    cv2.putText(frame, f"People: {crowd_count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += 28
    if movements:
        cv2.putText(frame, f"Max move: {max(movements):.1f}px", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y += 28
    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)
        y += 28
    if panic:
        cv2.putText(frame, "PANIC DETECTED!", (18, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


class Throttler:
    def __init__(self, min_interval: float = 3.0):
        self.min_interval = float(min_interval)
        self._last = 0.0

    def can_run(self) -> bool:
        return (time.time() - self._last) >= self.min_interval

    def ran(self) -> None:
        self._last = time.time()
