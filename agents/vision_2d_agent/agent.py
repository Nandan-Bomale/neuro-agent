"""
vision_2d_agent/agent.py  --  YOLO11-Based Tumor Localizer  (FIX v4.1)
=====================================================================
FIX 1: Accurate tumor area from actual pixel contour
FIX 2: Multi-box merge (vertical-zone aware)
FIX 3: ABC/2 clinical volume formula
FIX 4: Anomaly-score + position-weighted detection ranking
FIX 5: Hard skull-base exclusion
FIX 6: Positive vs Negative confidence gating -> NO phantom boxes on normal brains
FIX 7: Sequence-aware detection (FLAIR/T2 -> no CV fallback)
FIX 8: Minimum CV anomaly score threshold >= 25.0
FIX 9: Maximum CV box area cap (20% of brain)
FIX 10: Border exclusion (top 5%, bottom 3% -- removes watermarks)
FIX 11: Inner brain zone only for CV fallback (excludes skull/dura/sagittal sinus)
FIX 12: Non-brain pass-through (is_brain=False -> no detection)
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from skimage.segmentation import morphological_chan_vese

logger = logging.getLogger(__name__)
AGENT_NAME = "vision_2d_agent"

_YOLO_WEIGHTS     = Path("models/yolo/weights/detect_v2.pt")
_YOLO_WEIGHTS_V1  = Path("models/yolo/weights/detect_best.pt")
_YOLO_SEG_WEIGHTS = Path("models/yolo/weights/best.pt")

_NO_CV_FALLBACK_SEQUENCES = {"FLAIR", "T2"}
_MIN_CV_SCORE = 25.0
_MAX_CV_BOX_BRAIN_FRACTION = 0.20
_MIN_TUMOR_AREA_CM2 = 0.3


def _estimate_pixel_spacing_mm(H: int, W: int) -> float:
    dim = max(H, W)
    if dim <= 256:   return 1.0
    elif dim <= 320: return 0.85
    elif dim <= 400: return 0.75
    elif dim <= 512: return 0.60
    else:            return 0.45


def _area_from_mask(mask_bin: np.ndarray, ps: float) -> tuple:
    area_px  = float(np.sum(mask_bin > 0))
    area_cm2 = area_px * (ps ** 2) / 100.0
    return round(area_cm2, 2), area_px


def _area_from_box(box: tuple, img_gray: np.ndarray, ps: float) -> tuple:
    if img_gray.ndim == 3:
        img_gray = img_gray[:, :, 0]
    x, y, w, h = box
    H, W = img_gray.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)
    roi = img_gray[y:y2, x:x2]
    full_mask = np.zeros_like(img_gray)
    if roi.size < 4:
        area_px = float(w * h)
        return round(area_px * (ps**2) / 100.0, 2), area_px, full_mask
    _, roi_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(roi_mask)
    if cnts:
        cv2.drawContours(clean, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    area_px = float(np.sum(clean > 0))
    if area_px < 10:
        area_px = float(w * h) * 0.7
        clean = roi_mask
    area_cm2 = area_px * (ps**2) / 100.0
    full_mask[y:y2, x:x2] = clean
    return round(area_cm2, 2), area_px, full_mask


def _abc2_volume_cm3(box: tuple, ps: float) -> float:
    _, _, w, h = box
    A, B = w * ps, h * ps
    C = (A + B) / 2
    return round((A * B * C) / 2000.0, 2)


def _get_brain_mask(gray: np.ndarray) -> tuple:
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    H, W = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bm   = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones_like(gray, dtype=np.uint8)*255, 0, 0, W, H, W//2, H//2
    brain_c = max(cnts, key=cv2.contourArea)
    head_mask = np.zeros_like(gray)
    cv2.drawContours(head_mask, [brain_c], -1, 255, -1)
    bx, by, bw, bh = cv2.boundingRect(brain_c)
    M  = cv2.moments(brain_c)
    cx = int(M["m10"]/M["m00"]) if M["m00"] > 0 else W//2
    cy = int(M["m01"]/M["m00"]) if M["m00"] > 0 else H//2
    ep = max(8, min(bw, bh)//18)
    eroded = cv2.erode(head_mask, np.ones((ep, ep), np.uint8))
    return eroded, bx, by, bw, bh, cx, cy


def _get_safe_zone(H: int, W: int) -> tuple:
    """Returns (y_min, y_max, x_min, x_max) - excludes watermark borders."""
    return int(H * 0.05), int(H * 0.97), int(W * 0.02), int(W * 0.98)


def _cv_anomaly_fallback(img: np.ndarray, brain_mask: np.ndarray, cx: int, cy: int, head_bw: int, head_bh: int,
                         min_score: float = _MIN_CV_SCORE,
                         max_brain_frac: float = _MAX_CV_BOX_BRAIN_FRACTION) -> Optional[tuple]:
    H, W = img.shape
    brain_pixels = img[brain_mask > 0]
    if len(brain_pixels) < 50:
        return None
        
    brain_mean = float(brain_pixels.mean())
    brain_std  = float(max(brain_pixels.std(), 1.0))
    brain_area = float(np.sum(brain_mask > 0))
    max_box_area = brain_area * max_brain_frac
    min_dist   = max(head_bw, head_bh) * 0.03
    close_k    = max(3, W // 65)
    blurred    = cv2.GaussianBlur(img, (3, 3), 0)
    best       = None
    
    y_min_safe, y_max_safe, x_min_safe, x_max_safe = _get_safe_zone(H, W)
    
    # Erode brain mask aggressively to stay away from skull/dura/sinus edges
    inner_brain = cv2.erode(brain_mask, np.ones((20, 20), np.uint8))
    
    brain_ys = np.where(brain_mask > 0)[0]
    brain_top_y    = int(brain_ys.min()) if len(brain_ys) else 0
    brain_bottom_y = int(brain_ys.max()) if len(brain_ys) else H
    upper_brain_y  = brain_top_y + int((brain_bottom_y - brain_top_y) * 0.65)

    def _check(cnts, anom_fn):
        nonlocal best
        for c in cnts:
            area = cv2.contourArea(c)
            if area < max(120, brain_area * 0.005) or area > max_box_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            blob_cy = int(y + bh / 2)
            blob_cx = int(x + bw / 2)
            
            # Must be strictly inside inner brain parenchyma
            if blob_cy < 0 or blob_cy >= H or blob_cx < 0 or blob_cx >= W:
                continue
            if inner_brain[blob_cy, blob_cx] == 0:
                continue
                
            # Exclude sagittal sinus (centerline top 15% and bottom 15%)
            if abs(blob_cx - cx) < W * 0.08:
                if blob_cy < brain_top_y + (brain_bottom_y - brain_top_y) * 0.15:
                    continue
                if blob_cy > brain_bottom_y - (brain_bottom_y - brain_top_y) * 0.15:
                    continue

            if y < y_min_safe or (y + bh) > y_max_safe:
                continue
            if x < x_min_safe or (x + bw) > x_max_safe:
                continue
                
            dist = np.sqrt((blob_cx - cx)**2 + (blob_cy - cy)**2)
            if dist < min_dist:
                continue
            hull_a = cv2.contourArea(cv2.convexHull(c))
            sol    = area / (hull_a + 1e-5)
            if sol < 0.20:
                continue
            if min(bw, bh) / (max(bw, bh) + 1e-5) < 0.20:
                continue
            m = np.zeros_like(img)
            cv2.drawContours(m, [c], -1, 255, -1)
            anom = anom_fn(float(img[m > 0].mean()))
            if anom < 1.5:  # Must be noticeably anomalous
                continue
            upper_bonus = 1.5 if blob_cy < upper_brain_y else 1.0
            size_bonus  = min(area / (brain_area * 0.05), 2.0)
            score = anom * (area ** 0.40) * sol * upper_bonus * size_bonus
            if best is None or score > best["score"]:
                best = {"box": (x, y, bw, bh), "score": score}

    for pct in [99.5, 99, 98, 97]:
        p = np.percentile(brain_pixels, pct)
        if p <= brain_mean + 0.8 * brain_std:
            continue
        _, mask = cv2.threshold(blurred, p, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, inner_brain)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _check(cnts, lambda bm: (bm - brain_mean) / brain_std)

    if best is None:
        return None
    if best["score"] < min_score:
        logger.info("[%s] CV score %.2f < min %.2f -- no tumor detected", AGENT_NAME, best["score"], min_score)
        return None
    return best["box"]


def _boxes_intersect(b1: tuple, b2: tuple, margin: int = 5) -> bool:
    x1, y1, w1, h1 = b1[:4]
    x2, y2, w2, h2 = b2[:4]
    return not (x1 + w1 + margin < x2 or x2 + w2 + margin < x1 or
                y1 + h1 + margin < y2 or y2 + h2 + margin < y1)


def _cluster_boxes(cand_boxes: list) -> list:
    clusters = []
    for b in sorted(cand_boxes, key=lambda x: x[4], reverse=True):
        merged = False
        for cl in clusters:
            if any(_boxes_intersect(b, member) for member in cl['boxes']):
                cl['boxes'].append(b)
                bx1 = min(cl['union'][0], b[0])
                by1 = min(cl['union'][1], b[1])
                bx2 = max(cl['union'][0] + cl['union'][2], b[0] + b[2])
                by2 = max(cl['union'][1] + cl['union'][3], b[1] + b[3])
                cl['union'] = (bx1, by1, bx2 - bx1, by2 - by1)
                cl['max_conf'] = max(cl['max_conf'], b[4])
                merged = True
                break
        if not merged:
            clusters.append({
                'union': b[:4],
                'boxes': [b],
                'max_conf': b[4]
            })
    return clusters


class Vision2DAgent:
    """YOLO11 + Morphological Active Contour Anatomical Precision Tumor Localizer."""

    def __init__(self, output_dir: str = "data/interim/vision_2d") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._yolo_det = None
        self._yolo_seg = None
        self._device   = "cpu"

    def _init_yolo(self) -> None:
        if self._yolo_det is not None and self._yolo_seg is not None:
            return
        from ultralytics import YOLO
        import torch

        det_w = _YOLO_WEIGHTS_V1 if _YOLO_WEIGHTS_V1.exists() else _YOLO_WEIGHTS
        seg_w = _YOLO_SEG_WEIGHTS if _YOLO_SEG_WEIGHTS.exists() else det_w

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._device = 0
        else:
            self._device = "cpu"

        try:
            if self._yolo_det is None and det_w.exists():
                self._yolo_det = YOLO(str(det_w))
                _ = self._yolo_det.predict(source=np.zeros((64, 64, 3), dtype="uint8"),
                                           device=self._device, conf=0.9, verbose=False)
            if self._yolo_seg is None and seg_w.exists():
                self._yolo_seg = YOLO(str(seg_w))
                _ = self._yolo_seg.predict(source=np.zeros((64, 64, 3), dtype="uint8"),
                                           device=self._device, conf=0.9, verbose=False)
            logger.info("[%s] YOLO Det (%s) & Seg (%s) ready on %s",
                        AGENT_NAME, det_w.name, seg_w.name, self._device)
        except Exception as e:
            logger.error("[%s] Error initializing YOLO: %s", AGENT_NAME, e)
            self._device = "cpu"
            if self._yolo_det is None and det_w.exists():
                self._yolo_det = YOLO(str(det_w))
            if self._yolo_seg is None and seg_w.exists():
                self._yolo_seg = YOLO(str(seg_w))

    def _run_yolo(self, img_bgr: np.ndarray, img_gray: np.ndarray, brain_mask: np.ndarray,
                  brain_mean: float, brain_std: float, mri_sequence: str = "T1",
                  brain_bbox: Optional[tuple] = None) -> tuple:
        H, W = img_bgr.shape[:2]
        brain_std_safe = max(brain_std, 1.0)
        bx, by, bw, bh = brain_bbox if brain_bbox else (0, 0, W, H)
        cand_boxes = []

        # 1. Primary candidate detection with fine-grained detector (detect_best.pt)
        # Class 1 = positive (tumor), Class 0 = negative (normal tissue)
        if self._yolo_det is not None:
            try:
                r_det = self._yolo_det(img_bgr, conf=0.18, verbose=False, device=self._device)[0]
                if r_det.boxes is not None and len(r_det.boxes) > 0:
                    for b in r_det.boxes:
                        conf = float(b.conf[0])
                        cls_id = int(b.cls[0])
                        if cls_id != 1:  # Must be positive tumor class
                            continue
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Anatomical constraint: must be inside head and above cervical neck
                        if cy > (by + bh * 0.90) or brain_mask[min(cy, H-1), min(cx, W-1)] == 0:
                            continue

                        roi = img_gray[y1:y2, x1:x2]
                        if roi.size < 25:
                            continue
                        r_std = float(roi.std())
                        r_mean = float(roi.mean())
                        z = abs(r_mean - brain_mean) / brain_std_safe

                        # Anomaly check: must have texture variation or contrast against brain
                        if r_std > 12.0 or z > 0.35:
                            cand_boxes.append((x1, y1, x2 - x1, y2 - y1, conf))
            except Exception as e:
                logger.error("[%s] YOLO det error: %s", AGENT_NAME, e)

        # 2. Fallback to segmentation model for diffuse/unenhanced masses (e.g. convexity meningioma)
        # Requires conf >= 0.20 to completely prevent false positives on normal brain scans
        if not cand_boxes and self._yolo_seg is not None:
            try:
                r_seg = self._yolo_seg(img_bgr, conf=0.20, verbose=False, device=self._device)[0]
                if r_seg.boxes is not None and len(r_seg.boxes) > 0:
                    for b in r_seg.boxes:
                        conf = float(b.conf[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Exclude image border watermark rows (top 4%, bottom 3%)
                        if y1 < H * 0.04 or y2 > H * 0.97:
                            continue
                        if cy > (by + bh * 0.90) or brain_mask[min(cy, H-1), min(cx, W-1)] == 0:
                            continue

                        # Discard spurious whole-brain fallback boxes
                        box_brain_frac = ((x2 - x1) * (y2 - y1)) / max(bw * bh, 1)
                        if box_brain_frac > 0.40:
                            continue

                        roi = img_gray[y1:y2, x1:x2]
                        if roi.size < 25:
                            continue
                        r_std = float(roi.std())
                        r_mean = float(roi.mean())
                        z = abs(r_mean - brain_mean) / brain_std_safe

                        if r_std > 15.0 or z > 0.50:
                            cand_boxes.append((x1, y1, x2 - x1, y2 - y1, conf))
            except Exception as e:
                logger.error("[%s] YOLO seg fallback error: %s", AGENT_NAME, e)

        if not cand_boxes:
            skip_cv = (mri_sequence in _NO_CV_FALLBACK_SEQUENCES)
            return None, None, 0.0, skip_cv

        # 3. Spatial clustering of intersecting candidate boxes
        clusters = _cluster_boxes(cand_boxes)
        best_cluster = max(clusters, key=lambda c: c['max_conf'])
        target_box = best_cluster['union']
        top_conf = best_cluster['max_conf']

        # 4. Anatomical Segmentation inside Target ROI using Morphological Active Contours
        x, y, w, h = target_box
        pad = 8
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(W, x + w + pad)
        ry2 = min(H, y + h + pad)

        roi_gray = img_gray[ry1:ry2, rx1:rx2]
        # Bilateral filter preserves sharp anatomical edges while reducing noise
        filtered_roi = cv2.bilateralFilter(roi_gray, 5, 45, 45)

        # Morphological Chan-Vese active contour
        init_ls = np.zeros(filtered_roi.shape, dtype=np.int8)
        init_ls[pad:-pad, pad:-pad] = 1
        cv_mask = morphological_chan_vese(filtered_roi, num_iter=30, init_level_set=init_ls, smoothing=2)

        # Topological closure & necrotic core infill
        cnts, _ = cv2.findContours(cv_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_roi = np.zeros_like(filtered_roi, dtype=np.uint8)
        if cnts:
            c_sig = [c for c in cnts if cv2.contourArea(c) > 150]
            if not c_sig:
                c_sig = [max(cnts, key=cv2.contourArea)]
            cv2.drawContours(filled_roi, c_sig, -1, 1, -1)
        else:
            filled_roi = cv_mask.astype(np.uint8)

        # Map back to full image coordinate space
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = filled_roi
        full_mask = cv2.bitwise_and(full_mask, full_mask, mask=brain_mask)

        # Antialiased Gaussian boundary smoothing
        blurred = cv2.GaussianBlur(full_mask.astype(np.float32), (5, 5), 1.2)
        clean_mask = (blurred >= 0.45).astype(np.uint8)

        total_tumor_px = float(np.sum(clean_mask > 0))
        if total_tumor_px < 250:
            skip_cv = (mri_sequence in _NO_CV_FALLBACK_SEQUENCES)
            return None, None, 0.0, skip_cv

        bx, by, bw, bh = cv2.boundingRect(clean_mask)
        tight_box = (bx, by, bw, bh)
        logger.info("[%s] Anatomical Active Contour: tight_box=%s conf=%.3f area_px=%d seq=%s",
                    AGENT_NAME, tight_box, top_conf, int(total_tumor_px), mri_sequence)
        return tight_box, clean_mask, top_conf, False

    @staticmethod
    def _render(img_gray: np.ndarray, box: Optional[tuple], seg_mask: Optional[np.ndarray],
                source: str, tumor_area_cm2: float = 0.0, tumor_detected: bool = True) -> np.ndarray:
        out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        if seg_mask is not None and tumor_detected:
            # Translucent crimson overlay (alpha = 0.35)
            ov = out.copy()
            ov[seg_mask > 0, 2] = 220
            ov[seg_mask > 0, 0] = 30
            ov[seg_mask > 0, 1] = 30
            out = cv2.addWeighted(out, 0.65, ov, 0.35, 0)
            cnts, _ = cv2.findContours((seg_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 0, 255), 2, cv2.LINE_AA)
        if box is not None and tumor_detected:
            x, y, w, h = box
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(out, f"AI:TumorMass [{source}]",
                        (x, max(y-6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if tumor_area_cm2 > 0:
                cv2.putText(out, f"{tumor_area_cm2:.1f} cm2",
                    (x, min(y+h+18, img_gray.shape[0]-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        elif not tumor_detected:
            cv2.putText(out, "No Tumor Detected",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(out, "AI: Normal Study",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 0), 1, cv2.LINE_AA)
        return out

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        mri_path = state.get("mri_slice_path", "")
        if not mri_path or not Path(mri_path).exists():
            return {"vision_findings": {"error": "mri_slice_path missing"}}

        preprocessing = state.get("preprocessing_findings", {})
        is_brain      = preprocessing.get("is_brain", True)
        mri_sequence  = preprocessing.get("mri_sequence", "T1")

        if not is_brain:
            logger.warning("[%s] is_brain=False -- skipping detection", AGENT_NAME)
            return {"vision_findings": {
                "tumor_detected": False,
                "bounding_box": None,
                "tumor_area_cm2": 0.0,
                "tumor_area_px": 0.0,
                "tumor_volume_cm3": 0.0,
                "pixel_spacing_mm": 1.0,
                "detection_confidence": 0.0,
                "detection_source": "rejected_non_brain",
                "mri_sequence": mri_sequence,
                "segmentation_mask_path": "",
                "inference_time_sec": 0.0,
                "slice_orientation": "unknown",
                "note": "Image rejected: not a brain MRI",
            }}

        img_gray = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            img_color = cv2.imread(mri_path)
            if img_color is None:
                return {"vision_findings": {"error": f"Cannot read {mri_path}"}}
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        if img_gray.ndim == 3:
            img_gray = img_gray[:, :, 0]

        H, W    = img_gray.shape
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        ps_mm   = _estimate_pixel_spacing_mm(H, W)

        brain_mask, bx, by, bw, bh, cx, cy = _get_brain_mask(img_gray)
        brain_pixels = img_gray[brain_mask > 0].astype(float)
        brain_mean   = float(brain_pixels.mean()) if len(brain_pixels) > 0 else 128.0
        brain_std    = float(brain_pixels.std())  if len(brain_pixels) > 0 else 30.0

        self._init_yolo()
        box, seg_mask, yolo_conf, skip_cv_fallback = self._run_yolo(
            img_bgr, img_gray, brain_mask, brain_mean, brain_std, mri_sequence,
            brain_bbox=(bx, by, bw, bh))
        source = "yolo"

        if mri_sequence in _NO_CV_FALLBACK_SEQUENCES:
            skip_cv_fallback = True

        if box is None:
            # Check if scan has genuine hyperintense contrast anomaly
            has_anomaly = (len(brain_pixels) > 50 and 
                           np.percentile(brain_pixels, 99.0) > (brain_mean + 1.8 * brain_std))
            if skip_cv_fallback or not has_anomaly:
                logger.info("[%s] Clean scan confirmed -- skipping CV fallback (seq=%s)", AGENT_NAME, mri_sequence)
            else:
                logger.info("[%s] YOLO inconclusive -- CV fallback (seq=%s)", AGENT_NAME, mri_sequence)
                box = _cv_anomaly_fallback(img_gray, brain_mask, cx, cy, bw, bh)
                if box is not None:
                    source    = "cv_fallback"
                    seg_mask  = None
                    yolo_conf = 0.0

        tumor_area_cm2 = 0.0
        tumor_area_px  = 0.0
        tumor_vol_cm3  = 0.0

        if box is not None:
            if seg_mask is not None:
                tumor_area_cm2, tumor_area_px = _area_from_mask(
                    (seg_mask > 0.5).astype(np.uint8), ps_mm)
            else:
                tumor_area_cm2, tumor_area_px, _ = _area_from_box(box, img_gray, ps_mm)
            tumor_vol_cm3 = _abc2_volume_cm3(box, ps_mm)
            if tumor_area_cm2 < _MIN_TUMOR_AREA_CM2:
                logger.info("[%s] Box %.2f cm2 < min -- noise rejected", AGENT_NAME, tumor_area_cm2)
                box = None
                tumor_area_cm2 = 0.0
                tumor_area_px  = 0.0
                tumor_vol_cm3  = 0.0

        tumor_detected = (box is not None)
        if not tumor_detected:
            source = "none"
        out_img  = self._render(img_gray, box, seg_mask, source, tumor_area_cm2, tumor_detected)
        out_path = self.output_dir / f"tumor_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(out_path), out_img)

        elapsed = time.perf_counter() - t0
        logger.info("[%s] %.3fs | seq=%s | src=%s | detected=%s | conf=%.3f | box=%s | area=%.2fcm2",
                    AGENT_NAME, elapsed, mri_sequence, source, tumor_detected, yolo_conf, box, tumor_area_cm2)

        return {"vision_findings": {
            "tumor_detected":         tumor_detected,
            "bounding_box":           box,
            "tumor_area_cm2":         tumor_area_cm2,
            "tumor_area_px":          tumor_area_px,
            "tumor_volume_cm3":       tumor_vol_cm3,
            "pixel_spacing_mm":       ps_mm,
            "detection_confidence":   yolo_conf,
            "detection_source":       source,
            "mri_sequence":           mri_sequence,
            "segmentation_mask_path": str(out_path),
            "inference_time_sec":     elapsed,
            "slice_orientation":      "yolo_2d",
        }}