"""
centering.py — Pokemon card centering algorithm using OpenCV.

Pipeline:
  OUTER BOUNDARY (card vs table):
    For every row → scan left-to-right, find first Sobel-X peak → left edge pt.
    Scan right-to-left → right edge pt.
    For every column → scan top-to-bottom → top edge pt; bottom-to-top → bottom.
    RANSAC fits a line to each of the 4 point clouds, rejecting rounded corners
    and noise as outliers. Intersect 4 lines → card corners.

  INNER BOUNDARY (card border vs artwork):
    After perspective warp the card is flat and axis-aligned.
    Same scanline scan on the warped card, from each edge inward.
    Median of all hits per side → robust border width.

  GRADING:
    Measure 4 borders in pixels, compute left/right and top/bottom ratios.
    Evaluate against PSA 10 (45/55) and PSA 9 (60/40) standards.
"""

import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CenteringResult:
    lr_ratio: str
    tb_ratio: str
    lr_pct_left: float
    tb_pct_top: float
    left_px: int
    right_px: int
    top_px: int
    bottom_px: int
    psa10_lr_pass: bool
    psa10_tb_pass: bool
    psa9_lr_pass: bool
    psa9_tb_pass: bool
    confidence: str        # "high" | "low" | "card_not_found" | "no_inner_border"
    card_type: str = "standard"   # "standard" | "full_art" | "unknown"
    inner_consistency: float = 1.0  # 0–1, fraction of inner border hits in histogram peak bin

    def to_dict(self) -> dict:
        return asdict(self)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _intersect_lines(l1, l2):
    """
    Intersect two lines given as (vx, vy, x0, y0) from cv2.fitLine.
    Returns [x, y] or None if parallel.
    """
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < 1e-10:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
    return [x1 + t * vx1, y1 + t * vy1]


# ── Preprocessing ────────────────────────────────────────────────────────────

def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Absolute Sobel gradient magnitude of a single channel, normalised 0-255."""
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    if mag.max() > 0:
        mag = mag / mag.max() * 255
    return np.clip(mag, 0, 255).astype(np.uint8)


def _combined_magnitude(img: np.ndarray, use_clahe: bool = False) -> np.ndarray:
    """
    Max of grayscale, saturation, and LAB b* Sobel magnitudes, all normalised 0-255.

    Grayscale captures brightness transitions (most card-to-table edges).
    Saturation captures colour transitions (e.g. dark navy border vs dark table).
    LAB b* captures blue-yellow transitions — catches yellow/gold inner borders
    common on Pokemon cards even when grayscale and saturation look similar.
    Taking the max means any channel can carry the signal — whichever is
    stronger for a given edge wins.

    use_clahe: apply CLAHE to the grayscale channel before Sobel. Boosts local
    contrast for inner border detection on dark/low-contrast cards. Not needed
    for outer card detection where contrast is typically high.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    mag_gray = _sobel_magnitude(gray)

    sat = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)[:, :, 1]
    mag_sat = _sobel_magnitude(sat)

    lab_b = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)[:, :, 2]
    mag_lab_b = _sobel_magnitude(lab_b)

    return np.maximum(np.maximum(mag_gray, mag_sat), mag_lab_b)


# ── Scanline edge collection ──────────────────────────────────────────────────

def _scan_row_edges(mag: np.ndarray, threshold: int, skip_frac: float = 0.10,
                    search_frac: float = 0.50):
    """
    For every row, find the first column from the left within the search window
    whose magnitude exceeds `threshold` (left edge), and first from the right.

    skip_frac:   ignore the outermost fraction at each end (avoids image border
                 artifacts).
    search_frac: only accept hits within this fraction of each respective edge.
                 e.g. 0.35 means left hits must be in the left 35% of the image,
                 right hits in the right 35%. Prevents interior card features
                 from being mistaken for boundary edges.

    Returns (left_pts, right_pts) each as (N,2) float32 arrays of (x, y).
    """
    h, w = mag.shape
    skip = int(w * skip_frac)
    max_left  = int(w * search_frac)   # left hits must be left of this x
    min_right = w - int(w * search_frac)  # right hits must be right of this x
    left_pts, right_pts = [], []

    for row in range(h):
        line = mag[row, skip: max_left]
        hits = np.where(line > threshold)[0]
        if len(hits) > 0:
            left_pts.append([float(hits[0] + skip), float(row)])

        line_r = mag[row, min_right: w - skip]
        hits_r = np.where(line_r > threshold)[0]
        if len(hits_r) > 0:
            right_pts.append([float(hits_r[-1] + min_right), float(row)])

    return (np.array(left_pts,  dtype=np.float32) if left_pts  else None,
            np.array(right_pts, dtype=np.float32) if right_pts else None)


def _scan_col_edges(mag: np.ndarray, threshold: int, skip_frac: float = 0.10,
                    search_frac: float = 0.50):
    """
    For every column, find the first row from the top within the search window
    exceeding `threshold` (top edge), and first from the bottom.

    search_frac: top hits must be in the top 35% of the image; bottom hits in
                 the bottom 35%. Mirrors the logic of _scan_row_edges.

    Returns (top_pts, bottom_pts) each as (N,2) float32 arrays of (x, y).
    """
    h, w = mag.shape
    skip  = int(h * skip_frac)
    max_top   = int(h * search_frac)
    min_bot   = h - int(h * search_frac)
    top_pts, bot_pts = [], []

    for col in range(w):
        col_top = mag[skip: max_top, col]
        hits = np.where(col_top > threshold)[0]
        if len(hits) > 0:
            top_pts.append([float(col), float(hits[0] + skip)])

        col_bot = mag[min_bot: h - skip, col]
        hits_b  = np.where(col_bot > threshold)[0]
        if len(hits_b) > 0:
            bot_pts.append([float(col), float(hits_b[-1] + min_bot)])

    return (np.array(top_pts, dtype=np.float32) if top_pts else None,
            np.array(bot_pts, dtype=np.float32) if bot_pts else None)


# ── RANSAC line fitting ───────────────────────────────────────────────────────

def _ransac_line(pts: np.ndarray, iterations: int = 200, inlier_dist: float = 6.0):
    """
    Fit a line to pts using RANSAC.
    Returns (vx, vy, x0, y0) consistent with cv2.fitLine output, or None.
    inlier_dist: max perpendicular distance (px) to count as inlier.
    """
    if pts is None or len(pts) < 10:
        return None

    best_line   = None
    best_count  = 0
    n = len(pts)

    rng = np.random.default_rng(42)

    for _ in range(iterations):
        idx = rng.choice(n, 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1:
            continue
        d_hat = d / length
        n_hat = np.array([-d_hat[1], d_hat[0]])

        dists = np.abs((pts - p1) @ n_hat)
        count = int((dists < inlier_dist).sum())

        if count > best_count:
            best_count = count
            best_line  = (p1, d_hat)

    if best_line is None:
        return None

    # Refit using all inliers for a precise final line
    p1, d_hat = best_line
    n_hat = np.array([-d_hat[1], d_hat[0]])
    dists = np.abs((pts - p1) @ n_hat)
    inliers = pts[dists < inlier_dist]

    if len(inliers) < 4:
        return None

    line = cv2.fitLine(inliers, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return line  # (vx, vy, x0, y0)


# ── Outer boundary: card vs table ────────────────────────────────────────────

def _find_card_corners(img: np.ndarray):
    """
    Detect the 4 card corners using scanline Sobel scanning + RANSAC line fits.

    For every row, scan from the left until the first strong gradient → left
    edge candidate. Scan from the right → right edge candidate.
    Same top/bottom from columns. RANSAC fits a precise line through each cloud,
    discarding rounded corners and noise. Intersect 4 lines → corners.

    Returns (corners [TL,TR,BR,BL], confidence "high"|"low") or (None, "low").
    """
    h, w = img.shape[:2]

    # Combined grayscale+saturation Sobel — handles both brightness and colour
    # transitions. Saturation recovers low-contrast edges like dark navy border
    # on a dark table that are nearly invisible in grayscale.
    mag = _combined_magnitude(img)

    # Adaptive per-side threshold: start at 15% of max, fall back to lower
    # values if a side returns too few points (e.g. low-contrast dark border
    # against a dark table). Each side retries independently so a bright edge
    # on one side doesn't force a too-low threshold on all sides.
    MIN_PTS = 50
    THRESHOLDS = [0.15, 0.10, 0.07, 0.04]

    def _scan_with_fallback_row(side):
        for frac in THRESHOLDS:
            t = max(int(mag.max() * frac), 5)
            l, r = _scan_row_edges(mag, t)
            pts = l if side == "left" else r
            if pts is not None and len(pts) >= MIN_PTS:
                return pts
        return None

    def _scan_with_fallback_col(side):
        for frac in THRESHOLDS:
            t = max(int(mag.max() * frac), 5)
            tp, b = _scan_col_edges(mag, t)
            pts = tp if side == "top" else b
            if pts is not None and len(pts) >= MIN_PTS:
                return pts
        return None

    left_pts  = _scan_with_fallback_row("left")
    right_pts = _scan_with_fallback_row("right")
    top_pts   = _scan_with_fallback_col("top")
    bot_pts   = _scan_with_fallback_col("bottom")

    lines = {
        "left":   _ransac_line(left_pts),
        "right":  _ransac_line(right_pts),
        "top":    _ransac_line(top_pts),
        "bottom": _ransac_line(bot_pts),
    }

    if any(v is None for v in lines.values()):
        missing = [k for k, v in lines.items() if v is None]
        logger.error("RANSAC failed for edges: %s", missing)
        return None, "low"

    corners_raw = [
        _intersect_lines(lines["top"],    lines["left"]),
        _intersect_lines(lines["top"],    lines["right"]),
        _intersect_lines(lines["bottom"], lines["right"]),
        _intersect_lines(lines["bottom"], lines["left"]),
    ]

    if any(c is None for c in corners_raw):
        logger.error("Parallel lines in RANSAC result.")
        return None, "low"

    corners = _order_corners(np.array(corners_raw, dtype=np.float32))

    # Sanity check: corners inside image (with margin) and plausible aspect ratio
    margin = 50
    if not np.all(
        (corners[:, 0] >= -margin) & (corners[:, 0] <= w + margin) &
        (corners[:, 1] >= -margin) & (corners[:, 1] <= h + margin)
    ):
        logger.error("Card corners outside image bounds.")
        return None, "low"

    tl, tr, br, bl = corners
    cw = float(np.mean([np.linalg.norm(tr - tl), np.linalg.norm(br - bl)]))
    ch = float(np.mean([np.linalg.norm(bl - tl), np.linalg.norm(br - tr)]))
    if ch < 1 or not (0.45 <= cw / ch <= 1.8):
        logger.error("Implausible card aspect ratio: %.2f", cw / ch if ch > 0 else 0)
        return None, "low"

    logger.info("Scanline+RANSAC succeeded. corners=%s", corners.tolist())
    return corners, "high"


# ── Perspective warp ──────────────────────────────────────────────────────────

def _perspective_warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the card region to a flat rectangle."""
    tl, tr, br, bl = corners
    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (width, height))


# ── Inner boundary: card border vs artwork ────────────────────────────────────

def _histogram_peak(hits: list, lo: float, hi: float,
                    bin_size: int = 3, consistency_window_px: int = 15):
    """
    Find the peak position of a hit distribution via histogram mode.

    More accurate than median for border detection: a real card border produces
    a sharp peak at one position; scattered noise produces a wide flat histogram.

    Position: uses fine bins (default 3px) for sub-bin accuracy.
    Consistency: fraction of hits within `consistency_window_px` of the peak.
    Using a window (rather than single-bin fraction) handles the natural ±10px
    variation of hits along a real card border across scanlines.

    Returns (position, consistency) where consistency is 0–1.
    High consistency = clean, tight border. Low = scattered noise (full-art etc).
    """
    if not hits:
        return (lo + hi) / 2.0, 0.0
    hits_arr = np.asarray(hits, dtype=np.float32)
    bins = np.arange(lo, hi + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([lo, hi])
    counts, edges = np.histogram(hits_arr, bins=bins)
    peak_bin = int(np.argmax(counts))
    position = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
    near_peak = float(np.sum(np.abs(hits_arr - position) <= consistency_window_px))
    consistency = near_peak / len(hits)
    return position, consistency


def _collect_inner_hits(mag: np.ndarray, threshold: int,
                        x_lo: int, x_hi: int, y_lo: int, y_hi: int,
                        row_start: int, row_end: int,
                        col_start: int, col_end: int,
                        use_first_hit: bool = False):
    """
    Collect hit positions for all 4 sides within the search windows.

    use_first_hit=False (default): argmax — finds the strongest gradient in the
        window. Good for bordered cards where the border has the strongest signal.
    use_first_hit=True: first threshold-exceeding pixel from the edge inward.
        Used for the narrow-window saturation pre-pass where the window is small
        enough that the first hit IS the border, without interior artwork stealing
        the argmax.
    """
    ch, cw = mag.shape
    left_hits, right_hits, top_hits, bot_hits = [], [], [], []

    for row in range(row_start, row_end):
        strip = mag[row, x_lo: x_hi]
        if strip.max() > threshold:
            if use_first_hit:
                idx = int(np.argmax(strip > threshold))
            else:
                idx = int(np.argmax(strip))
            left_hits.append(float(idx + x_lo))

        strip_r = mag[row, cw - x_hi: cw - x_lo]
        if strip_r.max() > threshold:
            if use_first_hit:
                # Scan from the right edge inward → last exceeding pixel in strip
                idx = int(len(strip_r) - 1 - np.argmax((strip_r > threshold)[::-1]))
            else:
                idx = int(np.argmax(strip_r))
            right_hits.append(float(cw - x_hi + idx))

    for col in range(col_start, col_end):
        strip = mag[y_lo: y_hi, col]
        if strip.max() > threshold:
            if use_first_hit:
                idx = int(np.argmax(strip > threshold))
            else:
                idx = int(np.argmax(strip))
            top_hits.append(float(idx + y_lo))

        strip_b = mag[ch - y_hi: ch - y_lo, col]
        if strip_b.max() > threshold:
            if use_first_hit:
                idx = int(len(strip_b) - 1 - np.argmax((strip_b > threshold)[::-1]))
            else:
                idx = int(np.argmax(strip_b))
            bot_hits.append(float(ch - y_hi + idx))

    return left_hits, right_hits, top_hits, bot_hits


def _find_inner_borders(card_img: np.ndarray):
    """
    Find the 4 inner border widths (px) on a perspective-corrected card image.

    Two-pass cascade:

    Pass 1 — Narrow saturation + first-hit (3–8% window):
        Uses only the saturation channel so the colored border (yellow, blue, etc.)
        stands out from the artwork. Scans first-hit from each edge inward within
        a tight 3–8% zone. This finds the design border before interior elements
        (Pokémon text on card backs, artwork) appear in the search window.
        Handles: standard bordered cards AND card backs with a design rectangle.

    Pass 2 — Combined magnitude + argmax (3–20% window):
        Falls back to the existing approach when the narrow saturation pass fails.
        Handles vintage cards with wider borders (~8–12%).

    Both passes use histogram peak for position accuracy and a consistency check
    to reject full-art holographic noise.

    Returns (left, right, top, bottom, consistency) in pixels, or None if all
    passes fail (full-art or genuinely borderless card).
    """
    ch, cw = card_img.shape[:2]

    # Row/col range: scan the middle 80% to avoid corner noise
    row_start = int(ch * 0.10);  row_end = int(ch * 0.90)
    col_start = int(cw * 0.10);  col_end = int(cw * 0.90)
    min_hit_frac = 0.60
    min_row_hits = int((row_end - row_start) * min_hit_frac)
    min_col_hits = int((col_end - col_start) * min_hit_frac)

    MIN_CONSISTENCY = 0.30  # fraction of hits within ±15px of histogram peak

    blurred = cv2.GaussianBlur(card_img, (5, 5), 0)

    # ── Pass 1: saturation channel, narrow 3–8% window, first-hit ────────────
    # Why saturation: colored borders (yellow, blue) pop in saturation even when
    # grayscale contrast is low. Why narrow+first-hit: the actual design border
    # is always at 3–6%; interior artwork and text (10–20%) never enter the window
    # so the first threshold-exceeding gradient IS the border.
    sat = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)[:, :, 1]
    mag_sat = _sobel_magnitude(sat)

    x_lo_n = int(cw * 0.03);  x_hi_n = int(cw * 0.08)
    y_lo_n = int(ch * 0.03);  y_hi_n = int(ch * 0.08)

    result_hits = None
    for thresh_frac in [0.20, 0.15, 0.10, 0.07]:
        threshold = max(int(mag_sat.max() * thresh_frac), 10)
        lh, rh, th, bh = _collect_inner_hits(
            mag_sat, threshold,
            x_lo_n, x_hi_n, y_lo_n, y_hi_n,
            row_start, row_end, col_start, col_end,
            use_first_hit=True,
        )
        if not (len(lh) >= min_row_hits and len(rh) >= min_row_hits and
                len(th) >= min_col_hits and len(bh) >= min_col_hits):
            continue
        _, lc = _histogram_peak(lh, x_lo_n, x_hi_n)
        _, rc = _histogram_peak(rh, cw - x_hi_n, cw - x_lo_n)
        _, tc = _histogram_peak(th, y_lo_n, y_hi_n)
        _, bc = _histogram_peak(bh, ch - y_hi_n, ch - y_lo_n)
        if min(lc, rc, tc, bc) >= MIN_CONSISTENCY:
            logger.info("Pass 1 (sat/narrow) succeeded at thresh=%.2f  c: L=%.2f R=%.2f T=%.2f B=%.2f",
                        thresh_frac, lc, rc, tc, bc)
            result_hits = (lh, rh, th, bh, x_lo_n, x_hi_n, y_lo_n, y_hi_n)
            break
        logger.debug("Pass 1 thresh=%.2f count OK, consistency low (L=%.2f R=%.2f T=%.2f B=%.2f)",
                     thresh_frac, lc, rc, tc, bc)

    # ── Pass 2: combined magnitude + CLAHE, wide 3–20% window, argmax ────────
    # Fallback for vintage cards with wider borders (8–12%) that fall outside
    # the narrow pass 1 window, and for lower-saturation bordered cards.
    if result_hits is None:
        mag_combined = _combined_magnitude(card_img, use_clahe=True)
        x_lo_w = int(cw * 0.03);  x_hi_w = int(cw * 0.20)
        y_lo_w = int(ch * 0.03);  y_hi_w = int(ch * 0.20)

        for thresh_frac in [0.15, 0.10, 0.07]:
            threshold = max(int(mag_combined.max() * thresh_frac), 10)
            lh, rh, th, bh = _collect_inner_hits(
                mag_combined, threshold,
                x_lo_w, x_hi_w, y_lo_w, y_hi_w,
                row_start, row_end, col_start, col_end,
            )
            if not (len(lh) >= min_row_hits and len(rh) >= min_row_hits and
                    len(th) >= min_col_hits and len(bh) >= min_col_hits):
                continue
            _, lc = _histogram_peak(lh, x_lo_w, x_hi_w)
            _, rc = _histogram_peak(rh, cw - x_hi_w, cw - x_lo_w)
            _, tc = _histogram_peak(th, y_lo_w, y_hi_w)
            _, bc = _histogram_peak(bh, ch - y_hi_w, ch - y_lo_w)
            if min(lc, rc, tc, bc) >= MIN_CONSISTENCY:
                logger.info("Pass 2 (combined/wide) succeeded at thresh=%.2f  c: L=%.2f R=%.2f T=%.2f B=%.2f",
                            thresh_frac, lc, rc, tc, bc)
                result_hits = (lh, rh, th, bh, x_lo_w, x_hi_w, y_lo_w, y_hi_w)
                break
            logger.debug("Pass 2 thresh=%.2f count OK, consistency low (L=%.2f R=%.2f T=%.2f B=%.2f)",
                         thresh_frac, lc, rc, tc, bc)

    if result_hits is None:
        logger.warning("Inner border detection: all passes failed. Card has no detectable inner border.")
        return None

    lh, rh, th, bh, x_lo, x_hi, y_lo, y_hi = result_hits

    # Histogram peak: more accurate than median + free consistency score.
    left_border,  left_c  = _histogram_peak(lh, x_lo, x_hi)
    right_raw,    right_c = _histogram_peak(rh, cw - x_hi, cw - x_lo)
    top_border,   top_c   = _histogram_peak(th, y_lo, y_hi)
    bot_raw,      bot_c   = _histogram_peak(bh, ch - y_hi, ch - y_lo)

    right_border = float(cw - right_raw)
    bot_border   = float(ch - bot_raw)

    # Overall consistency: min across all 4 sides (weakest link).
    consistency = float(min(left_c, right_c, top_c, bot_c))

    # Clamp to the window that was actually used (pass 1 narrow or pass 2 wide)
    min_clamp = x_lo / cw   # same fraction for both x and y (both are 3% min)
    max_clamp_x = x_hi / cw
    max_clamp_y = y_hi / ch

    left_border  = float(np.clip(left_border,  cw * min_clamp, cw * max_clamp_x))
    right_border = float(np.clip(right_border, cw * min_clamp, cw * max_clamp_x))
    top_border   = float(np.clip(top_border,   ch * min_clamp, ch * max_clamp_y))
    bot_border   = float(np.clip(bot_border,   ch * min_clamp, ch * max_clamp_y))

    logger.info("Inner borders (px): L=%.1f R=%.1f T=%.1f B=%.1f  consistency=%.2f",
                left_border, right_border, top_border, bot_border, consistency)
    return left_border, right_border, top_border, bot_border, consistency


# ── Grading helpers ───────────────────────────────────────────────────────────

def _format_ratio(pct: float) -> str:
    left = round(pct)
    return f"{left}/{100 - left}"


def _psa_grade(pct: float, half_width: float) -> bool:
    return abs(pct - 50.0) <= half_width


# ── Debug visualisation ───────────────────────────────────────────────────────

def _warp_with_context(img: np.ndarray, corners: np.ndarray, ctx: int = 160):
    """
    Perspective-warp the card but include `ctx` pixels of the surrounding
    original photo on every side. The card occupies [ctx, ctx] → [ctx+w, ctx+h];
    outside is warped background from the original photo.
    Returns (warped_with_context, card_w, card_h).
    """
    tl, tr, br, bl = corners
    cw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    ch = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([
        [ctx,      ctx],
        [ctx + cw, ctx],
        [ctx + cw, ctx + ch],
        [ctx,      ctx + ch],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (cw + 2 * ctx, ch + 2 * ctx)), cw, ch


def draw_borders_debug(img: np.ndarray, corners: np.ndarray,
                       result: CenteringResult, max_width: int = 1600) -> np.ndarray:
    """
    Render the _4_borders debug image: warped card with photo context, inner
    border rectangle, measurement arrows, and footer summary.
    Identical output to the _4_borders.jpg produced by test_centering.py.
    """
    BG = (32, 32, 32)

    # Scale factor relative to a 600px-wide reference card
    _, ctx_cw, _ = _warp_with_context(img, corners, ctx=0)
    ts = max(1.0, ctx_cw / 600)
    ti = lambda v: max(1, int(round(v * ts)))
    tf = lambda v: v * ts

    CTX4   = ti(90)
    FOOTER = ti(110)
    ctx4, card_w, card_h = _warp_with_context(img, corners, ctx=CTX4)

    full_h = ctx4.shape[0] + FOOTER
    canvas = np.full((full_h, ctx4.shape[1], 3), BG, dtype=np.uint8)
    canvas[:ctx4.shape[0], :] = ctx4

    cv2.line(canvas, (0, ctx4.shape[0]), (canvas.shape[1], ctx4.shape[0]), (55, 55, 55), ti(2))
    cv2.rectangle(canvas, (CTX4, CTX4), (CTX4 + card_w - 1, CTX4 + card_h - 1), (60, 60, 60), ti(1))

    LR_CLR = (80, 150, 255)
    TB_CLR = (255, 190, 80)

    inner_c = result.inner_consistency
    has_border = result.left_px > 0 or result.right_px > 0

    if has_border:
        left_px  = result.left_px
        right_px = result.right_px
        top_px   = result.top_px
        bot_px   = result.bottom_px

        cv2.rectangle(canvas,
                      (CTX4 + left_px,           CTX4 + top_px),
                      (CTX4 + card_w - right_px, CTX4 + card_h - bot_px),
                      (0, 230, 100), ti(3))

        mid_y = CTX4 + card_h // 2
        mid_x = CTX4 + card_w // 2

        def _side_label(text, anchor_x, anchor_y, color, side):
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = tf(0.55)
            fthick = ti(2)
            gap    = ti(8)
            pad    = ti(10)
            (tw, th), bl = cv2.getTextSize(text, font, fscale, fthick)
            if side == 'L':
                bx2 = anchor_x - gap;  bx1 = bx2 - tw - pad * 2
                by1 = anchor_y - th // 2 - pad;  by2 = anchor_y + th // 2 + pad + bl
                tx = bx1 + pad;  ty = anchor_y + th // 2
                cv2.line(canvas, (bx2, anchor_y), (anchor_x, anchor_y), color, ti(1))
            elif side == 'R':
                bx1 = anchor_x + gap;  bx2 = bx1 + tw + pad * 2
                by1 = anchor_y - th // 2 - pad;  by2 = anchor_y + th // 2 + pad + bl
                tx = bx1 + pad;  ty = anchor_y + th // 2
                cv2.line(canvas, (anchor_x, anchor_y), (bx1, anchor_y), color, ti(1))
            elif side == 'T':
                by2 = anchor_y - gap;  by1 = by2 - th - pad * 2
                bx1 = anchor_x - tw // 2 - pad;  bx2 = anchor_x + tw // 2 + pad
                tx = bx1 + pad;  ty = by2 - pad // 2
                cv2.line(canvas, (anchor_x, by2), (anchor_x, anchor_y), color, ti(1))
            else:
                by1 = anchor_y + gap;  by2 = by1 + th + pad * 2
                bx1 = anchor_x - tw // 2 - pad;  bx2 = anchor_x + tw // 2 + pad
                tx = bx1 + pad;  ty = by1 + th + pad // 2
                cv2.line(canvas, (anchor_x, anchor_y), (anchor_x, by1), color, ti(1))
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (28, 28, 28), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, ti(1))
            cv2.putText(canvas, text, (tx, ty), font, fscale, color, fthick, cv2.LINE_AA)

        cv2.arrowedLine(canvas, (CTX4, mid_y), (CTX4 + left_px, mid_y),        LR_CLR, ti(3), tipLength=0.20)
        _side_label(f"L  {left_px}px",  CTX4,              mid_y, LR_CLR, 'L')
        cv2.arrowedLine(canvas, (CTX4 + card_w - 1, mid_y), (CTX4 + card_w - right_px, mid_y), LR_CLR, ti(3), tipLength=0.20)
        _side_label(f"R  {right_px}px", CTX4 + card_w - 1, mid_y, LR_CLR, 'R')
        cv2.arrowedLine(canvas, (mid_x, CTX4), (mid_x, CTX4 + top_px),         TB_CLR, ti(3), tipLength=0.20)
        _side_label(f"T  {top_px}px",   mid_x, CTX4,              TB_CLR, 'T')
        cv2.arrowedLine(canvas, (mid_x, CTX4 + card_h - 1), (mid_x, CTX4 + card_h - bot_px), TB_CLR, ti(3), tipLength=0.20)
        _side_label(f"B  {bot_px}px",   mid_x, CTX4 + card_h - 1, TB_CLR, 'B')

        # Footer
        lr_total = left_px + right_px
        tb_total = top_px  + bot_px
        lr_pct   = left_px / lr_total * 100 if lr_total > 0 else 50.0
        tb_pct   = top_px  / tb_total * 100 if tb_total > 0 else 50.0

        if result.psa10_lr_pass and result.psa10_tb_pass:
            grade_txt, grade_clr = "PSA 10", (50, 200, 50)
        elif result.psa9_lr_pass and result.psa9_tb_pass:
            grade_txt, grade_clr = "PSA 9",  (50, 180, 240)
        else:
            grade_txt, grade_clr = "< PSA 9", (60, 60, 220)

        font     = cv2.FONT_HERSHEY_SIMPLEX
        fy       = ctx4.shape[0] + ti(14)
        canvas_w = canvas.shape[1]

        lr_ratio_txt = f"{round(lr_pct)}/{100 - round(lr_pct)}"
        cv2.putText(canvas, "LR",          (CTX4, fy + ti(24)), font, tf(0.50), (120,120,120), ti(1), cv2.LINE_AA)
        cv2.putText(canvas, lr_ratio_txt,  (CTX4, fy + ti(58)), font, tf(1.0),  LR_CLR,        ti(2), cv2.LINE_AA)
        cv2.putText(canvas, f"{lr_pct:.1f}%", (CTX4, fy + ti(80)), font, tf(0.48), (160,160,160), ti(1), cv2.LINE_AA)

        tx = CTX4 + ti(140)
        tb_ratio_txt = f"{round(tb_pct)}/{100 - round(tb_pct)}"
        cv2.putText(canvas, "TB",          (tx, fy + ti(24)), font, tf(0.50), (120,120,120), ti(1), cv2.LINE_AA)
        cv2.putText(canvas, tb_ratio_txt,  (tx, fy + ti(58)), font, tf(1.0),  TB_CLR,        ti(2), cv2.LINE_AA)
        cv2.putText(canvas, f"{tb_pct:.1f}%", (tx, fy + ti(80)), font, tf(0.48), (160,160,160), ti(1), cv2.LINE_AA)

        cx2      = CTX4 + ti(280)
        cons_clr = (50,200,50) if inner_c >= 0.6 else (50,180,240) if inner_c >= 0.3 else (60,60,220)
        cv2.putText(canvas, "CONSISTENCY", (cx2, fy + ti(24)), font, tf(0.45), (120,120,120), ti(1), cv2.LINE_AA)
        cv2.putText(canvas, f"{inner_c:.2f}", (cx2, fy + ti(58)), font, tf(1.0), cons_clr,   ti(2), cv2.LINE_AA)

        (gtw, gth), gbl = cv2.getTextSize(grade_txt, font, tf(1.1), ti(2))
        gpad = ti(18)
        bx1  = canvas_w - gtw - gpad * 2 - ti(20)
        bx2  = canvas_w - ti(20)
        by1  = fy + ti(8)
        by2  = by1 + gth + gpad * 2 + gbl
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), grade_clr, -1)
        cv2.putText(canvas, grade_txt, (bx1 + gpad, by2 - gpad - gbl), font, tf(1.1), (18,18,18), ti(2), cv2.LINE_AA)

    else:
        fy = ctx4.shape[0] + ti(20)
        cv2.putText(canvas, "No inner border detected", (CTX4, fy + ti(55)),
                    cv2.FONT_HERSHEY_SIMPLEX, tf(0.9), (80, 80, 220), ti(2), cv2.LINE_AA)

    # Resize to max_width
    out_scale = min(1.0, max_width / canvas.shape[1])
    if out_scale < 1.0:
        out_h  = int(canvas.shape[0] * out_scale)
        canvas = cv2.resize(canvas, (max_width, out_h), interpolation=cv2.INTER_AREA)

    return canvas


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_centering(img: np.ndarray, return_debug_inputs: bool = False):
    t0 = time.time()
    h, w = img.shape[:2]
    logger.info("Analyzing image: %dx%d", w, h)

    corners, confidence = _find_card_corners(img)

    if corners is None:
        r = CenteringResult(
            lr_ratio="0/0", tb_ratio="0/0",
            lr_pct_left=0.0, tb_pct_top=0.0,
            left_px=0, right_px=0, top_px=0, bottom_px=0,
            psa10_lr_pass=False, psa10_tb_pass=False,
            psa9_lr_pass=False, psa9_tb_pass=False,
            confidence="card_not_found",
        )
        return (r, img, None) if return_debug_inputs else r

    card    = _perspective_warp(img, corners)
    card_h, card_w = card.shape[:2]

    # Warp quality gate: Pokemon cards are 63×88 mm → ratio ~0.716.
    # A badly-skewed warp (bad RANSAC on a difficult image) will fall outside
    # the 0.60–0.80 range. Flag it rather than silently measure a distorted card.
    _POKEMON_CARD_RATIO = 63.0 / 88.0  # 0.716
    actual_ratio = card_w / card_h if card_h > 0 else 0.0
    if not (0.60 <= actual_ratio <= 0.80):
        logger.warning("Warped card aspect ratio %.3f outside expected range [0.60, 0.80]",
                       actual_ratio)
        confidence = "low"

    inner = _find_inner_borders(card)

    if inner is None:
        # No inner border detected — full-art SIR, card back, or full-bleed card.
        # Return an explicit status rather than fabricating 50/50 data from the
        # 10% fallback, which would produce meaningless PSA pass/fail values.
        logger.warning("Inner border detection failed; card has no measurable inner border.")
        r = CenteringResult(
            lr_ratio="N/A", tb_ratio="N/A",
            lr_pct_left=0.0, tb_pct_top=0.0,
            left_px=0, right_px=0, top_px=0, bottom_px=0,
            psa10_lr_pass=False, psa10_tb_pass=False,
            psa9_lr_pass=False,  psa9_tb_pass=False,
            confidence="no_inner_border",
            card_type="full_art",
            inner_consistency=0.0,
        )
        return (r, img, corners) if return_debug_inputs else r

    left_px, right_px, top_px, bot_px, inner_consistency = inner
    left_px  = max(0, int(round(left_px)))
    right_px = max(0, int(round(right_px)))
    top_px   = max(0, int(round(top_px)))
    bot_px   = max(0, int(round(bot_px)))

    # Downgrade confidence if the inner border histogram was very noisy.
    # 0.25 is well below the minimum real-border consistency (≥0.30) but above
    # full-art noise levels, so this only triggers on genuinely borderline cards.
    if inner_consistency < 0.25 and confidence == "high":
        confidence = "low"

    lr_total = left_px + right_px
    tb_total = top_px  + bot_px

    lr_pct = (left_px / lr_total * 100) if lr_total > 0 else 50.0
    tb_pct = (top_px  / tb_total * 100) if tb_total > 0 else 50.0

    result = CenteringResult(
        lr_ratio=_format_ratio(lr_pct),
        tb_ratio=_format_ratio(tb_pct),
        lr_pct_left=round(lr_pct, 1),
        tb_pct_top=round(tb_pct, 1),
        left_px=left_px, right_px=right_px,
        top_px=top_px,   bottom_px=bot_px,
        psa10_lr_pass=_psa_grade(lr_pct, 5.0),
        psa10_tb_pass=_psa_grade(tb_pct, 5.0),
        psa9_lr_pass=_psa_grade(lr_pct, 10.0),
        psa9_tb_pass=_psa_grade(tb_pct, 10.0),
        confidence=confidence,
        card_type="standard",
        inner_consistency=round(inner_consistency, 3),
    )

    logger.info("Result: LR=%s TB=%s confidence=%s elapsed=%.2fs",
                result.lr_ratio, result.tb_ratio, result.confidence, time.time() - t0)
    return (result, img, corners) if return_debug_inputs else result
