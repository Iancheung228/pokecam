"""
test_centering.py — Run the centering algorithm on one or more images and print results.

Usage:
    python test_centering.py path/to/card.jpg
    python test_centering.py test_images/
    python test_centering.py path/to/card.jpg --debug

Tuning scanline parameters (only regenerates _1_scanline_hits.jpg, fast):
    python test_centering.py path/to/card.jpg --tune
    python test_centering.py path/to/card.jpg --tune --threshold 0.20 --skip 0.05

    --threshold  fraction of max Sobel magnitude to use as hit threshold (default 0.15)
    --skip       fraction of each row/col to ignore at both ends (default 0.10)
"""

import sys
import os
import cv2
import numpy as np

from centering import (
    analyze_centering,
    _find_card_corners,
    _perspective_warp,
    _find_inner_borders,
    _combined_magnitude,
    _scan_row_edges,
    _scan_col_edges,
)


def _draw_scanline_hits(img, mag, threshold: int, skip_frac: float, scale: float):
    """Render scanline hit points onto a copy of img. Returns the annotated image."""
    disp_w = int(img.shape[1] * scale)
    disp_h = int(img.shape[0] * scale)

    left_pts, right_pts = _scan_row_edges(mag, threshold, skip_frac)
    top_pts,  bot_pts   = _scan_col_edges(mag, threshold, skip_frac)

    vis = cv2.resize(img.copy(), (disp_w, disp_h))
    for pts, color in [
        (left_pts,  (255,  80,  80)),
        (right_pts, ( 80, 255,  80)),
        (top_pts,   ( 80,  80, 255)),
        (bot_pts,   (255, 255,  80)),
    ]:
        if pts is None:
            continue
        for pt in pts[::2]:
            vis[int(pt[1] * scale), int(pt[0] * scale)] = color

    counts = {
        "left":   len(left_pts)  if left_pts  is not None else 0,
        "right":  len(right_pts) if right_pts is not None else 0,
        "top":    len(top_pts)   if top_pts   is not None else 0,
        "bottom": len(bot_pts)   if bot_pts   is not None else 0,
    }
    legend_colors = {
        "left": (255,80,80), "right": (80,255,80),
        "top": (80,80,255), "bottom": (255,255,80),
    }
    for i, (name, col) in enumerate(legend_colors.items()):
        cv2.putText(vis, f"{name}: {counts[name]}pts",
                    (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    cv2.putText(vis, f"threshold={threshold}  skip={skip_frac:.2f}",
                (10, disp_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def tune_image(path: str, threshold_frac: float, skip_frac: float):
    """Regenerate only the scanline hits image with custom parameters."""
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not load: {path}")
        return

    mag       = _combined_magnitude(img)
    threshold = max(int(mag.max() * threshold_frac), 10)

    scale   = min(1.0, 1200 / max(img.shape[:2]))
    vis     = _draw_scanline_hits(img, mag, threshold, skip_frac, scale)

    stem    = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(path), f"debug_{stem}")
    os.makedirs(out_dir, exist_ok=True)
    out     = os.path.join(out_dir, f"{stem}_1_scanline_hits.jpg")
    cv2.imwrite(out, vis)
    print(f"threshold={threshold} ({threshold_frac:.2f}×max)  skip={skip_frac:.2f}  → {out}")


def save_debug_images(path: str, out_dir: str, threshold_frac: float = 0.15, skip_frac: float = 0.10):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not load image: {path}")
        return

    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)

    h, w  = img.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    disp_w = int(w * scale)
    disp_h = int(h * scale)

    mag       = _combined_magnitude(img)
    threshold = max(int(mag.max() * threshold_frac), 10)

    # ── Step 1: Scanline edge hits ────────────────────────────────────────────
    vis = _draw_scanline_hits(img, mag, threshold, skip_frac, scale)
    p1  = os.path.join(out_dir, f"{stem}_1_scanline_hits.jpg")
    cv2.imwrite(p1, vis)
    print(f"  Saved: {p1}")

    # ── Step 1a: Saturation channel Sobel ────────────────────────────────────
    sat     = cv2.cvtColor(cv2.GaussianBlur(img, (5,5), 0), cv2.COLOR_BGR2HSV)[:, :, 1]
    sat_sx  = cv2.Sobel(sat.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sat_sy  = cv2.Sobel(sat.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    sat_mag = np.sqrt(sat_sx**2 + sat_sy**2)
    if sat_mag.max() > 0:
        sat_mag = sat_mag / sat_mag.max() * 255
    sat_vis = cv2.applyColorMap(sat_mag.astype(np.uint8), cv2.COLORMAP_INFERNO)
    sat_vis = cv2.resize(sat_vis, (disp_w, disp_h))
    cv2.putText(sat_vis, "saturation Sobel", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    p1a = os.path.join(out_dir, f"{stem}_1a_saturation_sobel.jpg")
    cv2.imwrite(p1a, sat_vis)
    print(f"  Saved: {p1a}")

    # ── Step 2: RANSAC corners ────────────────────────────────────────────────
    corners, confidence = _find_card_corners(img)
    corners_vis = cv2.resize(img.copy(), (disp_w, disp_h))

    if corners is not None:
        scaled = (corners * scale).astype(np.int32)
        # Expand 4px outward so the line sits just outside the card edge
        centroid = scaled.mean(axis=0)
        expanded = (scaled + np.sign(scaled - centroid) * 4).astype(np.int32)
        cv2.polylines(corners_vis, [expanded], isClosed=True, color=(0, 255, 0), thickness=1)
        for i, (cx, cy) in enumerate(scaled):
            cv2.circle(corners_vis, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(corners_vis, ["TL", "TR", "BR", "BL"][i], (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        label, color = f"Detected ({confidence})", (0, 255, 0)
    else:
        label, color = "NO CARD DETECTED", (0, 0, 255)

    cv2.putText(corners_vis, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    p2 = os.path.join(out_dir, f"{stem}_2_card_corners.jpg")
    cv2.imwrite(p2, corners_vis)
    print(f"  Saved: {p2}")

    if corners is None:
        print("  Cannot continue — card not found.")
        return

    # ── Shared helpers ────────────────────────────────────────────────────────
    BG      = (32, 32, 32)      # dark charcoal background
    PAD     = 48                # pixels of breathing room around every card image
    CTX     = 160               # extra context pixels showing original photo background

    def _warp_with_context(src_img, card_corners, ctx=CTX):
        """
        Perspective-warp the card but include `ctx` pixels of the surrounding
        original photo on every side.  The card occupies the interior rectangle
        [ctx, ctx] → [ctx+w, ctx+h]; outside that is warped background from the
        original photo, giving a clear visual of where the card boundary sits.
        """
        tl, tr, br, bl = card_corners
        cw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        ch = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        dst = np.array([
            [ctx,      ctx],
            [ctx + cw, ctx],
            [ctx + cw, ctx + ch],
            [ctx,      ctx + ch],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(card_corners.astype(np.float32), dst)
        return cv2.warpPerspective(src_img, M, (cw + 2 * ctx, ch + 2 * ctx)), cw, ch

    def _with_bg(card_img):
        """Return card_img centred on a PAD-wide dark background."""
        return cv2.copyMakeBorder(card_img, PAD, PAD, PAD, PAD,
                                  cv2.BORDER_CONSTANT, value=BG)

    def _badge(canvas, text, cx, cy, txt_color=(255, 255, 255),
               bg_color=(20, 20, 20), font_scale=0.65, thickness=2, padding=7):
        """Draw `text` centred at (cx, cy) with a filled dark-background badge."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        x0 = cx - tw // 2 - padding
        y0 = cy - th // 2 - padding
        x1 = cx + tw // 2 + padding
        y1 = cy + th // 2 + padding + bl
        cv2.rectangle(canvas, (x0, y0), (x1, y1), bg_color, -1)
        cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2),
                    font, font_scale, txt_color, thickness, cv2.LINE_AA)

    # ── Step 3: Warped card (with real photo context around edges) ───────────
    card = _perspective_warp(img, corners)
    ctx_img, ctx_cw, ctx_ch = _warp_with_context(img, corners)
    # Green rectangle showing exactly where the detected card boundary sits
    cv2.rectangle(ctx_img, (CTX, CTX), (CTX + ctx_cw - 1, CTX + ctx_ch - 1),
                  (0, 220, 80), 2)
    p3 = os.path.join(out_dir, f"{stem}_3_warped.jpg")
    cv2.imwrite(p3, ctx_img)
    print(f"  Saved: {p3}")

    # ── Step 3b: Scanline hits on warped card ────────────────────────────────
    card_h, card_w = card.shape[:2]
    card_mag    = _combined_magnitude(card, use_clahe=True)
    card_thresh = max(int(card_mag.max() * 0.15), 10)

    # Mirror the exact search window used by _find_inner_borders (narrow pass: 3–8%)
    x_lo = int(card_w * 0.03);  x_hi = int(card_w * 0.08)
    y_lo = int(card_h * 0.03);  y_hi = int(card_h * 0.08)

    # Saturation Sobel of warped card with search window overlaid
    card_sat     = cv2.cvtColor(cv2.GaussianBlur(card, (5,5), 0), cv2.COLOR_BGR2HSV)[:, :, 1]
    sat_sx       = cv2.Sobel(card_sat.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sat_sy       = cv2.Sobel(card_sat.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    card_sat_mag = np.sqrt(sat_sx**2 + sat_sy**2)
    if card_sat_mag.max() > 0:
        card_sat_mag = card_sat_mag / card_sat_mag.max() * 255
    sat_color = cv2.applyColorMap(card_sat_mag.astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.rectangle(sat_color, (x_lo, y_lo), (card_w - x_lo, card_h - y_lo), (100, 100, 100), 1)
    cv2.rectangle(sat_color, (x_hi, y_hi), (card_w - x_hi, card_h - y_hi), (100, 100, 100), 1)
    cv2.putText(sat_color, "sat Sobel  gray boxes=search window",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    sat_padded = _with_bg(sat_color)
    p3b_sat = os.path.join(out_dir, f"{stem}_3b_sat_sobel.jpg")
    cv2.imwrite(p3b_sat, sat_padded)
    print(f"  Saved: {p3b_sat}")

    row_start, row_end = int(card_h * 0.10), int(card_h * 0.90)
    col_start, col_end = int(card_w * 0.10), int(card_w * 0.90)

    card_hits_vis = card.copy()
    cv2.rectangle(card_hits_vis, (x_lo, y_lo), (card_w - x_lo, card_h - y_lo), (60, 60, 60), 1)
    cv2.rectangle(card_hits_vis, (x_hi, y_hi), (card_w - x_hi, card_h - y_hi), (60, 60, 60), 1)

    for row in range(row_start, row_end, 2):
        strip = card_mag[row, x_lo: x_hi]
        if strip.max() > card_thresh:
            card_hits_vis[row, np.argmax(strip) + x_lo] = (255, 80, 80)
        strip_r = card_mag[row, card_w - x_hi: card_w - x_lo]
        if strip_r.max() > card_thresh:
            card_hits_vis[row, card_w - x_hi + np.argmax(strip_r)] = (80, 255, 80)

    for col in range(col_start, col_end, 2):
        strip = card_mag[y_lo: y_hi, col]
        if strip.max() > card_thresh:
            card_hits_vis[np.argmax(strip) + y_lo, col] = (80, 80, 255)
        strip_b = card_mag[card_h - y_hi: card_h - y_lo, col]
        if strip_b.max() > card_thresh:
            card_hits_vis[card_h - y_hi + np.argmax(strip_b), col] = (255, 255, 80)

    inner_preview = _find_inner_borders(card)
    if inner_preview is not None:
        lp, rp, tp, bp, _ic = inner_preview
        lp, rp, tp, bp = int(round(lp)), int(round(rp)), int(round(tp)), int(round(bp))
        cv2.rectangle(card_hits_vis, (lp, tp), (card_w - rp, card_h - bp), (0, 255, 80), 2)
    else:
        cv2.putText(card_hits_vis, "no inner border", (10, card_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

    legend = [("left", (255,80,80)), ("right", (80,255,80)),
              ("top", (80,80,255)), ("bottom", (255,255,80))]
    for i, (name, col) in enumerate(legend):
        cv2.putText(card_hits_vis, name, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    hits_padded = _with_bg(card_hits_vis)
    p3b = os.path.join(out_dir, f"{stem}_3b_inner_scanline.jpg")
    cv2.imwrite(p3b, hits_padded)
    print(f"  Saved: {p3b}")

    # ── Step 4: Inner border measurements ────────────────────────────────────
    inner = _find_inner_borders(card)

    # Base: context warp (real photo background) + footer strip for summary
    FOOTER    = 90
    ctx4, cw4, ch4 = _warp_with_context(img, corners)
    full_h    = ctx4.shape[0] + FOOTER
    canvas    = np.full((full_h, ctx4.shape[1], 3), BG, dtype=np.uint8)
    canvas[:ctx4.shape[0], :] = ctx4
    card_h, card_w = ch4, cw4   # card dims within canvas (offset by CTX)

    # Green card-boundary guide
    cv2.rectangle(canvas, (CTX, CTX), (CTX + card_w - 1, CTX + card_h - 1),
                  (70, 70, 70), 1)

    if inner is not None:
        left_px, right_px, top_px, bot_px, inner_c = inner
        left_px  = int(round(left_px))
        right_px = int(round(right_px))
        top_px   = int(round(top_px))
        bot_px   = int(round(bot_px))

        # Bright green inner-border rectangle (offset by CTX into canvas coords)
        cv2.rectangle(canvas,
                      (CTX + left_px,            CTX + top_px),
                      (CTX + card_w - right_px,  CTX + card_h - bot_px),
                      (0, 230, 100), 2)

        mid_y = CTX + card_h // 2
        mid_x = CTX + card_w // 2

        LR_CLR  = (80, 150, 255)   # warm orange
        TB_CLR  = (255, 190, 80)   # sky blue
        ARROW_T = 2
        TIP_LEN = 0.25             # tipLength relative to arrow length

        # Each arrow runs exactly from the card edge pixel to the inner border
        # pixel — no badge interrupts it.  The label lives entirely in the CTX
        # background zone and is connected to the arrow's tail by a short tick.
        #
        # Layout:
        #   Left  : arrow (CTX → CTX+left_px), label in left background strip
        #   Right : arrow (CTX+card_w-1 → CTX+card_w-right_px), label right bg
        #   Top   : arrow (CTX → CTX+top_px), label in top background strip
        #   Bottom: arrow (CTX+card_h-1 → CTX+card_h-bot_px), label bottom bg

        TICK = 8   # length of the small perpendicular tick at arrow tail

        def _side_label(text, anchor_x, anchor_y, color, side):
            """
            Draw label badge in the background zone beside `side` ('L','R','T','B').
            A short tick line connects the badge to the card edge.
            """
            font  = cv2.FONT_HERSHEY_SIMPLEX
            fscale, fthick = 0.60, 2
            (tw, th), bl = cv2.getTextSize(text, font, fscale, fthick)
            pad = 6
            if side == 'L':
                # badge to the left of card edge
                bx2 = anchor_x - 6
                bx1 = bx2 - tw - pad * 2
                by1 = anchor_y - th // 2 - pad
                by2 = anchor_y + th // 2 + pad + bl
                tx  = bx1 + pad
                ty  = anchor_y + th // 2
                # tick: short horizontal line from badge right edge to card edge
                cv2.line(canvas, (bx2, anchor_y), (anchor_x, anchor_y), color, 1)
            elif side == 'R':
                bx1 = anchor_x + 6
                bx2 = bx1 + tw + pad * 2
                by1 = anchor_y - th // 2 - pad
                by2 = anchor_y + th // 2 + pad + bl
                tx  = bx1 + pad
                ty  = anchor_y + th // 2
                cv2.line(canvas, (anchor_x, anchor_y), (bx1, anchor_y), color, 1)
            elif side == 'T':
                by2 = anchor_y - 6
                by1 = by2 - th - pad * 2
                bx1 = anchor_x - tw // 2 - pad
                bx2 = anchor_x + tw // 2 + pad
                tx  = bx1 + pad
                ty  = by2 - pad // 2
                cv2.line(canvas, (anchor_x, by2), (anchor_x, anchor_y), color, 1)
            else:  # 'B'
                by1 = anchor_y + 6
                by2 = by1 + th + pad * 2
                bx1 = anchor_x - tw // 2 - pad
                bx2 = anchor_x + tw // 2 + pad
                tx  = bx1 + pad
                ty  = by1 + th + pad // 2
                cv2.line(canvas, (anchor_x, anchor_y), (anchor_x, by1), color, 1)

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (18, 18, 18), -1)
            cv2.putText(canvas, text, (tx, ty), font, fscale, color, fthick, cv2.LINE_AA)

        # ── Left ──────────────────────────────────────────────────────────────
        cv2.arrowedLine(canvas,
                        (CTX,          mid_y),
                        (CTX + left_px, mid_y),
                        LR_CLR, ARROW_T, tipLength=TIP_LEN)
        _side_label(f"L  {left_px}px", CTX, mid_y, LR_CLR, 'L')

        # ── Right ─────────────────────────────────────────────────────────────
        cv2.arrowedLine(canvas,
                        (CTX + card_w - 1,       mid_y),
                        (CTX + card_w - right_px, mid_y),
                        LR_CLR, ARROW_T, tipLength=TIP_LEN)
        _side_label(f"R  {right_px}px", CTX + card_w - 1, mid_y, LR_CLR, 'R')

        # ── Top ───────────────────────────────────────────────────────────────
        cv2.arrowedLine(canvas,
                        (mid_x, CTX),
                        (mid_x, CTX + top_px),
                        TB_CLR, ARROW_T, tipLength=TIP_LEN)
        _side_label(f"T  {top_px}px", mid_x, CTX, TB_CLR, 'T')

        # ── Bottom ────────────────────────────────────────────────────────────
        cv2.arrowedLine(canvas,
                        (mid_x, CTX + card_h - 1),
                        (mid_x, CTX + card_h - bot_px),
                        TB_CLR, ARROW_T, tipLength=TIP_LEN)
        _side_label(f"B  {bot_px}px", mid_x, CTX + card_h - 1, TB_CLR, 'B')

        # ── Footer summary ────────────────────────────────────────────────────
        lr_total = left_px + right_px
        tb_total = top_px  + bot_px
        lr_pct   = left_px / lr_total * 100 if lr_total > 0 else 50.0
        tb_pct   = top_px  / tb_total * 100 if tb_total > 0 else 50.0

        psa10_lr = abs(lr_pct - 50) <= 5
        psa10_tb = abs(tb_pct - 50) <= 5
        psa9_lr  = abs(lr_pct - 50) <= 10
        psa9_tb  = abs(tb_pct - 50) <= 10

        if psa10_lr and psa10_tb:
            grade_txt, grade_clr = "PSA 10", (80, 220, 80)
        elif psa9_lr and psa9_tb:
            grade_txt, grade_clr = "PSA 9",  (80, 200, 255)
        else:
            grade_txt, grade_clr = "< PSA 9", (80, 80, 255)

        canvas_w = canvas.shape[1]
        fy   = ctx4.shape[0] + 8   # footer y-start (just below the warped region)
        font = cv2.FONT_HERSHEY_SIMPLEX

        lr_ratio = f"{round(lr_pct)}/{100 - round(lr_pct)}"
        cv2.putText(canvas, "LR", (CTX, fy + 22),
                    font, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, lr_ratio, (CTX, fy + 48),
                    font, 0.9, LR_CLR, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{lr_pct:.1f}%", (CTX, fy + 70),
                    font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        tb_ratio = f"{round(tb_pct)}/{100 - round(tb_pct)}"
        tx = CTX + 120
        cv2.putText(canvas, "TB", (tx, fy + 22),
                    font, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, tb_ratio, (tx, fy + 48),
                    font, 0.9, TB_CLR, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{tb_pct:.1f}%", (tx, fy + 70),
                    font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cx2 = CTX + 240
        cv2.putText(canvas, "CONSISTENCY", (cx2, fy + 22),
                    font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        cons_clr = (80, 220, 80) if inner_c >= 0.6 else (80, 200, 255) if inner_c >= 0.3 else (80, 80, 255)
        cv2.putText(canvas, f"{inner_c:.2f}", (cx2, fy + 48),
                    font, 0.9, cons_clr, 2, cv2.LINE_AA)

        gx = canvas_w - 138
        cv2.rectangle(canvas, (gx - 8, fy + 4), (gx + 130, fy + 58),
                      grade_clr, -1)
        cv2.putText(canvas, grade_txt, (gx, fy + 42),
                    font, 1.0, (20, 20, 20), 2, cv2.LINE_AA)

    else:
        fy = ctx4.shape[0] + 20
        cv2.putText(canvas, "No inner border detected", (CTX, fy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2, cv2.LINE_AA)

    p4 = os.path.join(out_dir, f"{stem}_4_borders.jpg")
    cv2.imwrite(p4, canvas)
    print(f"  Saved: {p4}")


def test_image(path: str, debug: bool = False, threshold_frac: float = 0.15, skip_frac: float = 0.10):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not load: {path}")
        return

    h, w = img.shape[:2]
    print(f"\n{'='*50}")
    print(f"File : {os.path.basename(path)}  ({w}x{h})")

    result = analyze_centering(img)

    print(f"Confidence : {result.confidence}  |  Card type: {result.card_type}  |  Inner consistency: {result.inner_consistency:.2f}")
    print(f"LR: {result.lr_ratio}  ({result.lr_pct_left:.1f}% left)")
    print(f"TB: {result.tb_ratio}  ({result.tb_pct_top:.1f}% top)")
    print(f"Borders: L={result.left_px}  R={result.right_px}  T={result.top_px}  B={result.bottom_px}")
    print(f"PSA 10: LR={'PASS' if result.psa10_lr_pass else 'FAIL'}  "
          f"TB={'PASS' if result.psa10_tb_pass else 'FAIL'}")
    print(f"PSA 9:  LR={'PASS' if result.psa9_lr_pass else 'FAIL'}  "
          f"TB={'PASS' if result.psa9_tb_pass else 'FAIL'}")

    if result.confidence == "card_not_found":
        grade = "Card not detected"
    elif result.confidence == "no_inner_border":
        grade = "Full art / no inner border — centering unmeasurable from photo"
    elif result.psa10_lr_pass and result.psa10_tb_pass:
        grade = "PSA 10 candidate"
    elif result.psa9_lr_pass and result.psa9_tb_pass:
        grade = "PSA 9 candidate"
    else:
        grade = "Below PSA 9"
    print(f"Grade: {grade}")

    if debug:
        stem    = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(os.path.dirname(path), f"debug_{stem}")
        print(f"\nDebug → {out_dir}/")
        save_debug_images(path, out_dir, threshold_frac, skip_frac)


def _get_arg(args, flag, default):
    """Parse --flag value from args list."""
    try:
        return type(default)(args[args.index(flag) + 1])
    except (ValueError, IndexError):
        return default


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    tune           = "--tune"  in args
    debug          = "--debug" in args
    threshold_frac = _get_arg(args, "--threshold", 0.15)
    skip_frac      = _get_arg(args, "--skip",      0.10)
    target         = args[0]

    if tune:
        # Fast path: only regenerate _1_scanline_hits.jpg
        if os.path.isdir(target):
            exts  = {".jpg", ".jpeg", ".png"}
            files = sorted(
                os.path.join(target, f)
                for f in os.listdir(target)
                if os.path.splitext(f)[1].lower() in exts
            )
            for f in files:
                tune_image(f, threshold_frac, skip_frac)
        else:
            tune_image(target, threshold_frac, skip_frac)
        return

    if os.path.isdir(target):
        exts  = {".jpg", ".jpeg", ".png"}
        files = sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not files:
            print(f"No images found in {target}")
            sys.exit(1)
        for f in files:
            test_image(f, debug, threshold_frac, skip_frac)
    else:
        test_image(target, debug, threshold_frac, skip_frac)


if __name__ == "__main__":
    main()
