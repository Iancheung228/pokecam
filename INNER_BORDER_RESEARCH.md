# Inner Border Detection Research

Findings from experiments on the `_find_inner_borders` pipeline.
All experiments run against the same baseline and test set: IMG_9970–9972, IMG_9992–9995.

---

## Baseline

Commit: `9e1f4ea` — "Snapshot before distance-penalty inner border experiment"

Two-pass cascade:
- **Pass 1**: Saturation Sobel, narrow 3–8% window, `use_first_hit=True`
- **Pass 2**: Combined magnitude (gray + sat + LAB-b*) + CLAHE, wide 3–20% window, `argmax`
- Both passes: histogram peak across scanline hits → border position

Baseline results on the two problem cards:
- **IMG_9992** (Pikachu EX SIR 219/191 holographic): `L=86 R=106 T=94 B=94` — B visually overshoots inner border by ~4px
- **IMG_9994** (Mega Pikachu EX SIR holographic): `L=66 R=92 T=88 B=86` — L visually undershoots inner border by ~4–6px

---

## Observed Problem

On holographic / SIR cards, the detected inner border line is off by ~4–6px on one or more sides.

The two failure modes are opposite in direction:
- **Overshoot** (B on 9992): detected line sits further inside the card than the real inner border
- **Undershoot** (L on 9994): detected line sits closer to the outer edge than the real inner border

Both failures occur on holographic SIR cards with consistency scores in the 0.41–0.68 range.
Non-holographic cards (9971, 9972) have consistency ≥ 0.90 and are accurate.

Root cause hypothesis: holographic shimmer creates **spatially-consistent** Sobel gradient clusters in the search window that compete with or displace the real inner border cluster in the histogram. The shimmer gradients are not random — they're structured enough to pass the consistency check.

---

## Experiments Tried

### 1. Distance-penalty argmax (normalised)

**What**: Replace `use_first_hit` and pure `argmax` with a penalised score:
`score = magnitude - weight * fractional_distance_from_card_edge`
where fractional distance is normalised 0→1 across the strip so weight has card-resolution-independent meaning.

**Both passes** used `distance_weight=2.0` (raw px), then `distance_weight=0.5/1.0` (raw px), then `distance_weight=30.0` (normalised 0–1 fraction).

**Result**: Unstable across the test set regardless of weight value.
- At high weights: behaves like first-hit, collapses borders on some cards (9972 R: 118→66)
- At low weights: no meaningful effect
- At normalised weight=30: 9993 and 9994 return `no_inner_border` (consistency check fails entirely)

**Conclusion**: A fixed per-scanline penalty doesn't generalise because the issue is which **histogram cluster** wins, not which pixel within a scanline wins. Shifting argmax per scanline doesn't reliably shift the cluster.

---

### 2. Non-Local Means (NL-means) denoising

**What**: `cv2.fastNlMeansDenoisingColored(blurred, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)` applied before both passes.

**Result**: Mixed, not viable.
- Some improvements (9970 LR: 42/58 → 49/51), but 9994-R collapsed (92→66), 9992-B unchanged at 94.
- The shimmer on SIR cards is too spatially structured for NL-means to distinguish from a real edge — it just shifts which wrong cluster wins.

---

### 3. Structure tensor coherence

**What**: Compute coherence map `(λ₁−λ₂)²/(λ₁+λ₂)²` from the structure tensor — pixels where gradients all point the same direction (real edges) get high coherence; pixels where gradients scatter (shimmer noise) get low coherence. Multiply Sobel magnitude by coherence map before both passes.

**Result**: Broke multiple cards.
- 9995: `no_inner_border` (was PSA 9)
- 9992 B: 94 → 112 (worse)
- 9994: unchanged

**Why it failed**: The inner border on holographic SIR cards is itself slightly iridescent. Its gradient direction is not perfectly coherent across a 9px window, so the coherence map partially suppresses the border signal along with the shimmer. The border and shimmer are not distinguishable by gradient coherence on these cards.

---

### 4. Color segmentation (dominant hue mask)

**What**: In the 3–15% edge zone, find the dominant hue among high-saturation pixels. Build a binary mask of pixels within ±18° of that hue. Scan innermost extent of mask per scanline → border position.

**Result**: Not viable.
- On cards where it "succeeded" (9970, 9971, 9993, 9995), it detected the **outer card frame** color, not the inner border line — measurements were 3–4× too large (e.g. L=250 instead of ~52)
- On the holographic problem cards (9992, 9994), it correctly fell through (no dominant hue found in edge zones) — confirming that holographic shimmer prevents a clean dominant hue from forming in the border region

---

### 5. MobileSAM segmentation

**What**: After perspective warp, use MobileSAM (TinyViT backbone, `vit_t`) with point prompts:
- 12 foreground points at 5% depth from each edge (top/mid/bottom of each side) = border region
- 3 background points at card center = artwork

Measure inner border from innermost extent of resulting mask.

**Setup**: System Python 3.9, `mobile_sam` from GitHub, `torch==2.2.2`, `timm`, `numpy<2`. Model: `mobile_sam.pt` (~39MB).

**Result**: Not viable for this use case.
- SAM segmented the large holographic artwork area as a single semantic region, not the thin border strip
- Output: `L=440 R=438 T=580 B=578` on a 2202×2896px warped card (≈20% depth, real border is ≈4%)
- The mask covered nearly the entire card interior with the cyan overlay

**Why it failed**: MobileSAM is designed to segment **semantic regions** (objects, visually coherent areas). The inner border is a thin ~3–4% perimeter strip — not a natural semantic region. SAM has no concept of "thin frame around card." Prompt placement at 5% depth landed inside the artwork on this card whose border ends at ~4%.

---

## What Works Reliably

- Non-holographic cards with matte borders (9971, 9972): baseline accuracy is excellent, consistency ≥ 0.90
- The outer card corner detection (RANSAC scanline) is stable across all tested cards
- The perspective warp is accurate — errors are in inner border detection, not geometry

## What Does Not Work on Holographic SIR Cards

Any approach that relies on gradient strength, gradient direction, or color frequency in the edge zone is fooled by holographic shimmer on these cards. Shimmer creates structured, spatially consistent signals that pass consistency checks.

## Untried Directions

- **Fine-tuned binary segmentation** (U-Net style, border=1/artwork=0): would require labeled training data (~50+ cards). Most likely to succeed but needs labeling pipeline.
- **GrabCut with geometric seeds**: not yet tested. Seeds outer 3% as definite foreground (border), inner 50% as definite background (artwork). Already in `cv2.grabCut`.
- **Widening PSA tolerance for low-consistency cards**: pragmatic fallback — if `inner_consistency < 0.5`, widen acceptance window or report "low confidence" grade rather than hard pass/fail.
