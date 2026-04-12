# Pokecam

A web app that uses computer vision to measure Pokemon card centering for PSA grading. Take a photo of a card and get instant Left/Right and Top/Bottom centering ratios with PSA 10 / PSA 9 pass/fail grades.

**Live app:** https://pokecam-app.web.app

---

## Repo Layout

```
pokecam/
├── backend/                    ← Python FastAPI server (deployed on Cloud Run)
│   ├── centering.py            ← Core OpenCV algorithm — edit this to tune accuracy
│   ├── main.py                 ← API endpoints (/health, /analyze)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── test_centering.py       ← CLI tool for testing the algorithm locally
│   └── test_images/            ← Sample cards + debug output folders
├── mobile/
│   └── pokecam_app/            ← Flutter project (web build deployed to Firebase)
│       ├── lib/
│       │   ├── main.dart
│       │   ├── screens/camera_screen.dart   ← Camera UI + result page
│       │   ├── services/api_service.dart    ← All HTTP calls
│       │   └── widgets/result_card.dart     ← Result display + debug image
│       └── web/                             ← Web entry point (index.html, icons)
├── firebase/
│   └── .firebaserc
├── firebase.json               ← Firebase Hosting config
├── CLAUDE.md                   ← Instructions for Claude Code
└── INNER_BORDER_RESEARCH.md    ← Research notes on holographic card edge detection
```

---

## How It Works

1. User opens the web app in Safari and taps **Scan a Card**
2. iPhone camera opens via `image_picker`
3. Photo is uploaded to the Cloud Run backend as a multipart POST to `/analyze`
4. Backend runs the OpenCV pipeline:
   - Detects card outer corners via RANSAC scanline fitting
   - Perspective-warps the card flat
   - Detects inner border (card design edge) via two-pass saturation + gradient scan
   - Computes L/R and T/B centering ratios and PSA grades
5. Response includes measurements + a base64 debug image (the `_4_borders` annotated view)
6. Flutter displays the annotated card image and result card

---

## Local Development

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run server locally
uvicorn main:app --reload --port 8080

# Test with a card image
curl -X POST -F "file=@test_images/IMG_9971.JPG" http://localhost:8080/analyze

# Test the algorithm in isolation (prints results, no server needed)
python test_centering.py test_images/IMG_9971.JPG

# Generate debug images (saves to test_images/debug_IMG_9971/)
python test_centering.py test_images/IMG_9971.JPG --debug

# Run on all images in a folder
python test_centering.py test_images/ --debug
```

The `--debug` flag saves 7 images per card showing each pipeline step:
- `_1_scanline_hits.jpg` — outer edge scan hits
- `_1a_saturation_sobel.jpg` — saturation channel gradients
- `_2_card_corners.jpg` — RANSAC corner detection
- `_3_warped.jpg` — perspective-corrected card
- `_3b_sat_sobel.jpg` — inner border saturation gradients
- `_3b_inner_scanline.jpg` — inner border scan hits
- `_4_borders.jpg` — final measurements (identical to what the app shows)

### Flutter Web

```bash
cd mobile/pokecam_app
flutter pub get
flutter build web --release   # output goes to build/web/
```

To point the app at a local backend instead of Cloud Run, edit `api_service.dart`:
```dart
static const String baseUrl = 'http://192.168.x.x:8080';  // your local IP
```

---

## Deploying

### Backend → Cloud Run

```bash
cd backend
gcloud builds submit --tag us-central1-docker.pkg.dev/pokecam-app/pokecam-repo/pokecam-api:latest
gcloud run deploy pokecam-api \
  --image us-central1-docker.pkg.dev/pokecam-app/pokecam-repo/pokecam-api:latest \
  --platform managed --region us-central1 --allow-unauthenticated \
  --memory 512Mi --cpu 1 --timeout 30 --max-instances 3
```

Verify it's live:
```bash
curl https://pokecam-api-633662296510.us-central1.run.app/health
# Expected: {"status":"ok"}
```

### Frontend → Firebase Hosting

```bash
cd mobile/pokecam_app
flutter build web --release

cd ../..   # repo root
firebase deploy --only hosting
```

Live URL: https://pokecam-app.web.app

---

## Key Files to Know

| File | What to change |
|------|----------------|
| `backend/centering.py` | Tune the CV algorithm — all edge detection logic lives here |
| `backend/main.py` | Add/modify API endpoints |
| `mobile/pokecam_app/lib/screens/camera_screen.dart` | Camera UI and result page layout |
| `mobile/pokecam_app/lib/widgets/result_card.dart` | How results are displayed |
| `mobile/pokecam_app/lib/services/api_service.dart` | Backend URL and HTTP logic |

---

## PSA Grading Standards

| Grade | Left/Right | Top/Bottom |
|-------|-----------|------------|
| PSA 10 | 45/55 or better | 45/55 or better |
| PSA 9  | 60/40 or better | 60/40 or better |

---

## GCP Project

- **Project:** `pokecam-app` (ID: `633662296510`)
- **Region:** `us-central1`
- **Artifact Registry:** `pokecam-repo`
- **Cloud Run service:** `pokecam-api`
- **Firebase Hosting:** `pokecam-app.web.app`

---

## Next Steps (pick up here next session)

### 1. Algorithm reliability — same card, different photos give different results

**Problem:** Scanning the same card twice produces different LR/TB numbers. The algorithm is sensitive to photo angle, lighting, and distance, which means measurements aren't reproducible enough to trust for grading decisions.

**Where to look:** `centering.py` — `_find_card_corners()` (outer edge RANSAC) and `_find_inner_borders()` (inner border two-pass scan). Run `test_centering.py --debug` on multiple photos of the same card to see which step is drifting.

**Likely causes:**
- Perspective warp quality varies with shooting angle — even small tilt changes the warped card shape
- Inner border scan is sensitive to lighting changes on the card surface
- The consistency score (see below) can tell you how noisy a given scan was

---

### 2. Understand and surface the consistency score better

**What it is:** After detecting inner border hits across all scanlines, the algorithm builds a histogram of hit positions. `consistency` = fraction of hits that fall within ±15px of the histogram peak. Range 0–1.

- **High (≥ 0.9):** Most scanlines agree on the border position — reliable measurement
- **Medium (0.3–0.9):** Some scatter — result is probably OK but less certain
- **Low (< 0.3):** Hits are spread out — algorithm is guessing, result is unreliable

**Current behavior:** Consistency is shown in the `_4_borders` debug image footer but the app only shows a generic "low confidence" warning if `confidence == "low"`. The confidence field is only downgraded to `"low"` if `inner_consistency < 0.25` — a very low bar.

**What to do:** Surface the raw consistency score in the app UI so users know how much to trust each scan. Also consider raising the threshold for downgrading confidence, or refusing to report a grade if consistency is below ~0.5.

---

### 3. Remove %-based search windows for outer and inner border detection

**Problem:** The outer border scan uses `search_frac=0.50` (only looks in the left/right 50% for edges) and the inner border scan uses hardcoded 3–8% and 3–20% windows relative to card size. These fractions are brittle — they assume the card fills a certain portion of the frame and that borders are within a fixed depth range.

**What breaks:** If the card is shot close-up or at an angle, the outer edge may land outside the search window. If it's a card with an unusually wide or narrow design border, the inner window misses it.

**Where to look:**
- Outer: `_scan_row_edges()` and `_scan_col_edges()` in `centering.py` — the `search_frac` parameter
- Inner: `_find_inner_borders()` — `x_lo_n / x_hi_n` (Pass 1: 3–8%) and `x_lo_w / x_hi_w` (Pass 2: 3–20%)

**Direction:** Replace fixed % windows with adaptive detection — e.g. scan the full strip and use the histogram to find the dominant peak, rather than constraining the search zone upfront.

---

## Future Work (longer term)

- Card name/set recognition (ML model or Vision API)
- eBay price lookup by card ID
- Firebase Auth + Firestore to save scan history per user
- Native iOS app (requires Xcode) for better camera control
- Fine-tuned U-Net segmentation for holographic SIR card inner border accuracy
