# Pokecam

A mobile app that uses computer vision to measure Pokemon card centering for PSA grading.

## Stack
- **Mobile**: Flutter (Dart) — cross-platform iOS + Android
- **Backend**: Python 3.11 + FastAPI + OpenCV, deployed on GCP Cloud Run
- **Firebase**: Firestore / Auth / Storage (configured, not yet used in MVP)

## Repository Layout
```
pokecam/
├── backend/          ← Python FastAPI app
│   ├── centering.py  ← Core OpenCV algorithm (edit this to tune accuracy)
│   ├── main.py       ← API endpoints
│   ├── Dockerfile
│   └── requirements.txt
├── mobile/
│   └── pokecam_app/  ← Flutter project root
│       └── lib/
│           ├── main.dart
│           ├── screens/camera_screen.dart
│           ├── services/api_service.dart
│           └── widgets/result_card.dart
└── firebase/
    └── .firebaserc
```

## Development Workflow

### Backend (Python)
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8080

# Test with a card image
curl -X POST -F "file=@test_card.jpg" http://localhost:8080/analyze

# Test the algorithm in isolation
python test_centering.py path/to/card.jpg
```

### Flutter App
```bash
cd mobile/pokecam_app
flutter pub get
flutter run            # runs on connected device or simulator
flutter run --release  # production build for performance testing
```

### Deploy Backend to Cloud Run
```bash
cd backend
gcloud builds submit --tag us-central1-docker.pkg.dev/pokecam-app/pokecam-repo/pokecam-api:latest
gcloud run deploy pokecam-api \
  --image us-central1-docker.pkg.dev/pokecam-app/pokecam-repo/pokecam-api:latest \
  --platform managed --region us-central1 --allow-unauthenticated \
  --memory 512Mi --cpu 1 --timeout 30 --max-instances 3
```

## Key Conventions

### Backend
- All CV logic lives in `centering.py` — `main.py` is only routing/validation
- Return `confidence: "high" | "low"` in every response so the app can warn users
- Image size limit: 10MB. Accept JPEG and PNG only.
- Use `opencv-python-headless` (not `opencv-python`) — no GUI on Cloud Run
- Never log raw image data; log only metadata (size, shape, processing time)

### Flutter
- `api_service.dart` owns all HTTP — screens never call `http` directly
- `baseUrl` is a `static const` in `ApiService` — swap for local IP during dev, Cloud Run URL for prod
- Use `ResolutionPreset.high` for camera (not `max`) — good quality, reasonable upload size
- Always test camera features on a **physical device**, not the simulator

### Algorithm Tuning
- Card edge detection: Canny with `(50, 150)` thresholds after `bilateralFilter`
- Inner border detection: Otsu threshold after cropping 3% margins from deskewed card
- PSA 10: `abs(side_pct - 50) <= 5` (45/55 standard)
- PSA 9: `abs(side_pct - 50) <= 10` (60/40 standard)

## Environment Variables
- `PORT` — set automatically by Cloud Run (default 8080)
- No other env vars needed for MVP

## PSA Grading Reference
- **PSA 10**: Left/Right must be within 45/55; Top/Bottom within 45/55
- **PSA 9**: Left/Right within 60/40; Top/Bottom within 60/40
- **Front vs. Back**: The app measures both — user specifies which side they're photographing
  (future feature; MVP measures any side submitted)

## Future Expansion (not yet implemented)
- Card recognition via ML model (TFLite on-device or Cloud Run inference endpoint)
- eBay Marketplace API for price lookup by card ID
- Firebase Auth + Firestore to save grading history per user
