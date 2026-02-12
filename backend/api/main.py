import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from backend.utils.preprocess import preprocess_image
from backend.utils.predictor import predict_with_uncertainty, device
from backend.api.consultation import router as consultation_router

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Skin Disease Prediction API",
    description="AI-assisted skin disease screening with uncertainty and explainability",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(consultation_router)


@app.get("/")
def root():
    return {"message": "Skin Disease Prediction API running", "version": "2.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted: JPEG, PNG, BMP, WebP.",
        )

    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Preprocess (now returns tensor + quality info)
    image_tensor, quality = preprocess_image(image_bytes, device)

    if image_tensor is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Please upload a valid image file.",
        )

    try:
        result = predict_with_uncertainty(image_tensor, image_quality=quality, raw_bytes=image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return JSONResponse(result)


@app.post("/predict-evolution")
async def predict_evolution(
    file_current: UploadFile = File(...),
    file_previous: UploadFile = File(...),
):
    """
    Compare two images of the same lesion over time.
    Runs full prediction on the current image, with evolution data
    derived from comparing current vs previous segmentation.
    """
    for f in [file_current, file_previous]:
        if f.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.content_type}")

    current_bytes = await file_current.read()
    previous_bytes = await file_previous.read()

    current_tensor, current_quality = preprocess_image(current_bytes, device)
    previous_tensor, _ = preprocess_image(previous_bytes, device)

    if current_tensor is None or previous_tensor is None:
        raise HTTPException(status_code=400, detail="Could not decode one or both images.")

    # Import evolution analysis
    from models.longitudinal.evolution import analyze_evolution
    from backend.utils.predictor import _run_segmentation, _tensor_to_rgb_numpy

    # Get masks and images for evolution comparison
    mask_current, _ = _run_segmentation(current_tensor)
    mask_previous, _ = _run_segmentation(previous_tensor)
    img_current = _tensor_to_rgb_numpy(current_tensor)
    img_previous = _tensor_to_rgb_numpy(previous_tensor)

    evolution_raw = analyze_evolution(img_previous, mask_previous, img_current, mask_current)

    evolution_data = {
        "alert": evolution_raw.get("Evolution Alert") == "YES",
        "area_change_pct": evolution_raw.get("Area Change (%)", 0),
        "diameter_change_mm": evolution_raw.get("Diameter Change (mm)", 0),
        "color_change": evolution_raw.get("Color Change", 0),
    }

    try:
        result = predict_with_uncertainty(
            current_tensor,
            image_quality=current_quality,
            evolution_data=evolution_data,
        )
        result["evolution"] = evolution_raw
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return JSONResponse(result)
