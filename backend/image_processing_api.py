from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from PIL import Image, ImageEnhance, ImageFilter
import io
import uuid
import json
from typing import Optional

router = APIRouter()

@router.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    filter_type: str = Form(...),
    params: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        filter_params = json.loads(params) if params else {}

        # Apply filter
        if filter_type == "blur":
            image = image.filter(ImageFilter.BLUR)
        elif filter_type == "sharpen":
            image = image.filter(ImageFilter.SHARPEN)
        elif filter_type == "brightness":
            factor = filter_params.get('factor', 1.0)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        elif filter_type == "grayscale":
            image = image.convert('L')
        elif filter_type == "rotate":
            angle = filter_params.get('angle', 0)
            image = image.rotate(angle, expand=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown filter type: {filter_type}")

        # Save and return processed image
        temp_path = f"processed_{uuid.uuid4()}.png"
        image.save(temp_path)
        return FileResponse(temp_path, media_type="image/png", headers={"X-Processed": "true"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-filters")
async def get_available_filters():
    return {"filters": [
        "blur", "sharpen", "brightness", "grayscale", "rotate"
    ]} 