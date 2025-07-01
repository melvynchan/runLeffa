import numpy as np
import os
import base64
from io import BytesIO
from typing import Optional, List
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Download only the checkpoints needed for virtual try-on
snapshot_download(
    repo_id="franciszzj/Leffa", 
    local_dir="./ckpts",
    allow_patterns=[
        "densepose/*",
        "examples/*",
        "humanparsing/*",
        "openpose/*",
        "schp/*",
        "stable-diffusion-inpainting/*",
        "virtual_tryon.pth",
        "virtual_tryon_dc.pth"
    ],
    ignore_patterns=["pose_transfer.pth", "stable-diffusion-xl*"]
)

class LeffaPredictor(object):
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)
        
        # Pose transfer model removed as it's not needed

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=None,  
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False
    ):
        # Open and resize the source image.
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        # For virtual try-on, optionally preprocess the garment (reference) image.
        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                # preprocess_garment_image returns a 768x1024 image.
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            # Otherwise, load the reference image.
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Only virtual_tryon is supported
        src_image = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
        mask = mask.resize((768, 1024))

        # Only virtual_tryon is supported
        src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        src_image_seg = Image.fromarray(src_image_seg_array)
        densepose = src_image_seg

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        # Only virtual_tryon is supported
        vt_model_type == "viton_hd"
        inference = self.vt_inference_hd
        # Use random seed if none provided
        if seed is None:
            import random
            seed = random.randint(1, 1000000)
        
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        gen_image = output["generated_image"][0]
        # Return PIL Image objects instead of numpy arrays
        return gen_image, mask, densepose

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment):
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,  # Pass through the new flag.
        )

    # Pose transfer functionality has been removed


# Helper function to convert PIL Image to base64 string for JSON response
def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Create FastAPI app
app = FastAPI(
    title="Leffa API",
    description="API for Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor at startup
leffa_predictor = None

@app.on_event("startup")
async def startup_event():
    global leffa_predictor
    leffa_predictor = LeffaPredictor()

# Define response model for virtual try-on
class VirtualTryOnResponse(BaseModel):
    generated_image: str
    mask_image: Optional[str] = None
    densepose_image: Optional[str] = None

# Define available example images
@app.get("/examples")
async def get_examples():
    example_dir = "./ckpts/examples"
    return {
        "person1": list_dir(f"{example_dir}/person1"),
        "person2": list_dir(f"{example_dir}/person2"),
        "garment": list_dir(f"{example_dir}/garment"),
    }

# Endpoint for virtual try-on with file uploads
@app.post("/virtual-tryon", response_model=VirtualTryOnResponse)
async def virtual_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    vt_model_type: str = Form("viton_hd"),
    vt_garment_type: str = Form("upper_body"),
    ref_acceleration: bool = Form(False),
    repaint: bool = Form(False),
    step: int = Form(10),
    scale: float = Form(2.5),
    seed: int = Form(None),  # Changed to None to allow random seed generation
    preprocess_garment: bool = Form(False),
):
    # Validate parameters
    if vt_model_type not in ["viton_hd", "dress_code"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    if vt_garment_type not in ["upper_body", "lower_body", "dresses"]:
        raise HTTPException(status_code=400, detail="Invalid garment type")
    
    if step < 10 or step > 100:
        raise HTTPException(status_code=400, detail="Step must be between 30 and 100")
    
    if scale < 0.1 or scale > 5.0:
        raise HTTPException(status_code=400, detail="Scale must be between 0.1 and 5.0")
    
    # Save uploaded files temporarily
    person_file_path = f"temp_person_{os.urandom(8).hex()}.png"
    garment_file_path = f"temp_garment_{os.urandom(8).hex()}.png"
    
    try:
        # Save uploaded files
        with open(person_file_path, "wb") as f:
            f.write(await person_image.read())
        
        with open(garment_file_path, "wb") as f:
            f.write(await garment_image.read())
        
        # Process images
        gen_image, mask, densepose = leffa_predictor.leffa_predict_vt(
            person_file_path,
            garment_file_path,
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            repaint,
            preprocess_garment
        )
        
        # Convert PIL images to base64 for JSON response
        response = VirtualTryOnResponse(
            generated_image=pil_image_to_base64(gen_image),
            mask_image=pil_image_to_base64(mask) if mask else None,
            densepose_image=pil_image_to_base64(densepose) if densepose else None
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        if os.path.exists(person_file_path):
            os.remove(person_file_path)
        if os.path.exists(garment_file_path):
            os.remove(garment_file_path)

# Endpoint for health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Endpoint for API info
@app.get("/")
async def root():
    return {
        "title": "Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation",
        "description": "Leffa is a unified framework for controllable person image generation that enables precise manipulation of appearance (i.e., virtual try-on).",
        "paper": "https://arxiv.org/abs/2412.08486",
        "code": "https://github.com/franciszzj/Leffa",
        "model": "https://huggingface.co/franciszzj/Leffa",
        "note": "The models used in the API are trained solely on academic datasets. Virtual try-on uses VITON-HD/DressCode."
    }

if __name__ == "__main__":
    # Download only the checkpoints needed for virtual try-on
    snapshot_download(
        repo_id="franciszzj/Leffa", 
        local_dir="./ckpts",
        allow_patterns=[
            "densepose/*",
            "examples/*",
            "humanparsing/*",
            "openpose/*",
            "schp/*",
            "stable-diffusion-inpainting/*",
            "virtual_tryon.pth",
            "virtual_tryon_dc.pth"
        ],
        ignore_patterns=["pose_transfer.pth", "stable-diffusion-xl*"]
    )
    
    # Run the FastAPI app
    uvicorn.run("app_api:app", host="0.0.0.0", port=7860, reload=False)