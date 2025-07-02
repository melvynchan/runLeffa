import os
import base64
from io import BytesIO
import tempfile
from PIL import Image
import runpod
import numpy as np
import time
import uuid

from app_api import LeffaPredictor, pil_image_to_base64

# Initialize the model
leffa_predictor = LeffaPredictor()

def save_encoded_image(encoded_image, output_path):
    """
    Save base64 encoded image to a file
    """
    with open(output_path, "wb") as fh:
        fh.write(base64.b64decode(encoded_image))
    return output_path

def base64_to_image(base64_string):
    """
    Convert base64 string to PIL Image
    """
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def handler(job):
    """
    RunPod handler function for virtual try-on
    
    Input job format:
    {
        "input": {
            "person_image": "base64_encoded_image",
            "garment_image": "base64_encoded_image",
            "model_type": "viton_hd",  # or "dress_code"
            "garment_type": "upper_body",  # or "lower_body", "dresses"
            "ref_acceleration": false,
            "repaint": false,
            "step": 30,
            "scale": 2.5,
            "seed": 42,  # set to null for random seed
            "preprocess_garment": false
        }
    }
    """
    job_input = job["input"]
    
    # Set default values for optional parameters
    model_type = job_input.get("model_type", "viton_hd")
    garment_type = job_input.get("garment_type", "upper_body")
    ref_acceleration = job_input.get("ref_acceleration", False)
    repaint = job_input.get("repaint", False)
    step = job_input.get("step", 30)
    scale = job_input.get("scale", 2.5)
    seed = job_input.get("seed", None)
    preprocess_garment = job_input.get("preprocess_garment", False)
    
    # Validate required parameters
    if "person_image" not in job_input:
        return {"error": "Missing required parameter: person_image"}
    if "garment_image" not in job_input:
        return {"error": "Missing required parameter: garment_image"}
    
    # Create temporary files for the images
    temp_dir = os.environ.get("TEMP_DIR", "/tmp")
    person_file_path = os.path.join(temp_dir, f"person_{uuid.uuid4()}.png")
    garment_file_path = os.path.join(temp_dir, f"garment_{uuid.uuid4()}.png")
    
    try:
        # Save the base64 encoded images to temporary files
        save_encoded_image(job_input["person_image"], person_file_path)
        save_encoded_image(job_input["garment_image"], garment_file_path)
        
        # Process the images
        start_time = time.time()
        gen_image, mask, densepose = leffa_predictor.leffa_predict_vt(
            person_file_path,
            garment_file_path,
            ref_acceleration,
            step,
            scale,
            seed,
            model_type,
            garment_type,
            repaint,
            preprocess_garment
        )
        processing_time = time.time() - start_time
        
        # Convert the results to base64 strings
        result = {
            "generated_image": pil_image_to_base64(gen_image),
            "mask_image": pil_image_to_base64(mask) if mask else None,
            "densepose_image": pil_image_to_base64(densepose) if densepose else None,
            "processing_time": processing_time
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Clean up temporary files
        if os.path.exists(person_file_path):
            os.remove(person_file_path)
        if os.path.exists(garment_file_path):
            os.remove(garment_file_path)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
