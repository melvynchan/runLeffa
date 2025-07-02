import os
import asyncio
import runpod
import logging
from handler import handler as single_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of workers to process in parallel
# Adjust based on your model's memory requirements and available GPU VRAM
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "5"))

# The input schema defines the expected format of the API requests
INPUT_SCHEMA = {
    "person_image": {
        "type": "string",
        "required": True
    },
    "garment_image": {
        "type": "string",
        "required": True
    },
    "model_type": {
        "type": "string",
        "required": False,
        "default": "viton_hd",
        "enum": ["viton_hd", "dress_code"]
    },
    "garment_type": {
        "type": "string",
        "required": False,
        "default": "upper_body",
        "enum": ["upper_body", "lower_body", "dresses"]
    },
    "ref_acceleration": {
        "type": "boolean",
        "required": False,
        "default": False
    },
    "repaint": {
        "type": "boolean",
        "required": False,
        "default": False
    },
    "step": {
        "type": "integer",
        "required": False,
        "default": 30,
        "minimum": 10,
        "maximum": 100
    },
    "scale": {
        "type": "number",
        "required": False,
        "default": 2.5,
        "minimum": 0.1,
        "maximum": 5.0
    },
    "seed": {
        "type": ["integer", "null"],
        "required": False,
        "default": None
    },
    "preprocess_garment": {
        "type": "boolean",
        "required": False,
        "default": False
    }
}

# Wrap the handler to process input
async def inference(job):
    try:
        # Call the original handler directly
        # We'll rely on the handler's own validation
        result = single_handler(job)
        return result
    except Exception as e:
        logger.error(f"Error processing job {job['id']}: {str(e)}")
        return {"error": str(e)}

async def health_check():
    """Health check endpoint for RunPod"""
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info(f"Starting RunPod serverless with {NUM_WORKERS} workers")
    
    # Start the serverless handler
    # The older RunPod API uses handler parameter instead of handlers dictionary
    runpod.serverless.start({
        "handler": inference,
        "concurrency_modifier": NUM_WORKERS
    })
