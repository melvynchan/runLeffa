# Leffa API Documentation

## Overview

The Leffa API provides access to the "Learning Flow Fields in Attention for Controllable Person Image Generation" model, which enables virtual try-on functionality. This API allows you to upload images of people and garments to generate realistic virtual try-on results.

## Base URL

```
http://localhost:7860
```

## Endpoints

### 1. Virtual Try-On

Generate a virtual try-on image by uploading a person image and a garment image.

**Endpoint:** `POST /virtual-tryon`

**Content-Type:** `multipart/form-data` (IMPORTANT: Must use this content type, not application/json)

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| person_image | File | Yes | - | Image of the person who will wear the garment |
| garment_image | File | Yes | - | Image of the garment to be tried on |
| vt_model_type | String | No | "viton_hd" | Model type to use ("viton_hd" or "dress_code") |
| vt_garment_type | String | No | "upper_body" | Type of garment ("upper_body", "lower_body", or "dresses") |
| ref_acceleration | Boolean | No | false | Whether to use reference acceleration |
| repaint | Boolean | No | false | Whether to use repainting |
| step | Integer | No | 30 | Number of inference steps (30-100) |
| scale | Float | No | 2.5 | Scale factor (0.1-5.0) |
| seed | Integer | No | 42 | Random seed for reproducibility |
| preprocess_garment | Boolean | No | false | Whether to preprocess the garment image |

**Response:**

```json
{
  "generated_image": "base64_encoded_image_string",
  "mask_image": "base64_encoded_mask_image_string",
  "densepose_image": "base64_encoded_densepose_image_string"
}
```

- `generated_image`: Base64-encoded string of the generated try-on image
- `mask_image`: Base64-encoded string of the mask image (may be null)
- `densepose_image`: Base64-encoded string of the DensePose visualization (may be null)

**Example Usage (Python):**

```python
import requests

url = "http://localhost:7860/virtual-tryon"

# Prepare files and data
files = {
    'person_image': open('path/to/person.jpg', 'rb'),
    'garment_image': open('path/to/garment.png', 'rb')
}

data = {
    'vt_model_type': 'viton_hd',
    'vt_garment_type': 'upper_body',
    'ref_acceleration': 'false',
    'repaint': 'false',
    'step': '30',
    'scale': '2.5',
    'seed': '42',
    'preprocess_garment': 'false'
}

# Send request
response = requests.post(url, files=files, data=data)

# Process response
if response.status_code == 200:
    result = response.json()
    # The base64 encoded images can be decoded and saved or displayed
    generated_image = result['generated_image']
    mask_image = result['mask_image']
    densepose_image = result['densepose_image']
else:
    print(f"Error: {response.status_code}, {response.text}")
```

**Working with Image URLs:**

If you need to use image URLs instead of file uploads, you must download the images first and then upload them as files:

```python
import requests
from io import BytesIO
from PIL import Image

# URLs of the images
person_image_url = "https://example.com/person.jpg"
garment_image_url = "https://example.com/garment.jpg"

# Download images
person_response = requests.get(person_image_url)
person_image = BytesIO(person_response.content)

garment_response = requests.get(garment_image_url)
garment_image = BytesIO(garment_response.content)

# Prepare files and data for the API
files = {
    'person_image': ('person.jpg', person_image, 'image/jpeg'),
    'garment_image': ('garment.jpg', garment_image, 'image/jpeg')
}

data = {
    'vt_model_type': 'viton_hd',
    'vt_garment_type': 'upper_body'
}

# Send request to Leffa API
response = requests.post("http://localhost:7860/virtual-tryon", files=files, data=data)

# Process response
if response.status_code == 200:
    result = response.json()
    generated_image = result['generated_image']
else:
    print(f"Error: {response.status_code}, {response.text}")
```

**Example Usage (cURL):**

```bash
curl -X POST "http://localhost:7860/virtual-tryon" \
  -F "person_image=@path/to/person.jpg" \
  -F "garment_image=@path/to/garment.png" \
  -F "vt_model_type=viton_hd" \
  -F "vt_garment_type=upper_body" \
  -F "ref_acceleration=false" \
  -F "repaint=false" \
  -F "step=30" \
  -F "scale=2.5" \
  -F "seed=42" \
  -F "preprocess_garment=false"
```

### 2. Get Example Images

Retrieve a list of available example images that can be used for testing.

**Endpoint:** `GET /examples`

**Response:**

```json
{
  "person1": ["example1.jpg", "example2.jpg", ...],
  "person2": ["example1.jpg", "example2.jpg", ...],
  "garment": ["garment1.jpg", "garment2.jpg", ...]
}
```

### 3. Health Check

Check if the API is running correctly.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "ok"
}
```

### 4. API Information

Get general information about the API.

**Endpoint:** `GET /`

**Response:**

```json
{
  "title": "Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation",
  "description": "Leffa is a unified framework for controllable person image generation that enables precise manipulation of appearance (i.e., virtual try-on).",
  "paper": "https://arxiv.org/abs/2412.08486",
  "code": "https://github.com/franciszzj/Leffa",
  "model": "https://huggingface.co/franciszzj/Leffa",
  "note": "The models used in the API are trained solely on academic datasets. Virtual try-on uses VITON-HD/DressCode."
}
```

## Parameter Details

### Model Types

- `viton_hd`: Model trained on the VITON-HD dataset
- `dress_code`: Model trained on the DressCode dataset

### Garment Types

- `upper_body`: For tops, shirts, t-shirts, etc.
- `lower_body`: For pants, skirts, etc.
- `dresses`: For full dresses

### Other Parameters

- `ref_acceleration`: Enables reference acceleration to speed up the generation process
- `repaint`: Enables repainting for improved quality
- `step`: Number of inference steps (higher values may produce better quality but take longer)
- `scale`: Scale factor for the diffusion process
- `seed`: Random seed for reproducible results
- `preprocess_garment`: Preprocess the garment image (only works with PNG files)

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid parameters
- `500 Internal Server Error`: Server-side error

Error responses include a detail message explaining the issue:

```json
{
  "detail": "Error message"
}
```

## Notes

1. The API requires both person and garment images to be uploaded as files using multipart/form-data.
2. Images are automatically resized to 768x1024 pixels.
3. For best results, use clear images with good lighting and minimal background clutter.
4. When `preprocess_garment` is set to `true`, the garment image must be a PNG file.
5. The API processes images temporarily and deletes them after processing.
6. **IMPORTANT**: The API does not accept JSON payloads with image URLs. If you have image URLs, you must download the images first and then upload them as files.
7. Sending a JSON payload instead of multipart/form-data will result in a 422 Unprocessable Entity error with a message indicating that the person_image and garment_image fields are required.

## Common Errors

### 422 Unprocessable Entity

This error occurs when:

1. The request is sent with Content-Type: application/json instead of multipart/form-data
2. The required files (person_image and garment_image) are missing
3. You're trying to send image URLs instead of file uploads

Example error response:
```json
{
  "detail": [
    {"loc":["body","person_image"],"msg":"Field required","type":"missing"},
    {"loc":["body","garment_image"],"msg":"Field required","type":"missing"}
  ]
}
```

**Solution**: Make sure you're using multipart/form-data and uploading actual files, not sending URLs in a JSON payload.
