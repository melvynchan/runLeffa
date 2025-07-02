import json
import os
import time
import argparse
from pathlib import Path

# Import the handler
from handler import handler

def run_test(input_file):
    """
    Test the handler with a sample input from a JSON file
    """
    print(f"Loading test input from {input_file}")
    
    # Load test input
    with open(input_file, 'r') as f:
        test_data = json.load(f)
    
    # Add job_id to simulate RunPod job
    job = {
        "id": "test-job-" + str(int(time.time())),
        "input": test_data["input"]
    }
    
    print(f"Processing test job {job['id']}...")
    start_time = time.time()
    
    try:
        # Call the handler
        result = handler(job)
        
        # Process time
        end_time = time.time()
        process_time = end_time - start_time
        
        # Save results to output directory
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"result_{job['id']}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Test completed in {process_time:.2f} seconds")
        print(f"Results saved to {output_file}")
        
        # If there's an error in the result, print it
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return False
        
        # If successful, save the generated image if present
        if "generated_image" in result:
            import base64
            img_data = result["generated_image"]
            if "base64," in img_data:
                img_data = img_data.split("base64,")[1]
            
            img_path = output_dir / f"generated_image_{job['id']}.png"
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(img_data))
            print(f"Generated image saved to {img_path}")
        
        return True
    
    except Exception as e:
        print(f"Error running test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RunPod handler with sample input')
    parser.add_argument('--input', type=str, default='test_input.json', 
                        help='Path to test input JSON file (default: test_input.json)')
    
    args = parser.parse_args()
    
    success = run_test(args.input)
    
    if success:
        print("✅ Test completed successfully")
    else:
        print("❌ Test failed")
