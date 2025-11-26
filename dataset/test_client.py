import requests
import argparse
import json
import os

API_URL = "http://localhost:8001"

def search(image_path, top_k=5):
    print(f"Searching for similar images to: {image_path}")
    
    # Check if we should use the upload endpoint or the path endpoint
    # Since the current main.py only supports path, we'll use that.
    # But if the path is local and not on server, this will fail.
    # Ideally we should support upload.
    
    url = f"{API_URL}/search"
    payload = {
        "user_id": "test_user",
        "image_path": image_path,
        "top_k": top_k
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results.get('results', []))} matches:")
            print(json.dumps(results, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect to {API_URL}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBIR Test Client")
    parser.add_argument("image_path", help="Path to the query image (must be accessible by the server container)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    search(args.image_path, args.top_k)
