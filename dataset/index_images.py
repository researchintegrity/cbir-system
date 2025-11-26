import os
import requests
import glob

API_URL = "http://localhost:8001/index"
RANK_IMAGES_DIR = "rank_images"
USER_ID = "test_user_1"

def index_images():
    if not os.path.exists(RANK_IMAGES_DIR):
        print(f"Directory {RANK_IMAGES_DIR} not found.")
        return

    print(f"Scanning {RANK_IMAGES_DIR} for images...")
    count = 0
    success = 0
    failed = 0

    # Walk through the directory
    for root, dirs, files in os.walk(RANK_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                count += 1
                # Get relative path from rank_images
                # e.g. rank_images/subdir/image.jpg -> subdir/image.jpg
                rel_path = os.path.relpath(os.path.join(root, file), RANK_IMAGES_DIR)
                
                # Construct container path
                # Mounted at /app/rank_images
                # Note: We need to ensure forward slashes for Linux container
                container_path = f"/app/rank_images/{rel_path}".replace("\\", "/")
                
                payload = {
                    "image_path": container_path,
                    "user_id": USER_ID
                }
                
                try:
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        print(f"[{count}] Indexed: {file}")
                        success += 1
                    else:
                        print(f"[{count}] Failed: {file} - {response.text}")
                        failed += 1
                except Exception as e:
                    print(f"[{count}] Error: {file} - {e}")
                    failed += 1

    print(f"\nIndexing complete.")
    print(f"Total found: {count}")
    print(f"Successfully indexed: {success}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    index_images()
