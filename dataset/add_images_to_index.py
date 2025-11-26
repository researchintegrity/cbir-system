import sys
import argparse
from pathlib import Path
from PIL import Image

# Try to import tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None):
        print(f"Processing {desc}...")
        return iterable

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from config import settings
from models.sscd_model import SSCDEncoder
from database.milvus_manager import MilvusManager

def main():
    parser = argparse.ArgumentParser(description="Index images into Milvus")
    parser.add_argument("--milvus-host", type=str, default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--images-dir", type=str, default=str(project_root / "dataset" / "rank_images"), help="Directory containing images")
    parser.add_argument("--user-id", type=str, default="admin", help="User ID for the images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()

    # Override settings
    settings.milvus.host = args.milvus_host
    settings.milvus.port = args.milvus_port

    print(f"Connecting to Milvus at {settings.milvus.host}:{settings.milvus.port}")
    try:
        milvus_manager = MilvusManager()
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        print("Ensure Milvus is running and accessible.")
        return

    print("Loading model...")
    try:
        encoder = SSCDEncoder()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    print(f"Scanning {images_dir} for images...")
    image_files = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    for ext in extensions:
        # Recursive search
        image_files.extend(list(images_dir.rglob(ext)))
        # Also case insensitive check if needed, but rglob is usually case sensitive on Linux
        # For simplicity, we assume lowercase extensions or standard ones.
    
    # Filter out duplicates if any
    image_files = sorted(list(set(image_files)))
    
    print(f"Found {len(image_files)} images.")
    
    if len(image_files) == 0:
        return

    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    processed_count = 0
    
    # Use tqdm if available, else manual batch loop
    iterator = range(0, len(image_files), batch_size)
    if 'tqdm' in sys.modules:
        iterator = tqdm(iterator, total=total_batches, desc="Indexing batches")

    for i in iterator:
        batch_paths = image_files[i : i + batch_size]
        batch_images = []
        valid_paths = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                valid_paths.append(str(img_path))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        
        if not batch_images:
            continue
            
        try:
            # Encode batch
            embeddings = encoder.encode(batch_images) # Returns (N, dim) array
            embeddings_list = embeddings.tolist()
            
            # Insert into Milvus
            milvus_manager.insert_vectors(
                user_id=args.user_id,
                image_paths=valid_paths,
                embeddings=embeddings_list
            )
            processed_count += len(valid_paths)
            
        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")

    print(f"Indexing complete. Processed {processed_count}/{len(image_files)} images.")

if __name__ == "__main__":
    main()
