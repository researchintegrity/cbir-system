from glob import glob
import sys
import argparse
import json
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


def load_labels_mapping(labels_file: str) -> dict:
    """
    Load a JSON file mapping image paths to their labels.
    
    Expected format:
    {
        "path/to/image1.jpg": ["Western Blot", "Microscopy"],
        "path/to/image2.png": ["X-Ray"]
    }
    
    Or simpler format with just a default label for all images:
    {
        "default": ["Western Blot"]
    }
    """
    if not labels_file:
        return {}
    
    with open(labels_file, 'r') as f:
        return json.load(f)


def get_labels_for_image(image_path: str, labels_mapping: dict, default_labels: list) -> list:
    """Get labels for an image from the mapping, falling back to default."""
    # Try exact path match
    if str(image_path) in labels_mapping:
        return labels_mapping[str(image_path)]
    
    # Try filename match
    filename = Path(image_path).name
    if filename in labels_mapping:
        return labels_mapping[filename]
    
    # Use default labels
    return default_labels

def main():
    parser = argparse.ArgumentParser(description="Index images into Milvus")
    parser.add_argument("--milvus-host", type=str, default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--images-dir", type=str, default=str(project_root / "dataset" / "rank_images"), help="Directory containing images")
    parser.add_argument("--user-id", type=str, default="admin", help="User ID for the images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--labels", type=str, nargs='+', default=[], help="Default labels for all images (e.g., --labels 'Western Blot' 'Microscopy')")
    parser.add_argument("--labels-file", type=str, default=None, help="JSON file mapping image paths to labels")
    
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
        settings.model.path = str(project_root / "models" / "sscd_disc_mixup.torchscript.pt")
        encoder = SSCDEncoder()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    print(f"Scanning {images_dir} for images...")
    # image_files = []
    # extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    # for ext in extensions:
    #     # Recursive search
    #     image_files.extend(list(images_dir.rglob(ext)))
    #     # Also case insensitive check if needed, but rglob is usually case sensitive on Linux
    #     # For simplicity, we assume lowercase extensions or standard ones.
    
    # # Filter out duplicates if any
    # image_files = sorted(list(set(image_files)))
    
    # print(f"Found {len(image_files)} images.")
    
    # if len(image_files) == 0:
    #     return
    image_files = sorted(glob(f"{images_dir}/**/*.*", recursive=True))
    # Load labels mapping if provided
    labels_mapping = load_labels_mapping(args.labels_file) if args.labels_file else {}
    
    default_labels = args.labels
    
    if default_labels:
        print(f"Default labels: {default_labels}")
    if labels_mapping:
        print(f"Loaded labels mapping with {len(labels_mapping)} entries.")

    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    processed_count = 0
    
    # Use tqdm if available, else manual batch loop
    iterator = range(0, len(image_files), batch_size)
    iterator = tqdm(iterator, total=total_batches, desc="Indexing batches")

    for i in iterator:
        batch_paths = image_files[i : i + batch_size]
        batch_images = []
        valid_paths = []
        batch_labels = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                valid_paths.append(str(img_path))
                # Get labels for this image
                img_labels = get_labels_for_image(img_path, labels_mapping, default_labels)
                batch_labels.append(img_labels)
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
                embeddings=embeddings_list,
                labels=batch_labels
            )
            processed_count += len(valid_paths)
            
        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")

    print(f"Indexing complete. Processed {processed_count}/{len(image_files)} images.")

if __name__ == "__main__":
    main()
