import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import Union, List
import os
import sys
import requests
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from models.base import ImageEncoder

class SSCDEncoder(ImageEncoder):
    def __init__(self):
        self.device = torch.device(settings.model.device if torch.cuda.is_available() and settings.model.device == "cuda" else "cpu")
        self.model = None
        self.transform = self._build_transform()
        self.load_model()

    def _download_model(self, model_path: str, download_url: str) -> bool:
        """
        Download the model from the configured URL.
        
        Args:
            model_path: Path where model should be saved
            download_url: URL to download model from
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            model_dir = os.path.dirname(model_path)
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"Downloading SSCD model from {download_url}...")
            print(f"This may take a few minutes (model size ~200MB)...")
            
            # Download with progress tracking
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
            
            print(f"Model downloaded successfully to {model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to download model: {e}")
            # Clean up partial download
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except:
                    pass
            return False

    def load_model(self):
        model_path = settings.model.path
        download_url = settings.model.download_url
        
        # If model doesn't exist, try to download it
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            
            if download_url:
                print(f"Attempting to auto-download model from {download_url}...")
                success = self._download_model(model_path, download_url)
                if not success:
                    raise FileNotFoundError(
                        f"Failed to download model from {download_url}. "
                        f"Please download manually and place at {model_path}"
                    )
            else:
                raise FileNotFoundError(
                    f"Model file not found at {model_path} and no download URL configured. "
                    f"Please download the model manually or set MODEL_DOWNLOAD_URL environment variable."
                )
        
        print(f"Loading SSCD model from {model_path} to {self.device}...")
        try:
            # Load TorchScript model
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print("✓ Model loaded successfully.")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise e
    
    def encode(self, image: Union[Image.Image, List[Image.Image]], batch_size: int = 32) -> np.ndarray:
        if self.model is None:
            self.load_model()

        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image

        if not images:
            return np.array([])

        all_embeddings = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            
            # Preprocess
            batch_tensors = []
            for img in batch_imgs:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_tensors.append(self.transform(img))
            
            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                # Forward pass
                embeddings = self.model(batch)
                
                # SSCD embeddings are usually already normalized in the model, 
                # but it's good practice to ensure L2 normalization for Cosine Similarity
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
            
        return np.concatenate(all_embeddings, axis=0)

