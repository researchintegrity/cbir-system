import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import Union, List
import os
import sys

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

    def _build_transform(self):
        """
        Build the preprocessing transform for SSCD.
        Standard SSCD usually expects:
        - Resize to input_size (e.g. 224 or 288)
        - Normalize with ImageNet mean/std
        """
        input_size = settings.model.input_size
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def load_model(self):
        model_path = settings.model.path
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please download it first.")
            # In a real scenario, we might auto-download here
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading SSCD model from {model_path} to {self.device}...")
        try:
            # Load TorchScript model
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
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

