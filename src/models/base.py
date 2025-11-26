from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import Union, List

class ImageEncoder(ABC):
    """Abstract base class for image encoders."""

    @abstractmethod
    def load_model(self):
        """Load the model into memory."""
        pass

    @abstractmethod
    def encode(self, image: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Encode an image or list of images into embeddings.
        
        Args:
            image: PIL Image or list of PIL Images.
            
        Returns:
            Numpy array of embeddings. Shape: (N, embedding_dim)
        """
        pass
