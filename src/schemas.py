from pydantic import BaseModel
from typing import List, Optional

class IndexRequest(BaseModel):
    user_id: str
    image_path: str # Absolute path to the image file
    labels: Optional[List[str]] = None # Image class labels (e.g., ['Western Blot', 'Microscopy'])

class IndexItem(BaseModel):
    image_path: str
    labels: Optional[List[str]] = None

class BatchIndexRequest(BaseModel):
    user_id: str
    items: List[IndexItem]

class SearchRequest(BaseModel):
    user_id: str
    image_path: Optional[str] = None # Path to query image
    top_k: int = 10
    labels: Optional[List[str]] = None # Filter results to images with ANY of these labels
    
class SearchResult(BaseModel):
    id: int
    distance: float
    user_id: str
    image_path: str
    labels: List[str] = []

class SearchResponse(BaseModel):
    results: List[SearchResult]

class DeleteRequest(BaseModel):
    user_id: str
    image_path: str

class BatchDeleteRequest(BaseModel):
    user_id: str
    image_paths: List[str]
