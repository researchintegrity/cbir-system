from pydantic import BaseModel
from typing import List, Optional

class IndexRequest(BaseModel):
    user_id: str
    image_path: str # Absolute path to the image file

class SearchRequest(BaseModel):
    user_id: str
    image_path: Optional[str] = None # Path to query image
    top_k: int = 10
    
class SearchResult(BaseModel):
    id: int
    distance: float
    user_id: str
    image_path: str

class SearchResponse(BaseModel):
    results: List[SearchResult]

class DeleteRequest(BaseModel):
    user_id: str
    image_path: str
