from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import uvicorn
import os
import sys
from PIL import Image
import io
import shutil

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from models.sscd_model import SSCDEncoder
from database.milvus_manager import MilvusManager
from schemas import IndexRequest, SearchRequest, SearchResponse, DeleteRequest, BatchIndexRequest, BatchDeleteRequest

app = FastAPI(title="CBIR Microservice", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
encoder = None
milvus_manager = None

@app.on_event("startup")
async def startup_event():
    global encoder, milvus_manager
    print("Starting CBIR Service...")
    
    # Initialize Model
    try:
        encoder = SSCDEncoder()
        print("Model initialized.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        # Don't crash, maybe model file is missing and will be mounted later?
        # But for now, let's print error.

    # Initialize Database
    try:
        milvus_manager = MilvusManager()
        print("Database initialized.")
    except Exception as e:
        print(f"Error initializing database: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": encoder is not None, "database": milvus_manager is not None}

@app.post("/index")
async def index_image(request: IndexRequest):
    """
    Index an image from a file path.
    """
    if not encoder or not milvus_manager:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {request.image_path}")

    try:
        # Load Image
        image = Image.open(request.image_path).convert("RGB")
        
        # Encode
        embedding = encoder.encode(image) # Returns (1, dim) array
        embedding_list = embedding[0].tolist()
        
        # Insert into Milvus
        ids = milvus_manager.insert_vectors(
            user_id=request.user_id,
            image_paths=[request.image_path],
            embeddings=[embedding_list],
            labels=[request.labels or []]
        )
        
        return {"status": "success", "id": ids[0]}
    except Exception as e:
        print(f"Error indexing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/batch")
async def index_images_batch(request: BatchIndexRequest):
    """
    Index multiple images in batch.
    """
    if not encoder or not milvus_manager:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    # Validate paths and load images
    valid_items = []
    images = []
    paths = []
    labels_list = []
    
    for item in request.items:
        if os.path.exists(item.image_path):
            try:
                img = Image.open(item.image_path).convert("RGB")
                images.append(img)
                paths.append(item.image_path)
                labels_list.append(item.labels or [])
                valid_items.append(item)
            except Exception as e:
                print(f"Error loading image {item.image_path}: {e}")
                # Skip failed loads
        else:
            print(f"Image not found: {item.image_path}")

    if not images:
        raise HTTPException(status_code=400, detail="No valid images found to index")

    try:
        # Encode in batch
        # encoder.encode handles batching internally
        embeddings = encoder.encode(images) # Returns (N, dim) array
        embeddings_list = embeddings.tolist()
        
        # Insert into Milvus
        ids = milvus_manager.insert_vectors(
            user_id=request.user_id,
            image_paths=paths,
            embeddings=embeddings_list,
            labels=labels_list
        )
        
        # Ensure ids is a proper Python list for JSON serialization
        ids = [int(id_val) for id_val in ids]
        
        return {
            "status": "success", 
            "indexed_count": len(ids), 
            "ids": ids,
            "failed_count": len(request.items) - len(ids)
        }
    except Exception as e:
        print(f"Error batch indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_image(request: SearchRequest):
    """
    Search for similar images using an existing image path.
    """
    if not encoder or not milvus_manager:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    if not request.image_path or not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Query image path required and must exist")

    try:
        # Load Image
        image = Image.open(request.image_path).convert("RGB")
        
        # Encode
        embedding = encoder.encode(image)
        embedding_list = embedding[0].tolist()
        
        # Search
        # Enforce isolation: only search within the same user's images
        results = milvus_manager.search_vectors(
            query_embedding=embedding_list,
            top_k=request.top_k,
            user_id=request.user_id,
            labels=request.labels
        )
        
        return SearchResponse(results=results)
    except Exception as e:
        print(f"Error searching image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/upload")
async def search_image_upload(user_id: str, file: UploadFile = File(...), top_k: int = 10, labels: Optional[List[str]] = Query(default=None)):
    """
    Search for similar images by uploading a file directly.
    """
    if not encoder or not milvus_manager:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    # Debug: Log incoming parameters
    print(f"[DEBUG] Search request - user_id: {user_id}, top_k: {top_k}, labels: {labels}")

    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Encode
        embedding = encoder.encode(image)
        embedding_list = embedding[0].tolist()
        
        # Search
        results = milvus_manager.search_vectors(
            query_embedding=embedding_list,
            top_k=top_k,
            user_id=user_id,
            labels=labels
        )
        
        return SearchResponse(results=results)
    except Exception as e:
        print(f"Error searching uploaded image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete_image(request: DeleteRequest):
    """
    Delete an image from the index.
    """
    if not milvus_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        milvus_manager.delete_by_path(user_id=request.user_id, image_path=request.image_path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete/batch")
async def delete_images_batch(request: BatchDeleteRequest):
    """
    Delete multiple images from the index.
    """
    if not milvus_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        milvus_manager.delete_batch_by_paths(user_id=request.user_id, image_paths=request.image_paths)
        return {"status": "success", "deleted_count": len(request.image_paths)}
    except Exception as e:
        print(f"Error batch deleting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)
