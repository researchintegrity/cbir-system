from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
from schemas import IndexRequest, SearchRequest, SearchResponse, DeleteRequest

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
            embeddings=[embedding_list]
        )
        
        return {"status": "success", "id": ids[0]}
    except Exception as e:
        print(f"Error indexing image: {e}")
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
            user_id=request.user_id
        )
        
        return SearchResponse(results=results)
    except Exception as e:
        print(f"Error searching image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/upload")
async def search_image_upload(user_id: str, file: UploadFile = File(...), top_k: int = 10):
    """
    Search for similar images by uploading a file directly.
    """
    if not encoder or not milvus_manager:
        raise HTTPException(status_code=503, detail="Service not fully initialized")

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
            user_id=user_id
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

if __name__ == "__main__":
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)
