from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import sys
import os
from typing import List, Dict, Any, Optional

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class MilvusManager:
    def __init__(self):
        self.host = settings.milvus.host
        self.port = settings.milvus.port
        self.collection_name = settings.milvus.collection_name
        self.dim = settings.model.embedding_dim
        self.collection = None
        self.connect()
        self.init_collection()

    def connect(self):
        print(f"Connecting to Milvus at {self.host}:{self.port}...")
        try:
            connections.connect("default", host=self.host, port=self.port)
            print("Connected to Milvus.")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise e

    def init_collection(self):
        """Initialize the collection schema if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"Collection {self.collection_name} loaded.")
        else:
            print(f"Creating collection {self.collection_name}...")
            # Define Schema
            # Primary Key: Auto-generated ID (Int64)
            # User ID: String (for isolation)
            # Image Path: String (reference to file)
            # Embedding: FloatVector
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64), # For multi-tenancy
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            
            schema = CollectionSchema(fields, "Image embeddings for CBIR")
            self.collection = Collection(self.collection_name, schema)
            
            # Create Index
            index_params = {
                "metric_type": settings.milvus.metric_type,
                "index_type": settings.milvus.index_type,
                "params": {"nlist": settings.milvus.nlist}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            print(f"Collection {self.collection_name} created and loaded.")

    def insert_vectors(self, user_id: str, image_paths: List[str], embeddings: List[List[float]]) -> List[int]:
        """
        Insert vectors into the collection.
        
        Args:
            user_id: The owner of the images.
            image_paths: List of file paths.
            embeddings: List of embedding vectors.
            
        Returns:
            List of inserted IDs.
        """
        # Prepare data for insertion
        # Milvus expects columns: [user_id_list, image_path_list, embedding_list]
        # Note: id is auto-generated
        
        user_ids = [user_id] * len(image_paths)
        data = [
            user_ids,
            image_paths,
            embeddings
        ]
        
        res = self.collection.insert(data)
        self.collection.flush() # Ensure data is visible
        return res.primary_keys

    def search_vectors(self, query_embedding: List[float], top_k: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            user_id: If provided, filter results to only this user.
            
        Returns:
            List of results (id, distance, image_path, user_id).
        """
        search_params = {
            "metric_type": settings.milvus.metric_type,
            "params": {"nprobe": 10}, # nprobe should be <= nlist
        }
        
        # Construct expression for filtering
        expr = None
        if user_id:
            expr = f"user_id == '{user_id}'"
            
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["user_id", "image_path"]
        )
        
        # Parse results
        parsed_results = []
        for hits in results:
            for hit in hits:
                parsed_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "user_id": hit.entity.get("user_id"),
                    "image_path": hit.entity.get("image_path")
                })
                
        return parsed_results

    def delete_by_path(self, user_id: str, image_path: str):
        """Delete a vector by image path and user_id."""
        expr = f"user_id == '{user_id}' && image_path == '{image_path}'"
        self.collection.delete(expr)
        self.collection.flush()
