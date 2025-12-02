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
                FieldSchema(name="labels", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=128), # Image class labels
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

    def insert_vectors(self, user_id: str, image_paths: List[str], embeddings: List[List[float]], labels: Optional[List[List[str]]] = None) -> List[int]:
        """
        Insert vectors into the collection.
        
        Args:
            user_id: The owner of the images.
            image_paths: List of file paths.
            embeddings: List of embedding vectors.
            labels: Optional list of label lists for each image (e.g., [['Western Blot'], ['Microscopy', 'Fluorescent']]).
            
        Returns:
            List of inserted IDs.
        """
        # Prepare data for insertion
        # Milvus expects columns: [user_id_list, image_path_list, labels_list, embedding_list]
        # Note: id is auto-generated
        
        user_ids = [user_id] * len(image_paths)
        # If labels not provided, use empty list for each image
        if labels is None:
            labels = [[] for _ in range(len(image_paths))]
        
        data = [
            user_ids,
            image_paths,
            labels,
            embeddings
        ]
        
        res = self.collection.insert(data)
        self.collection.flush() # Ensure data is visible
        return res.primary_keys

    def search_vectors(self, query_embedding: List[float], top_k: int = 10, user_id: Optional[str] = None, labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            user_id: If provided, filter results to only this user.
            labels: If provided, filter results to images that have ANY of these labels.
            
        Returns:
            List of results (id, distance, image_path, user_id, labels).
        """
        search_params = {
            "metric_type": settings.milvus.metric_type,
            "params": {"nprobe": 10}, # nprobe should be <= nlist
        }
        
        # Construct expression for filtering
        expr_parts = []
        if user_id:
            expr_parts.append(f"user_id == '{user_id}'")
        
        # Filter by labels using array_contains_any for OR logic
        if labels and len(labels) > 0:
            # Build expression: array_contains_any(labels, ["label1", "label2"])
            labels_str = ', '.join([f'"{label}"' for label in labels])
            expr_parts.append(f"array_contains_any(labels, [{labels_str}])")
        
        expr = ' && '.join(expr_parts) if expr_parts else None
        
        # Debug: Log the expression
        print(f"[DEBUG] Search expression: {expr}")
            
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["user_id", "image_path", "labels"]
        )
        
        # Parse results
        parsed_results = []
        for hits in results:
            for hit in hits:
                # Access entity fields using dictionary-style access
                entity = hit.entity
                entity_labels = entity.labels if hasattr(entity, 'labels') else []
                parsed_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "user_id": entity.user_id if hasattr(entity, 'user_id') else None,
                    "image_path": entity.image_path if hasattr(entity, 'image_path') else None,
                    "labels": entity_labels if entity_labels is not None else []
                })
                
        return parsed_results

    def delete_by_path(self, user_id: str, image_path: str):
        """Delete a vector by image path and user_id."""
        expr = f"user_id == '{user_id}' && image_path == '{image_path}'"
        self.collection.delete(expr)
        self.collection.flush()

    def delete_batch_by_paths(self, user_id: str, image_paths: List[str]):
        """Delete multiple vectors by image paths and user_id."""
        if not image_paths:
            return
            
        # Construct expression: user_id == 'X' && image_path in ['A', 'B', 'C']
        # Note: Milvus 'in' operator syntax
        paths_str = ", ".join([f"'{p}'" for p in image_paths])
        expr = f"user_id == '{user_id}' && image_path in [{paths_str}]"
        
        self.collection.delete(expr)
        self.collection.flush()

    def delete_by_user(self, user_id: str):
        """Delete all vectors for a specific user."""
        expr = f"user_id == '{user_id}'"
        self.collection.delete(expr)
        self.collection.flush()
        print(f"Deleted all data for user: {user_id}")
