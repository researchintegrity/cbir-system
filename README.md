# CBIR Microservice (Content-Based Image Retrieval)

A standalone microservice for image similarity search and deduplication, powered by **Milvus** (Vector Database) and **SSCD** (Self-Supervised Copy Detection).

##  Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- GPU (Optional, but recommended for speed)

### 2. Setup

1. **Download the Model**:
   The service requires the SSCD model weights. Download them to the `models/` directory:

   ```bash
   mkdir -p models
   wget -O models/sscd_disc_mixup.torchscript.pt https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
   ```

2. **Start the Service**:

   ```bash
   docker-compose up -d
   ```

   This starts:
   - **Milvus Standalone** (Vector DB) on port `19530`
   - **CBIR API** on port `8001`

3. **Verify**:
   Visit `http://localhost:8001/docs` to see the API documentation.

4. **Visualization**:
   Access the **Attu** interface at `http://localhost:3322` to visualize and manage the Milvus database.

---

## API Usage

The service exposes a REST API for indexing and searching images.

### Index an Image

Add an image to the database. The `image_path` must be accessible to the container (e.g., via a shared volume).

**POST** `/index`

```json
{
  "user_id": "user_123",
  "image_path": "/workspace/data/image1.jpg",
  "labels": ["Western Blot", "Microscopy"]
}
```

The `labels` field is optional and allows you to tag images with class labels for filtered retrieval.

### Search for Similar Images

Find images similar to a query image. **Results are strictly isolated by `user_id`.**

**POST** `/search`

```json
{
  "user_id": "user_123",
  "image_path": "/workspace/data/query.jpg",
  "top_k": 10,
  "labels": ["Western Blot"]
}
```

The `labels` field is optional. When provided, only images with **any** of the specified labels will be returned (OR logic). When omitted, all images are considered.

### Search by File Upload

Upload an image directly to search for similar images.

**POST** `/search/upload`

Query parameters:

- `user_id` (required): User ID for isolation
- `top_k` (optional): Number of results (default: 10)
- `labels` (optional): Filter by labels (can be repeated for multiple labels)

Example:

```bash
curl -X POST "http://localhost:8001/search/upload?user_id=user_123&top_k=5&labels=Western%20Blot&labels=Microscopy" \
  -F "file=@query_image.jpg"
```

### Delete an Image

Remove an image vector from the index.

**POST** `/delete`

```json
{
  "user_id": "user_123",
  "image_path": "/workspace/data/image1.jpg"
}
```

---

## Image Class Labels

Scientific images can have different classes such as Western Blots, Fluorescent Microscopy, X-Ray, Graphs, etc. The CBIR system supports **label-based filtering** to retrieve only images of desired classes.

### How it works

- **Indexing**: When adding an image, you can optionally provide a list of labels.
- **Searching**: When querying, you can filter results to only include images with specific labels.
- **OR Logic**: If multiple labels are specified, images matching **any** of the labels are returned.
- **No Filter**: If no labels are provided during search, all images are considered.

### Example: Indexing with Labels

```bash
python dataset/add_images_to_index.py --images-dir ./images --labels "Western Blot" "Microscopy"
```

Or using a JSON mapping file:

```bash
python dataset/add_images_to_index.py --images-dir ./images --labels-file labels.json
```

Where `labels.json` maps image paths to their labels:

```json
{
  "image1.jpg": ["Western Blot"],
  "image2.png": ["Microscopy", "Fluorescent"]
}
```

---

## Multi-Tenancy & Data Isolation

This system is designed for multi-user environments (like ELIS).

- **Isolation Strategy**: Every vector is tagged with a `user_id`.
- **Indexing**: You must provide a `user_id` when indexing.
- **Searching**: Searches are **mandatory filtered** by `user_id`. A user can only find matches within their own uploaded images. Cross-user search is disabled by design to ensure privacy.

## Configuration

Configuration is managed in `config/config.yaml`.

- **Model**: Change `device` to `"cuda"` to enable GPU acceleration.
- **Milvus**: Configure host, port, and index parameters (IVF_FLAT, HNSW, etc.).

## Development

- **Source Code**: Located in `src/`.
- **Hot Reload**: The `docker-compose.yml` mounts the `src/` directory, so changes to the code are reflected immediately (restart container to apply).
