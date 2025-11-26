# Dataset & Tools

This directory contains the dataset and scripts to test  the Content-Based Image Retrieval (CBIR) system.

## Source

The data in this directory is sourced from the **SILA System** repository:
[https://github.com/researchintegrity/sila-system/tree/content-ranking](https://github.com/researchintegrity/sila-system/tree/content-ranking)

## Structure

- **`rank_images/`**: Directory containing the image dataset used for ranking and retrieval tasks.
- **`gt-ranks.json`**: Ground truth data for image rankings, used for evaluating the retrieval system.
- **`image_list.txt`**: A text file listing the images in the dataset.
- **`rank_list.txt`**: A text file containing ranking information.

## Scripts

### `add_images_to_index.py`

This is the main script to index images from the `rank_images/` directory into the Milvus vector database.

**Usage:**

```bash
# Run from the project root
python dataset/add_images_to_index.py
```

**Arguments:**

- `--milvus-host`: Hostname of the Milvus server (default: `localhost`).
- `--milvus-port`: Port of the Milvus server (default: `19530`).
- `--images-dir`: Path to the directory containing images (default: `dataset/rank_images`).
- `--user-id`: User ID to associate with the indexed images (default: `admin`).
- `--batch-size`: Number of images to process in one batch (default: `32`).

### `index_images.py`

An alternative or legacy script for indexing images via the HTTP API.

### `test_client.py`

A simple command-line client to test the search functionality of the CBIR system.

**Usage:**

```bash
python dataset/test_client.py /path/to/query/image.jpg
```

**Arguments:**

- `image_path`: Path to the query image (must be accessible by the server container if using path-based search, or local path if using upload).
- `--top-k`: Number of results to return (default: 5).

## Notes

- Ensure the `rank_images/` directory is populated before running the indexing scripts.
- The `gt-ranks.json` file is essential for benchmarking the performance of the CBIR model.
