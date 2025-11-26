import sys
import argparse
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from config import settings
from database.milvus_manager import MilvusManager

def main():
    parser = argparse.ArgumentParser(description="Clean up Milvus database for a specific user")
    parser.add_argument("--user-id", type=str, required=True, help="User ID to delete data for")
    parser.add_argument("--milvus-host", type=str, default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
    
    args = parser.parse_args()

    # Override settings
    settings.milvus.host = args.milvus_host
    settings.milvus.port = args.milvus_port

    print(f"Connecting to Milvus at {settings.milvus.host}:{settings.milvus.port}")
    try:
        milvus_manager = MilvusManager()
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        print("Ensure Milvus is running and accessible.")
        return

    confirm = input(f"Are you sure you want to delete ALL data for user '{args.user_id}'? (y/N): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    try:
        milvus_manager.delete_by_user(args.user_id)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error deleting data: {e}")

if __name__ == "__main__":
    main()
