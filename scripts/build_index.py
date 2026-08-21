import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.ingestion.loader import ingest_and_split
from src.retrieval.retriever import build_vectorstore, save_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from curriculum files")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Raw data directory (default: data/raw)",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.configure_observability()

    print(f"Loading documents from {args.source or settings.data_raw_dir}...")
    chunks = ingest_and_split(args.source)
    if not chunks:
        print("No documents found. Add PDF or TXT files to data/raw/")
        sys.exit(1)

    print(f"Split into {len(chunks)} chunks. Building FAISS index...")
    store = build_vectorstore(chunks, settings)
    path = save_vectorstore(store, settings=settings)
    print(f"Index saved to {path}")


if __name__ == "__main__":
    main()
