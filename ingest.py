"""Build the local Chroma index from the financial PDF corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "financial-rag-agent" / "data" / "raw"
DEFAULT_VECTORSTORE_DIR = ROOT_DIR / "vectorstore" / "chroma"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "nomic-embed-text"


def build_index(data_dir: Path, vectorstore_dir: Path, reset: bool = False) -> int:
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    if reset and vectorstore_dir.exists():
        import shutil

        shutil.rmtree(vectorstore_dir)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        for page in PyPDFLoader(str(pdf_file)).load():
            page.metadata["source"] = pdf_file.name
            documents.append(page)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(vectorstore_dir),
    )

    batch_size = 100
    for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        vectorstore.add_documents(chunks[start : start + batch_size])

    print(f"Indexed {len(chunks)} chunks from {len(pdf_files)} PDFs")
    print(f"Saved Chroma database to {vectorstore_dir}")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--vectorstore-dir", type=Path, default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the existing index")
    args = parser.parse_args()
    build_index(args.data_dir.resolve(), args.vectorstore_dir.resolve(), args.reset)


if __name__ == "__main__":
    main()