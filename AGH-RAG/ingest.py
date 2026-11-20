from pypdf import PdfReader
from pathlib import Path
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import ollama
import chromadb
from typing import TypedDict

class Chunk(TypedDict):
    text: str
    source: str
    page: int

DATA_DIR = Path(__file__).resolve().parent / "data"
SKIP_PAGES = {
    "isi.pdf": {1, 2, *range(11, 28)},
    "regulamin.pdf": {1}
}
EMBED_MODEL = "bge-m3"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "agh_docs"

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text

def load_pdfs(skip_pages: dict[str, set[int]]) -> list[Chunk]:
    pages = []

    for file in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(file)
        name = file.name
        skip = skip_pages.get(name, set())

        for idx, page in enumerate(reader.pages, start=1):
            if idx in skip:
                continue
            pages.append({
                "text": clean_text(page.extract_text()),
                "source": file.name,
                "page": idx,
            })

    return pages

def chunk_pages(documents: list[Chunk], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    result = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for page in documents:
        text = page["text"]
        chunks = splitter.split_text(text)
        for chunk in chunks:
            result.append({"text": chunk, "source": page["source"], "page": page["page"]})

    return result

def embed_chunks(documents: list[Chunk]) -> np.ndarray:
    texts = [t["text"] for t in documents]
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    embeddings = np.array(response.embeddings)
    return embeddings

def build_index(chunks: list[Chunk], embeddings: np.ndarray) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(name=COLLECTION_NAME)

    collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    ids = [f"{c['source']}_p{c['page']}_c{i}" for i, c in enumerate(chunks)]

    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    return collection

if __name__ == "__main__":
    pdfs = load_pdfs(skip_pages=SKIP_PAGES)
    chunked_pages = chunk_pages(pdfs, 1000, 150)
    embed = embed_chunks(chunked_pages)
    coll = build_index(chunked_pages, embed)

    question = "Ile razy mogę poprawić egzamin"
    q_embed = ollama.embed(model=EMBED_MODEL, input=question)
    result = coll.query(query_embeddings=q_embed.embeddings, n_results=3)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        print(f"{dist:.3f} | {meta['source']} s. {meta['page']}")
        print(doc[:200])