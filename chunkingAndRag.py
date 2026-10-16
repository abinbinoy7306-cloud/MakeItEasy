from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from chromadb import PersistentClient
import os

db_name = "data/chroma"
collection_name = "docs"

os.makedirs(db_name, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

chroma = PersistentClient(path=db_name)

collection = chroma.get_or_create_collection(
    name=collection_name
)


def split_pages_into_chunks(pages, filename,session_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_number, chunk_text in enumerate(page_chunks):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": filename,
                        "page": page.get(
                            "page_number",
                            page.get("page", 1)
                        ),
                        "chunk": chunk_number + 1,
                        "session_id": session_id
                    }
                )
            )

    return chunks


def vector_store(chunks,session_id):
    if not chunks:
        return 0

    filename = chunks[0].metadata["source"]

    for doc in chunks:
        if doc.metadata.get("session_id") != session_id:
            raise ValueError(
                "Chunk session_id does not match the requested session."
            )

        if doc.metadata.get("source") != filename:
            raise ValueError(
                "Pass chunks from only one PDF at a time."
            )

    # Remove older chunks for this PDF before re-indexing it.
    collection.delete(
        where={
            "$and": [
                {"session_id": session_id},
                {"source": filename}
            ]
        }
    )

    texts = [doc.page_content for doc in chunks]

    embeddings = embedding_model.embed_documents(texts)

    metadatas = [doc.metadata for doc in chunks]

    ids = [
        (
            f"{session_id}_{filename}"
            f"_page{doc.metadata['page']}"
            f"_chunk{doc.metadata['chunk']}"
        )
        for doc in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    added_count = len(chunks)

    print(f"Stored {added_count} chunks from {filename}")

    return added_count


def search_chunks(question,session_id, n_results=5, candidates=10):

    if not question or not question.strip():
        return []

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    # 1. Retrieve candidate chunks from Chroma.
    question_embedding = embedding_model.embed_query(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(candidates, total_chunks),
        where={"session_id": session_id},
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return []

    # 2. Rerank candidates against the actual question.
    pairs = [(question, text) for text in documents]
    scores = reranker.predict(pairs)

    matches = []

    for i, text in enumerate(documents):
        metadata = metadatas[i]

        matches.append({
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk": metadata["chunk"],
            "distance": distances[i],
            "rerank_score": float(scores[i])
        })

    # Higher cross-encoder scores mean greater relevance.
    matches.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return matches[:n_results]

def delete_pdf_chunks(filename, session_id):
    """
    Delete indexed chunks for one filename in one session.
    Other sessions' chunks remain untouched.
    """

    if not session_id:
        raise ValueError("session_id is required.")

    if not filename:
        raise ValueError("filename is required.")

    collection.delete(
        where={
            "$and": [
                {"session_id": session_id},
                {"source": filename}
            ]
        }
    )

    print(
        f"Deleted indexed chunks for {filename} "
        f"from session {session_id}"
    )