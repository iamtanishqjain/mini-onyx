from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional

from schemas.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    CollectionsResponse,
    CollectionInfo,
)
from services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["rag"])

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
):
    """
    Upload a document (PDF, DOCX, TXT, MD).
    It gets parsed, chunked, embedded via Ollama, and stored in ChromaDB.
    """
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Sanitize collection name (ChromaDB rules)
    safe_collection = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection_name)
    safe_collection = safe_collection[:63] or "default"

    try:
        col_name, chunks_added = await rag_service.ingest_document(
            filename=filename,
            file_bytes=file_bytes,
            collection_name=safe_collection,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResponse(
        collection_name=col_name,
        chunks_added=chunks_added,
        filename=filename,
        message=f"Successfully ingested '{filename}' into collection '{col_name}' ({chunks_added} chunks).",
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Search a collection for chunks relevant to a query.
    Returns top-k chunks with similarity scores.
    """
    try:
        chunks = await rag_service.query(
            query_text=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(query=request.query, results=chunks)


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all ChromaDB collections with document counts."""
    try:
        collections = rag_service.list_collections()
        return CollectionsResponse(
            collections=[CollectionInfo(**c) for c in collections]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection and all its documents."""
    success = rag_service.delete_collection(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    return {"message": f"Collection '{collection_name}' deleted successfully."}
