
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from schemas.schemas import (
    CollectionInfo,
    CollectionsResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from services.rag_service import CollectionNotFoundError, rag_service

router = APIRouter(prefix="/rag", tags=["rag"])

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
):
    """
    Upload a document (PDF, DOCX, TXT, MD).
    It gets parsed, chunked, embedded via Ollama, and stored in the vector store.
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

    # The whole upload is held in memory to be parsed, so cap it rather than
    # letting one request decide how much the process allocates.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {settings.max_upload_mb} MB limit.",
        )

    try:
        col_name, chunks_added = await rag_service.ingest_document(
            filename=filename,
            file_bytes=file_bytes,
            collection_name=collection_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e

    return IngestResponse(
        collection_name=col_name,
        chunks_added=chunks_added,
        filename=filename,
        message=(
            f"Successfully ingested '{filename}' into collection "
            f"'{col_name}' ({chunks_added} chunks)."
        ),
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
    except CollectionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e

    return QueryResponse(query=request.query, results=chunks)


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all collections with document counts."""
    try:
        collections = rag_service.list_collections()
        return CollectionsResponse(
            collections=[CollectionInfo(**c) for c in collections]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection and all its documents."""
    success = rag_service.delete_collection(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    return {"message": f"Collection '{collection_name}' deleted successfully."}
