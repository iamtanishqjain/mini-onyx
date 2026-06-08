from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import chat, rag, models, health

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="Mini Onyx — Open Source AI Platform (local, free, self-hosted)",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(rag.router)
app.include_router(models.router)


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app_title}!",
        "docs": "/docs",
        "health": "/health",
    }
