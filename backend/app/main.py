import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import chat, health, models, rag

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="Mini Onyx — Open Source AI Platform (local, free, self-hosted)",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# The spec forbids credentials with a wildcard origin. Starlette works around
# the wildcard by echoing back whichever Origin the request carried, and with
# allow_credentials it sent Access-Control-Allow-Credentials: true alongside
# it. Any site could then call this API with the browser's cookies attached
# and read the replies, which is the exact thing the restriction prevents.
cors_origins = settings.cors_origins_list
allow_credentials = "*" not in cors_origins
if not allow_credentials:
    logger.warning(
        "CORS_ORIGINS is a wildcard, so credentials are disabled with it. "
        "List origins explicitly if the browser needs to send them."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
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
