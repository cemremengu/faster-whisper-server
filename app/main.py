import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.transcribe import format_response

logger = logging.getLogger("faster-whisper-server")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

VALID_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from faster_whisper import WhisperModel

    device = settings.resolve_device()
    compute_type = settings.compute_type
    logger.info(
        "Loading WhisperModel name=%s device=%s compute_type=%s (this may download from HF Hub on first run)",
        settings.model,
        device,
        compute_type,
    )
    app.state.model = WhisperModel(
        settings.model,
        device=device,
        compute_type=compute_type,
        cpu_threads=settings.cpu_threads,
        num_workers=settings.num_workers,
        download_root=settings.download_root,
    )
    app.state.device = device
    app.state.ready = True
    logger.info("Model ready.")
    try:
        yield
    finally:
        app.state.ready = False
        app.state.model = None


app = FastAPI(title="faster-whisper-server", lifespan=lifespan)
app.state.ready = False
app.state.model = None


@app.get("/healthz")
async def healthz(request: Request):
    if not getattr(request.app.state, "ready", False):
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": settings.model, "device": request.app.state.device}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": settings.model, "object": "model", "owned_by": "faster-whisper"}
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] | None = Form(
        default=None, alias="timestamp_granularities[]"
    ),
):
    if (
        not getattr(request.app.state, "ready", False)
        or request.app.state.model is None
    ):
        raise HTTPException(status_code=503, detail="model not ready")

    if response_format not in VALID_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"response_format must be one of {sorted(VALID_RESPONSE_FORMATS)}",
        )

    if model and model != settings.model:
        logger.warning(
            "Client requested model=%r but server is serving %r; ignoring.",
            model,
            settings.model,
        )

    granularities = set(timestamp_granularities or [])
    word_timestamps = "word" in granularities

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        finally:
            await file.close()

    try:
        segments, info = request.app.state.model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )
        return format_response(
            segments,
            info,
            response_format=response_format,
            include_words=word_timestamps,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
