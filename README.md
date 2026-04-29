# faster-whisper-server

OpenAI-compatible HTTP transcription server backed by [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper). Models are downloaded automatically from the Hugging Face Hub on first use and persisted to a Docker volume so subsequent starts are instant.

- `POST /v1/audio/transcriptions` — drop-in replacement for OpenAI's transcription endpoint (multipart upload).
- `GET /v1/models` — lists the model the server is configured for.
- `GET /healthz` — returns 503 while the model is still loading, 200 once ready.

## Quickstart

### CPU

```bash
docker compose --profile cpu up --build
```

First startup will download the model (`large-v3` by default) into the `hf-cache` Docker volume. Re-running `up` afterwards is a no-op fetch.

### NVIDIA GPU

Requires a host with NVIDIA drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker compose --profile gpu up --build
```

### Transcribe a file

```bash
curl -F file=@sample.wav \
     -F response_format=verbose_json \
     http://localhost:9000/v1/audio/transcriptions
```

### Use with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="unused")
with open("sample.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="large-v3", file=f).text)
```

## Configuration

All settings are environment variables (also accepted with the `WHISPER_` prefix in `app/config.py`):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `distil-large-v3`, …) or any HF repo id / local path. |
| `WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` (picks `cuda` if a CUDA device is visible). |
| `WHISPER_COMPUTE_TYPE` | `default` (CPU image: `int8`, GPU image: `float16`) | ctranslate2 compute type — e.g. `float16`, `int8_float16`, `int8`. |
| `WHISPER_CPU_THREADS` | `0` | CPU threads (0 = library default). |
| `WHISPER_NUM_WORKERS` | `1` | Concurrent transcription workers. |
| `WHISPER_DOWNLOAD_ROOT` | unset | Override download dir; otherwise `HF_HOME` is used. |
| `HF_HOME` | `/data` (in image) | HF cache root. Mount this as a volume to persist downloads. |
| `HF_HUB_OFFLINE` | unset | Set to `1` to forbid network calls (useful after the cache is warmed). |
| `PORT` | `9000` | Host port published by Compose. |

Override per-run, e.g.:

```bash
WHISPER_MODEL=distil-large-v3 docker compose --profile gpu up
```

## Pre-warming the model cache

To populate the `hf-cache` volume without holding a server port — useful for CI or for baking a deployable volume:

```bash
docker compose --profile cpu run --rm server-cpu python -m app.preload
```

The `app/preload.py` entrypoint instantiates the configured `WhisperModel` and exits, leaving the weights in `/data`.

## Endpoint reference

### `POST /v1/audio/transcriptions`

Multipart form fields (matching the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio/createTranscription)):

| Field | Required | Notes |
|---|---|---|
| `file` | yes | Audio file. Decoded by PyAV (FFmpeg), so most container/codec combinations work. |
| `model` | no | Accepted for SDK compatibility; the server's loaded model is authoritative. A mismatch is logged. |
| `language` | no | ISO code, e.g. `en`. Auto-detected if omitted. |
| `prompt` | no | Initial prompt to bias decoding. |
| `response_format` | no | `json` (default), `text`, `srt`, `vtt`, or `verbose_json`. |
| `temperature` | no | Float, default `0`. |
| `timestamp_granularities[]` | no | Pass `word` to enable per-word timestamps in `verbose_json`. |

## Local development (without Docker)

```bash
uv sync                 # or: pip install -e .
uvicorn app.main:app --reload
```

Make sure your Python is 3.12 (`pyproject.toml` requires `>=3.12,<3.13`). On a CUDA host you'll also need cuBLAS for CUDA 12 and cuDNN 9 on the library path — see the [faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu) for details.

## Project layout

```
app/
  main.py        # FastAPI app, lifespan-loaded WhisperModel
  config.py      # env-driven settings
  transcribe.py  # response formatting (json/text/srt/vtt/verbose_json)
  preload.py     # `python -m app.preload` to warm the cache
docker/
  Dockerfile.cpu
  Dockerfile.gpu
docker-compose.yml
```
