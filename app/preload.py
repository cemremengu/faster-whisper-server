import logging
import sys

from app.config import settings


def main() -> int:
    logging.basicConfig(level="INFO")
    log = logging.getLogger("preload")

    from faster_whisper import WhisperModel

    device = settings.resolve_device()
    log.info(
        "Preloading model=%s device=%s compute_type=%s",
        settings.model,
        device,
        settings.compute_type,
    )
    WhisperModel(
        settings.model,
        device=device,
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        num_workers=settings.num_workers,
        download_root=settings.download_root,
    )
    log.info("Preload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
