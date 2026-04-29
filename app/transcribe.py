from __future__ import annotations

from typing import Iterable

from fastapi import Response
from fastapi.responses import JSONResponse, PlainTextResponse


def _format_timestamp(seconds: float, *, separator: str = ",") -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _segments_to_srt(segments: list) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{_format_timestamp(seg.start, separator='.')} --> "
            f"{_format_timestamp(seg.end, separator='.')}"
        )
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _segment_to_dict(seg, include_words: bool) -> dict:
    out = {
        "id": seg.id,
        "seek": seg.seek,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "tokens": list(seg.tokens) if seg.tokens is not None else [],
        "temperature": seg.temperature,
        "avg_logprob": seg.avg_logprob,
        "compression_ratio": seg.compression_ratio,
        "no_speech_prob": seg.no_speech_prob,
    }
    if include_words and seg.words:
        out["words"] = [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for w in seg.words
        ]
    return out


def format_response(
    segments_iter: Iterable,
    info,
    response_format: str,
    *,
    include_words: bool,
) -> Response:
    segments = list(segments_iter)
    full_text = "".join(s.text for s in segments).strip()

    if response_format == "text":
        return PlainTextResponse(full_text + "\n")
    if response_format == "srt":
        return PlainTextResponse(_segments_to_srt(segments), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(_segments_to_vtt(segments), media_type="text/vtt")
    if response_format == "verbose_json":
        payload = {
            "task": "transcribe",
            "language": info.language,
            "duration": info.duration,
            "text": full_text,
            "segments": [_segment_to_dict(s, include_words) for s in segments],
        }
        if include_words:
            payload["words"] = [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for s in segments
                if s.words
                for w in s.words
            ]
        return JSONResponse(payload)
    return JSONResponse({"text": full_text})
