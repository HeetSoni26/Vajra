"""Production middleware for FastAPI server: CORS, Request ID, Logging, Exception Handling."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logging import setup_logger

logger = setup_logger("api.middleware")


class RequestIDAndLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for generating X-Request-ID header and logging request execution details."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        logger.info(f"[{request_id}] START {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = str(process_time_ms)
            logger.info(
                f"[{request_id}] END {request.method} {request.url.path} -> {response.status_code} ({process_time_ms}ms)"
            )
            return response
        except Exception as exc:
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"[{request_id}] ERROR {request.method} {request.url.path} -> Exception: {exc} ({process_time_ms}ms)",
                exc_info=True,
            )
            raise exc


def setup_middleware(app):
    """Attach CORS and request tracking middleware to FastAPI application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDAndLoggingMiddleware)
