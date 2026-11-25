import os
from contextlib import contextmanager
from typing import Any, Iterator

from src.config import get_settings


class ObservabilityTracker:
    """Version-agnostic Langfuse wrapper that supports v2 (.trace) and v3+ (.start_as_current_span)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self._client = None
        self._mode: str = "disabled"

        if self.enabled:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                if hasattr(self._client, "start_as_current_span"):
                    self._mode = "otel_span"
                elif hasattr(self._client, "trace"):
                    self._mode = "v2_trace"
                else:
                    print(
                        "[Observability] Unrecognized Langfuse SDK interface; "
                        "continuing with local telemetry."
                    )
                    self._mode = "noop"
            except Exception as exc:
                print(f"[Observability] Langfuse init skipped ({exc}); continuing locally.")
                self.enabled = False
                self._mode = "disabled"

    @contextmanager
    def trace_request(
        self,
        name: str,
        user_id: str = "spectrace-user",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Yield a trace/span handle if supported by the installed Langfuse SDK."""
        if not self.enabled or self._client is None or self._mode == "noop":
            yield None
            return

        if self._mode == "otel_span":
            try:
                with self._client.start_as_current_span(name=name) as span:
                    try:
                        if hasattr(span, "update_trace"):
                            span.update_trace(user_id=user_id, metadata=metadata or {})
                    except Exception:
                        pass
                    yield span
            finally:
                try:
                    self._client.flush()
                except Exception:
                    pass
            return

        # Langfuse v2 style: client.trace(...)
        trace_obj = None
        try:
            trace_obj = self._client.trace(
                name=name,
                user_id=user_id,
                metadata=metadata or {},
            )
            yield trace_obj
        except Exception as exc:
            print(f"[Observability] Trace skipped ({exc}); continuing.")
            yield None
        finally:
            if trace_obj is not None and hasattr(self._client, "flush"):
                try:
                    self._client.flush()
                except Exception:
                    pass


def record_trace_io(
    handle: Any,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Safely attach input/output/metadata across v2 trace or v3+ span handles."""
    if handle is None:
        return
    for method_name in ("update", "update_trace"):
        method = getattr(handle, method_name, None)
        if callable(method):
            try:
                method(input=input_data, output=output_data, metadata=metadata or {})
            except TypeError:
                try:
                    method(metadata=metadata or {})
                except Exception:
                    pass
            except Exception:
                pass