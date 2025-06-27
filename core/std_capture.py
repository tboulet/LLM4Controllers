import sys
import io


class StreamCapture:
    """Context manager to capture stdout and stderr with a prefix.
    Optionally logs to the original CLI.

    The stdout/stderr is captured in a buffer accessible via `get_output()`.
    If `log_to_cli` is False, nothing is written to the real console.

    Usage:
        with StreamCapture(prefix="[X]", log_to_cli=True) as cap:
            print("Captured with prefix.")
        output = cap.get_output()
    """

    def __init__(self, prefix=None, log_to_cli=True):
        self.prefix = prefix
        self.log_to_cli = log_to_cli
        self._buffer = io.StringIO()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

    def _make_stream(self, original_stream: io.TextIOWrapper):
        # Proxy that writes to buffer and (optionally) to original stream
        class StreamProxy:
            def write(inner_self, data):
                self._buffer.write(data)

                if not self.log_to_cli:
                    return

                if data:
                    lines = data.splitlines(True)  # keepends=True
                    for line in lines:
                        if line.strip() and self.prefix:
                            original_stream.write(f"{self.prefix} {line}")
                        else:
                            original_stream.write(line)
                    original_stream.flush()

            def flush(inner_self):
                if self.log_to_cli:
                    original_stream.flush()

        return StreamProxy()

    def __enter__(self):
        sys.stdout = self._make_stream(self._old_stdout)
        sys.stderr = self._make_stream(self._old_stderr)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

    def get_output(self) -> str:
        return self._buffer.getvalue()
