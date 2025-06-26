import sys
import io

class StdCapture:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self._buffer = io.StringIO()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

    def _make_stream(self, original_stream):
        # Stream that writes to both the buffer and original stream with prefix
        class StreamProxy:
            def write(inner_self, data):
                # Write raw data to buffer (no prefix, raw capture)
                self._buffer.write(data)

                # Write to original stream with prefix per line
                if data:
                    lines = data.splitlines(True)  # keepends=True
                    for line in lines:
                        if line.strip():  # avoid prefixing empty lines
                            original_stream.write(self.prefix + line)
                        else:
                            original_stream.write(line)
                    original_stream.flush()

            def flush(inner_self):
                original_stream.flush()

        return StreamProxy()

    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        proxy = self._make_stream
        sys.stdout = proxy(self._old_stdout)
        sys.stderr = proxy(self._old_stderr)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

    def get_output(self) -> str:
        return self._buffer.getvalue()
