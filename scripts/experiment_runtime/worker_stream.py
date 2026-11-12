"""Line-buffered credential redaction for isolated worker diagnostics."""

from threading import RLock


class RedactingStream:
    """Redact complete diagnostic lines, including secrets split across writes.

    ``flush`` flushes the underlying stream but retains an unfinished line.
    ``finish`` publishes the final unterminated diagnostic at process shutdown.
    """

    def __init__(self, stream, secrets):
        self.stream = stream
        self.secrets = sorted({value for value in secrets if isinstance(value, str) and value},
                              key=len, reverse=True)
        self.pending = ""
        self.lock = RLock()

    def _clean(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    def write(self, text):
        with self.lock:
            self.pending += text
            complete, separator, remainder = self.pending.rpartition("\n")
            if separator:
                self.stream.write(self._clean(complete + separator))
                self.pending = remainder
        return len(text)

    def flush(self):
        with self.lock:
            self.stream.flush()

    def finish(self):
        with self.lock:
            if self.pending:
                self.stream.write(self._clean(self.pending))
                self.pending = ""
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)
