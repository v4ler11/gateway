import pysbd


class SentenceCollector:
    def __init__(
            self, 
            segmenter: pysbd.Segmenter,
            min_check_interval: int = 15,
    ):
        self._segmenter = segmenter
        self._buffer = ""
        self._min_check_interval = min_check_interval
        self._token_counter = 0
        self._trigger_chars = {'.', '!', '?', ';', ':', '\n'}

    def put(self, token: str) -> list[str]:
        if not token:
            return []

        self._buffer += token
        self._token_counter += 1

        is_punctuation = any(char in token for char in self._trigger_chars)

        if is_punctuation or self._token_counter >= self._min_check_interval:
            return self._process_buffer()

        return []

    def flush(self) -> list[str]:
        if not self._buffer.strip():
            return []

        remainder = self._buffer
        self._buffer = ""
        self._token_counter = 0
        return [remainder]

    def _process_buffer(self) -> list[str]:
        self._token_counter = 0

        parts = self._segmenter.segment(self._buffer)

        if len(parts) > 1:
            complete_sentences = parts[:-1]
            # The last part is incomplete; keep it in the buffer
            self._buffer = parts[-1]

            # Filter empty strings just in case
            return [s for s in complete_sentences if s.strip()]

        return []
