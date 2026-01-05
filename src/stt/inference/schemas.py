import time
from dataclasses import dataclass
from typing import Union

from generated import stt_service


@dataclass
class SpeechStart:
    def to_proto(self):
        return stt_service.SpeechStart(
            timestamp=time.time()
        )


@dataclass
class SpeechStop:
    def to_proto(self):
        return stt_service.SpeechStop(
            timestamp=time.time()
        )


@dataclass
class SpeechTranscription:
    text: str

    def to_proto(self):
        return stt_service.SpeechTranscription(
            text=self.text,
            timestamp=time.time(),
        )


ParakeetEvent = Union[SpeechStart, SpeechStop, SpeechTranscription]
