from threading import Lock
from typing import Optional

from pydantic import BaseModel, PrivateAttr, computed_field


class Status(BaseModel):
    _ping_ok: bool = PrivateAttr(default=False)
    _request_ok: bool = PrivateAttr(default=False)
    _first_request_ok: bool = PrivateAttr(default=False)
    _error: Optional[str] = PrivateAttr(default=None)

    _lock: Lock = PrivateAttr(default_factory=Lock)

    @computed_field
    @property
    def ping_ok(self) -> bool:
        with self._lock:
            return self._ping_ok

    @ping_ok.setter
    def ping_ok(self, value: bool):
        with self._lock:
            self._ping_ok = value

    @computed_field
    @property
    def request_ok(self) -> bool:
        with self._lock:
            return self._request_ok

    @request_ok.setter
    def request_ok(self, value: bool):
        with self._lock:
            self._request_ok = value

    @property
    def first_request_ok(self) -> bool:
        with self._lock:
            return self._first_request_ok

    @first_request_ok.setter
    def first_request_ok(self, value: bool):
        with self._lock:
            self._first_request_ok = value

    @computed_field
    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @error.setter
    def error(self, value: Optional[str]):
        with self._lock:
            self._error = value

    @computed_field
    @property
    def running(self) -> bool:
        return self.ping_ok and self.request_ok and self.error is None
