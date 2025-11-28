import threading
from dataclasses import dataclass


@dataclass
class Worker:
    name: str
    thread: threading.Thread
    stop_event: threading.Event
