from enum import Enum
from dataclasses import dataclass
from typing import Any


class TaskType(Enum):
    ping = "ping"
    request = "request"


@dataclass
class Task:
    task_type: TaskType
    model: Any
