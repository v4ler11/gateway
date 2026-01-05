import asyncio
import soundfile as sf
from typing import AsyncGenerator, Any

from core.logger import error
from core.status.models import Task, TaskType
from models.definitions import ModelSTTAny
from stt.client import ping_stt, stream_transcriptions
from stt.globals import MOCK_FILE
from stt.models import ModelRecordParakeet


STARTUP_TIME: float = 360.


async def async_audio_generator(filename: str, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
    data, sr = sf.read(filename, dtype='float32')

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield chunk.tobytes()
        await asyncio.sleep(0)


async def task_worker(
        loop: asyncio.AbstractEventLoop,
        _a_session: Any,
        t0: float,
        task: Task,
):
    assert isinstance(task.model, ModelSTTAny)
    assert isinstance(task.model.record, ModelRecordParakeet)

    host = task.model.config.container
    model_name = task.model.record.model

    try:
        if task.task_type == TaskType.ping:
            is_alive, err = await ping_stt(host)

            if is_alive:
                task.model.status.ping_ok = True
            else:
                if loop.time() - t0 > STARTUP_TIME:
                    task.model.status.ping_ok = False
                    task.model.status.error = err
                    error(f"MODEL {model_name}: {err}")

        elif task.task_type == TaskType.request:
            byte_stream = async_audio_generator(str(MOCK_FILE))
            has_response = False
            async for _response in stream_transcriptions(host, model_name, byte_stream):
                has_response = True

            if has_response:
                task.model.status.request_ok = True
                if task.model.status.error is not None:
                    task.model.status.error = None
            else:
                if loop.time() - t0 > STARTUP_TIME:
                    err = "REQUEST FAILED: Stream yielded no events"
                    task.model.status.request_ok = False
                    task.model.status.error = err
                    error(f"MODEL {model_name}: {err}")

        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    except Exception as e:
        if task.task_type == TaskType.ping:
            task.model.status.ping_ok = False
        elif task.task_type == TaskType.request:
            task.model.status.request_ok = False

        if loop.time() - t0 > STARTUP_TIME:
            err = f"RPC ERROR: {str(e)}"
            task.model.status.error = err
            error(f"MODEL {model_name}: {err}")
