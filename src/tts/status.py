import asyncio

import aiohttp

from core.logger import error
from core.status.models import Task, TaskType
from models.definitions import ModelTTSAny
from tts.client import ping_tts, stream_audio
from tts.inference.schemas import TTSAudioPost
from tts.models import ModelRecordKokoro


STARTUP_TIME: float = 360.


async def task_worker(
        loop: asyncio.AbstractEventLoop,
        _a_session: aiohttp.ClientSession,
        t0: float,
        task: Task,
):
    assert isinstance(task.model, ModelTTSAny)
    assert isinstance(task.model.record, ModelRecordKokoro) # todo: remove when >1 model

    host = task.model.config.container
    model_name = task.model.record.model

    try:
        if task.task_type == TaskType.ping:
            is_alive, err = await ping_tts(host)

            if is_alive:
                task.model.status.ping_ok = True
            else:
                if loop.time() - t0 > STARTUP_TIME:
                    task.model.status.ping_ok = False
                    task.model.status.error = err
                    error(f"MODEL {model_name}: {err}")

        elif task.task_type == TaskType.request:
            post = TTSAudioPost(
                model=model_name,
                text="Hello, world!",
                voice=task.model.record.params.voice,
            )

            has_response = False
            async for _response in stream_audio(host, post):
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
