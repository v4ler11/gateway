import asyncio
import io

import aiohttp
from fastapi import UploadFile

from core import BASE_DIR
from core.logger import error
from core.routers.oai.router_transcriptions import file_to_stream, get_pcm_stream
from core.status.models import Task, TaskType
from models.definitions import ModelSTTAny
from stt.models import ModelRecordParakeet


STARTUP_TIME: float = 360.
MOCK_FILE = BASE_DIR / "assets" / "mock" / "mock_stt.flac"


async def task_worker(
        loop: asyncio.AbstractEventLoop,
        a_session: aiohttp.ClientSession,
        t0: float,
        task: Task,
):
    assert isinstance(task.model, ModelSTTAny)
    assert isinstance(task.model.record, ModelRecordParakeet) # todo: remove when >1 model

    try:
        if task.task_type == TaskType.ping:
            async with a_session.get(
                    task.model.urls.ping,
                    timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 200:
                    task.model.status.ping_ok = True
                else:
                    if loop.time() - t0 > STARTUP_TIME:
                        text = await resp.text()
                        err = f"PING FAILED: {text}"
                        task.model.status.ping_ok = False
                        task.model.status.error = err
                        error(f"MODEL {task.model.record.model}: {err}")

        elif task.task_type == TaskType.request:
            upload_file = UploadFile(
                file=io.BytesIO(MOCK_FILE.read_bytes()),
                filename="audio.flac",
            )
            pcm_data = b""

            file_stream = file_to_stream(upload_file)
            pcm_stream = get_pcm_stream(file_stream)
            async for chunk in pcm_stream:
                pcm_data += chunk

            payload = {
                "model": task.model.record.model,
                "audio": pcm_data,
            }
            async with a_session.post(
                    url=task.model.urls.generate,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    task.model.status.request_ok = True
                    if task.model.status.error is not None:
                        task.model.status.error = None

                else:
                    if loop.time() - t0 > STARTUP_TIME:
                            text = await response.text()
                            err = f"REQUEST FAILED: {text}"
                            task.model.status.request_ok = False
                            task.model.status.error = err
                            error(f"MODEL {task.model.record.model}: {err}")

        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        if task.task_type == TaskType.ping:
            task.model.status.ping_ok = False
        elif task.task_type == TaskType.request:
            task.model.status.request_ok = False
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

        if loop.time() - t0 > STARTUP_TIME:
            err = f"NETWORK ERROR: {str(e)}"
            task.model.status.error = err
            error(f"MODEL {task.model.record.model}: {err}")
