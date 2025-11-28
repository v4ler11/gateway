import asyncio

import aiohttp

from core.logger import error
from core.status.models import Task, TaskType


STARTUP_TIME: float = 360.


async def task_worker(
        loop: asyncio.AbstractEventLoop,
        a_session: aiohttp.ClientSession,
        t0: float,
        task: Task,
):
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
            payload = {
                "messages": [
                    {"role": "user", "content": "Echo"},
                ],
                "stream": False,
                "max_tokens": 10,
                "model": task.model.record.model,
            }
            async with a_session.post(
                    task.model.urls.generate,
                    json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    task.model.status.request_ok = True
                    if task.model.status.error is not None:
                        task.model.status.error = None
                else:
                    if loop.time() - t0 > STARTUP_TIME:
                        text = await resp.text()
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
