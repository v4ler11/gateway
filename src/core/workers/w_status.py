import asyncio
import threading

from dataclasses import dataclass
from enum import Enum
from typing import List

import aiohttp

from core.logger import error, info
from core.workers.w_abstract import Worker
from models.s3_models.models import ModelAny


STARTUP_TIME: float = 360.


class TaskType(Enum):
    ping = "ping"
    request = "request"


@dataclass
class Task:
    task_type: TaskType
    model: ModelAny


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
                    task.model.urls.chat_completions,
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


async def smart_sleep(stop_event: threading.Event, delay: float) -> bool:
    loop = asyncio.get_running_loop()
    is_stopped = await loop.run_in_executor(None, stop_event.wait, delay)
    return is_stopped


async def monitor_single_model(
        model: ModelAny,
        a_session: aiohttp.ClientSession,
        stop_event: threading.Event,
):
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    info(f"MODEL {model.record.resolve_name}: Start status worker")

    while not stop_event.is_set():
        try:
            if not model.status.ping_ok:
                next_task = Task(TaskType.ping, model)
            elif not model.status.request_ok:
                next_task = Task(TaskType.request, model)
            else:
                next_task = Task(TaskType.ping, model)

            await task_worker(
                loop,
                a_session,
                t0,
                next_task,
            )

            if await smart_sleep(stop_event, 5.0):
                info(f"MODEL {model.record.resolve_name}: Stop signal received. Exiting loop.")
                break

        except Exception as e:
            err = str(e)
            model.status.error = err
            error(f"MODEL {model.record.model}: {err}")
            if await smart_sleep(stop_event, 10.0):
                break


async def _async_entrypoint(models: List[ModelAny], stop_event: threading.Event):
    connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as a_session:
        await asyncio.gather(*[
            asyncio.create_task(monitor_single_model(m, a_session, stop_event))
            for m in models
        ])


def worker_thread_target(models: List[ModelAny], stop_event: threading.Event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_entrypoint(models, stop_event))
    finally:
        try:
            _tasks = asyncio.all_tasks(loop)
            for t in _tasks: t.cancel()
            loop.run_until_complete(asyncio.gather(*_tasks, return_exceptions=True))
        except:
            pass
        loop.close()


def spawn_worker(models: List[ModelAny]) -> Worker:
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=worker_thread_target,
        args=(models, stop_event),
        daemon=True
    )
    worker_thread.start()
    return Worker("status_worker", worker_thread, stop_event)
