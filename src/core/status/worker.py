import asyncio
import threading

from typing import List, Callable, Awaitable

import aiohttp

from core.logger import error, info
from core.status.models import TaskType, Task
from core.abstract import Worker
from models.definitions import ModelAny, ModelLLMAny, ModelTTSAny, task_worker_llm


async def smart_sleep(stop_event: threading.Event, delay: float) -> bool:
    loop = asyncio.get_running_loop()
    is_stopped = await loop.run_in_executor(None, stop_event.wait, delay)
    return is_stopped


async def monitor_single_model(
        model: ModelAny,
        a_session: aiohttp.ClientSession,
        stop_event: threading.Event,
        task_worker: Callable[[asyncio.AbstractEventLoop, aiohttp.ClientSession, float, Task], Awaitable[None]],
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

            if not model.status.ping_ok:
                delay = 5.0
            elif not model.status.request_ok:
                delay = 5.0
            else:
                delay = 30.

            if await smart_sleep(stop_event, delay):
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
        tasks = []
        for model in models:
            if isinstance(model, ModelLLMAny):
                task_worker = task_worker_llm

            elif isinstance(model, ModelTTSAny):
                raise NotImplementedError # todo: implement me!

            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            tasks.append(asyncio.create_task(monitor_single_model(
                model, a_session, stop_event, task_worker
            )))

        await asyncio.gather(*tasks)


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
        except: # noqa: *
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
