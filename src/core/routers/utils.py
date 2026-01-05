from typing import AsyncIterator, Dict, Any

import ujson as json


def str_to_streaming(s: str) -> str:
    return "data: " + s + "\n\n"


async def parse_sse_streaming(lines_iterator) -> AsyncIterator[Dict[str, Any]]:
    async for line in lines_iterator:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        line = line.strip()
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
