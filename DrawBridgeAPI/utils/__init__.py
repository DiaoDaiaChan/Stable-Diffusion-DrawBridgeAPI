import httpx
from fastapi.exceptions import HTTPException
import asyncio


async def http_request(
        method,
        target_url,
        headers=None,
        params=None,
        content=None,
        format=True
):
    async with httpx.AsyncClient() as client:

        response = await client.request(
            method,
            target_url,
            headers=headers,
            params=params,
            content=content
        )

        if response.status_code != 200:
            raise HTTPException(500)
        if format:
            return response.json()
        else:
            return response


async def run_later(func, delay=1):
    loop = asyncio.get_running_loop()
    loop.call_later(
        delay,
        lambda: loop.create_task(
            func
        )
    )