import httpx
import json
from fastapi.exceptions import HTTPException


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


# from starlette.requests import Request
# from starlette.datastructures import Headers
# from starlette.types import Scope, Receive, Send
#
# # 构造Scope
# scope: Scope = {
#     'type': 'http',
#     'method': 'GET',
#     'path': '/example',
#     'headers': Headers({
#         'content-type': 'application/json'
#     }).raw,
#     'query_string': b'',  # Query参数，注意需要是bytes类型
# }
#
# async def receive() -> dict:
#     return {'type': 'http.request', 'body': b'', 'more_body': False}  # 模拟无Body的GET请求
#
# async def send(message: dict) -> None:
#     pass
#
# # 创建Request对象
# request_: Request = Request(scope, receive=receive, send=send)