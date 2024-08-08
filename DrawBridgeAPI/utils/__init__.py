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


def format_vram_api_resp(self):

    build_resp = {
      "ram": {
        "free": 61582063428.50122,
        "used": 2704183296,
        "total": 64286246724.50122
      },
      "cuda": {
        "system": {
          "free": 4281335808,
          "used": 2160787456,
          "total": 85899345920
        },
        "active": {
          "current": 699560960,
          "peak": 3680867328
        },
        "allocated": {
          "current": 699560960,
          "peak": 3680867328
        },
        "reserved": {
          "current": 713031680,
          "peak": 3751804928
        },
        "inactive": {
          "current": 13470720,
          "peak": 650977280
        },
        "events": {
          "retries": 0,
          "oom": 0
        }
      }
    }
    return build_resp


def format_progress_api_resp(self, progress, start_time) -> dict:
    build_resp = {
        "progress": progress,
        "eta_relative": 0.0,
        "state": {
            "skipped": False,
            "interrupted": False,
            "job": "",
            "job_count": 0,
            "job_timestamp": start_time,
            "job_no": 0,
            "sampling_step": 0,
            "sampling_steps": 0
        },
        "current_image": "data:image/png;base64,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",
        "textinfo": None
    }

    return build_resp
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