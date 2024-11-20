import time

import aiohttp

from .base import Backend
from PIL import Image
import asyncio
import json
import traceback
import math
import zipfile
import io
import os
import aiofiles
import base64

from pathlib import Path

class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = self.setup_logger('[MidJourney]')

    async def heart_beat(self, id_):
        task_url = f"{self.backend_id}/mj/task/{id_}/fetch"

        while True:
            try:
                resp = await self.http_request("GET", task_url, format=True)
                status = resp.get('status')
                content = ''

                if status == "SUCCESS":
                    content = resp['imageUrl']
                    self.img_url.append(resp['imageUrl'])
                    self.logger.img(f"任务{id_}成功完成，图片URL：{resp['imageUrl']}")
                    return content

                elif status == "FAILED":
                    content = resp.get('failReason') or '未知原因'
                    self.logger.error(f"任务处理失败，原因：{content}")

                    raise Exception(f"任务处理失败，原因：{content}")

                elif status == "NOT_START":
                    content = '任务未开始'

                elif status == "IN_PROGRESS":
                    content = '任务正在运行'
                    if resp.get('progress'):
                        content += f"，进度：{resp['progress']}"

                elif status == "SUBMITTED":
                    content = '任务已提交处理'

                elif status == "FAILURE":
                    fail_reason = resp.get('failReason') or '未知原因'
                    self.logger.error(f"任务处理失败，原因：{fail_reason}")
                    if "Banned prompt detected" in fail_reason:
                        await self.return_build_image("NSFW Prompt Detected")
                        return
                    else:
                        raise Exception(f"任务处理失败，原因：{content}")

                else:
                    content = status

                self.logger.info(f"任务{id_}状态：{content}")

                await asyncio.sleep(5)

            except Exception as e:
                self.logger.error(f"任务{id_}心跳监控出错: {str(e)}")
                raise

    async def update_progress(self):
        # 覆写函数
        pass

    async def get_shape(self):

        gcd = math.gcd(self.width, self.height)

        simplified_width = self.width // gcd
        simplified_height = self.height // gcd

        ar = f"{simplified_width}:{simplified_height}"

        return ar

    async def check_backend_usability(self):
        pass

    async def split_image(self):
        img = Image.open(io.BytesIO(self.img_btyes[0]))
        width, height = img.size

        half_width = width // 2
        half_height = height // 2

        coordinates = [(0, 0, half_width, half_height),
                       (half_width, 0, width, half_height),
                       (0, half_height, half_width, height),
                       (half_width, half_height, width, height)]

        images = [img.crop(c) for c in coordinates]

        images_bytes = [io.BytesIO() for _ in range(4)]
        base64_images = []

        for i in range(4):
            images[i].save(images_bytes[i], format='PNG')

            images_bytes[i].seek(0)
            base64_image = base64.b64encode(images_bytes[i].getvalue()).decode('utf-8')

            base64_images.append(base64_image)

        self.img_btyes += images_bytes
        self.img += base64_images

    # async def formating_to_sd_style(self):
    #
    #     await self.download_img()
    #     await self.split_image()
    #
    #     self.format_api_respond()
    #     self.result = self.build_respond

    async def posting(self):

        accept_ratio = await self.get_shape()

        ntags = f"--no {self.negative_prompt}" if self.negative_prompt else ""

        build_prompt = f"{self.prompt} --ar {accept_ratio} --seed {self.seed}" + ' ' + ntags + ' '

        payload = {
            "prompt": build_prompt
        }

        if self.config.midjourney['auth_toekn'][self.count]:
            self.headers.update({"mj-api-secret": self.config.midjourney['auth_toekn'][self.count]})

        resp = await self.http_request(
            "POST",
            f"{self.backend_url}/mj/submit/imagine",
            headers=self.headers,
            content=json.dumps(payload),
            format=True
        )

        if resp.get('code') == 24:
            await self.return_build_image(text="NSFW Prompt Detected")

        elif resp.get('code') == 1:
            task_id = resp.get('result')
            self.task_id = task_id
            self.logger.info(f"任务提交成功，任务id: {task_id}")

            await self.heart_beat(task_id)
            await self.download_img()
            await self.split_image()

        self.format_api_respond()
        self.result = self.build_respond
