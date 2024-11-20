import time

import aiohttp

from .base import Backend
import asyncio
import json
import traceback
import zipfile
import io
import os
import aiofiles
import base64

from pathlib import Path

class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = self.setup_logger('[NovelAI]')

        self.reflex_dict['sampler'] = {
            "DPM++ 2M": "k_dpmpp_2m",
            "DPM++ SDE": "k_dpmpp_sde",
            "DPM++ 2M SDE": "k_dpmpp_2m_sde",
            "DPM++ 2S a": "k_dpmpp_2s_ancestral",
            "Euler a": "k_euler_ancestral",
            "Euler": "k_euler",
            "DDIM": "ddim_v3"
        }

    async def update_progress(self):
        # 覆写函数
        pass

    async def get_shape(self):
        aspect_ratio = self.width / self.height

        resolutions = {
            "832x1216": (832, 1216),
            "1216x832": (1216, 832),
            "1024x1024": (1024, 1024),
        }

        closest_resolution = min(resolutions.keys(),
                                 key=lambda r: abs((resolutions[r][0] / resolutions[r][1]) - aspect_ratio))

        self.width, self.height = resolutions[closest_resolution]

        return closest_resolution

    async def check_backend_usability(self):
        pass

    async def err_formating_to_sd_style(self):

        if self.nsfw_detected:
            await self.return_build_image()

        self.format_api_respond()

        self.result = self.build_respond

    async def posting(self):

        self.sampler = self.reflex_dict['sampler'].get(self.sampler, "k_euler_ancestral")

        header = {
            "authorization": "Bearer " + self.backend_id,
            ":authority": "https://api.novelai.net",
            ":path": "/ai/generate-image",
            "content-type": "application/json",
            "referer": "https://novelai.net",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
        }

        post_api = "https://image.novelai.net/ai/generate-image"

        await self.get_shape()

        parameters = {
            "width": self.width,
            "height": self.height,
            "qualityToggle": False,
            "scale": self.scale,
            "sampler": self.sampler,
            "steps": self.steps,
            "seed": self.seed,
            "n_samples": 1,
            "ucPreset": 0,
            "negative_prompt": self.negative_prompt,
        }

        json_data = {
            "input": self.prompt,
            "model": self.config.novelai_setting['model'][self.count],
            "parameters": parameters
        }

        async def send_request():

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                while True:
                    async with session.post(
                            post_api,
                            headers=header,
                            json=json_data,
                            ssl=False,
                            proxy=self.config.server_settings['proxy']
                    ) as response:

                        if response.status == 429:
                            resp_text = await response.json()
                            if resp_text['message'] == 'Rate limited':
                                raise Exception("触发频率限制")
                            self.logger.warning(f"token繁忙中..., {resp_text}")
                            wait_time = 5
                            await asyncio.sleep(wait_time)
                        else:
                            response_data = await response.read()
                            try:
                                with zipfile.ZipFile(io.BytesIO(response_data)) as z:
                                    z.extractall(self.save_path)
                            except:
                                try:
                                    resp_text = await response.json()
                                except:
                                    if resp_text['statusCode'] == 402:
                                        self.logger.warning(f"token余额不足, {resp_text}")
                            return

        await send_request()

        await self.images_to_base64(self.save_path)
        await self.err_formating_to_sd_style()

    async def images_to_base64(self, save_path):

        for filename in os.listdir(save_path):
            if filename.endswith('.png'):
                file_path = os.path.join(save_path, filename)
                async with aiofiles.open(file_path, "rb") as image_file:
                    image_data = await image_file.read()
                    encoded_string = base64.b64encode(image_data).decode('utf-8')
                    self.img.append(encoded_string)
                    self.img_btyes.append(image_data)
