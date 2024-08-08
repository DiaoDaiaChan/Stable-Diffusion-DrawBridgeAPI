import asyncio
import json
import traceback

import aiohttp
import piexif
import os
import replicate

from io import BytesIO
from PIL import Image

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)

        self.model = "LiblibAI - DiaoDaia_mix_4.5"
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[LiblibAI]')

        token = self.config.liblibai[self.count]
        self.token = token
        self.backend_name = self.config.backend_name_list[4]
        self.workload_name = f"{self.backend_name}-{token}"

    async def heart_beat(self, id_):
        self.logger.info(f"{id_}开始请求")
        for i in range(60):
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                        url=f"https://liblib-api.vibrou.com/gateway/sd-api/generate/progress/msg/v3/{id_}",
                        data=json.dumps({"flag": 0})) as resp:

                    if resp.status != 200:
                        raise RuntimeError
                    resp_json = await resp.json()
                    if resp_json['code'] != 0 or resp_json['data']['statusMsg'] == '执行异常':
                        raise RuntimeError('服务器返回错误')

                    images = resp_json['data']['images']

                    if images is None:
                        self.logger.info(f"第{i+1}次心跳，未返回结果")
                        await asyncio.sleep(5)
                        continue
                    else:
                        await self.set_backend_working_status(available=True)
                        for i in images:
                            self.img_url.append(i['previewPath'])
                            self.comment = i['imageInfo']
                        break

    async def update_progress(self):
        # 覆写函数
        pass

    async def get_backend_working_progress(self):
        try:

            resp = await self.set_backend_working_status(get=True)
            progress = resp['idle']
            available = resp['available']

            progress = 0.99 if progress is False else 0.0

            build_resp = {
                "progress": progress,
                "eta_relative": 0.0,
                "state": {
                "skipped": False,
                "interrupted": False,
                "job": "",
                "job_count": 0,
                "job_timestamp": self.start_time,
                "job_no": 0,
                "sampling_step": 0,
                "sampling_steps": 0
                },
                "current_image": None,
                "textinfo": None
            }

            sc = 200 if available is True else 500
        except:
            traceback.print_exc()

        return build_resp, sc, self.token, sc

    async def check_backend_usability(self):
        pass

    async def formating_to_sd_style(self):

        await self.download_img()

        self.build_api_respond()

        self.result = self.build_respond

    async def posting(self):

        input_ = {
            "checkpointId": 2332049,
            "generateType": 1,
            "frontCustomerReq": {
                # "frontId": "f46f8e35-5728-4ded-b163-832c3b85009d",
                "frontId": "cb30fc54-db0e-4760-b40c-4fc7427ef7bc",
                "windowId": "",
                "tabType": "txt2img",
                "conAndSegAndGen": "gen"
            }
        ,
            "adetailerEnable": 0,
            "text2img": {
                "prompt": self.tags,
                "negativePrompt": self.ntags,
                "extraNetwork": "",
                "samplingMethod": 0,
                "samplingStep": self.steps,
                "width": self.width,
                "height": self.height,
                "imgCount": self.total_img_count,
                "cfgScale": self.scale,
                "seed": self.seed,
                "seedExtra": 0,
                "hiResFix": 0,
                "restoreFaces": 0,
                "tiling": 0,
                "clipSkip": 2,
                "randnSource": 0,
                "tileDiffusion": None
            }
        ,
            "taskQueuePriority": 1
        }

        if self.enable_hr:

            hr_payload = {
                "hiresSteps": self.hr_second_pass_steps,
                "denoisingStrength": self.denoising_strength,
                "hiResFix": 1 if self.enable_hr else 0,
                "hiResFixInfo": {
                    "upscaler": 6,
                    "upscaleBy": self.hr_scale,
                    "resizeWidth": int(self.width * self.hr_scale),
                    "resizeHeight": int(self.height * self.hr_scale)
                }
            }

            input_['text2img'].update(hr_payload)

        new_headers = {
            "Accept": "application/json, text/plain, */*",
            "Token": self.token
        }
        self.headers.update(new_headers)

        await self.set_backend_working_status(available=False)
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(
                    url="https://liblib-api.vibrou.com/gateway/sd-api/generate/image",
                    data=json.dumps(input_)
            ) as resp:
                if resp.status not in [200, 201]:
                    pass
                else:
                    task = await resp.json()
                    task_id = task['data']
                    await self.heart_beat(task_id)

        await self.formating_to_sd_style()

