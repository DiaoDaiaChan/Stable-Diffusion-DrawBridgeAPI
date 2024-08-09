import asyncio
import json
import traceback

import aiohttp

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)
        # 需要更改
        self.model = f"SeaArt - {self.config.seaart_setting['model'][self.count]}"
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[SeaArt]')
        token = self.config.seaart[self.count]

        self.token = token
        self.backend_name = self.config.backend_name_list[6]
        self.workload_name = f"{self.backend_name}-{token}"

    async def heart_beat(self, id_):
        self.logger.info(f"{id_} 开始请求")
        for i in range(60):
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                        url="https://www.seaart.me/api/v1/task/batch-progress",
                        data=json.dumps({"task_ids": [id_]})) as resp:

                    if resp.status != 200:
                        raise RuntimeError(f"请求失败，状态码: {resp.status}")

                    resp_json = await resp.json()

                    items = resp_json.get('data', {}).get('items', [])

                    if not items:
                        self.logger.info(f"第{i + 1}次心跳，未返回结果")
                        await asyncio.sleep(5)
                        continue

                    for item in items:
                        urls = item.get("img_uris")

                        if urls is None:
                            self.logger.info(f"第{i + 1}次心跳，未返回结果")
                            await asyncio.sleep(5)
                            continue

                        elif isinstance(urls, list):
                            for url in urls:
                                await self.set_backend_working_status(available=True)
                                self.logger.img(f"图片url: {url['url']}")
                                self.img_url.append(url['url'])
                            return

        raise RuntimeError(f"任务 {id_} 在60次心跳后仍未完成")


    async def update_progress(self):
        # 覆写函数
        pass

    async def get_backend_working_progress(self):
        try:

            resp = await self.set_backend_working_status(get=True)
            progress = resp['idle']
            available = resp['available']

            progress = 0.99 if progress is False else 0.0

            build_resp = self.format_progress_api_resp(progress, self.start_time)

            sc = 200 if available is True else 500
        except:
            traceback.print_exc()

        return build_resp, sc, self.token, sc

    async def check_backend_usability(self):
        pass

    async def formating_to_sd_style(self):

        await self.download_img()

        self.format_api_respond()

        self.result = self.build_respond

    async def posting(self):

        input_ = {
            "action": 1,
            "art_model_no": "1a486c58c2aa0601b57ddc263fc350d0",
            "category": 1,
            "speed_type": 1,
            "meta":
                {
                    "prompt": self.tags,
                    "negative_prompt": self.ntags,
                    "restore_faces": self.restore_faces,
                    "seed": self.seed,
                    "sampler_name": self.sampler,
                    "width": self.width,
                    "height": self.height,
                    "steps": self.steps,
                    "cfg_scale": self.scale,
                    "lora_models": [],
                    "vae": "vae-ft-mse-840000-ema-pruned",
                    "clip_skip": 1,
                    "hr_second_pass_steps": 20,
                    "lcm_mode": 0,
                    "n_iter": 1,
                    "embeddings": []
                }
        }

        if self.enable_hr:

            hr_payload = {
                "hr_second_pass_steps": self.hr_second_pass_steps,
                "enable_hr": True,
                "hr_upscaler": "4x-UltraSharp",
                "hr_scale": self.hr_scale,
            }

            input_['meta'].update(hr_payload)

        new_headers = {
            "Accept": "application/json, text/plain, */*",
            "Token": self.token
        }
        self.headers.update(new_headers)

        await self.set_backend_working_status(available=False)
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(
                    url="https://www.seaart.me/api/v1/task/create",
                    data=json.dumps(input_)
            ) as resp:
                if resp.status not in [200, 201]:
                    pass
                else:
                    task = await resp.json()
                    task_id = task['data']['id']
                    await self.heart_beat(task_id)

        await self.formating_to_sd_style()

