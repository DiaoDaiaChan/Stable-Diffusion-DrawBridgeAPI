import asyncio
import json
import traceback

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = self.setup_logger('[SeaArt]')

    async def heart_beat(self, id_):
        self.logger.info(f"{id_} 开始请求")
        data = json.dumps({"task_ids": [id_]})
        for i in range(60):
            response = await self.http_request(
                method="POST",
                target_url="https://www.seaart.me/api/v1/task/batch-progress",
                headers=self.headers,
                content=data
            )

            if isinstance(response, dict) and 'error' in response:
                raise RuntimeError(f"请求失败，错误信息: {response.get('details')}")
            else:
                items = response.get('data', {}).get('items', [])

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
                            self.logger.img(f"图片url: {url['url']}")
                            self.img_url.append(url['url'])
                        return

        raise RuntimeError(f"任务 {id_} 在60次心跳后仍未完成")


    async def update_progress(self):
        # 覆写函数
        pass

    async def check_backend_usability(self):
        pass

    async def err_formating_to_sd_style(self):

        await self.download_img()

        self.format_api_respond()

        self.result = self.build_respond

    async def posting(self):

        input_ = {
            "action": 1,
            "art_model_no": self.model_path or "1a486c58c2aa0601b57ddc263fc350d0",
            "category": 1,
            "speed_type": 1,
            "meta":
                {
                    "prompt": self.prompt,
                    "negative_prompt": self.negative_prompt,
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
            "Token": self.backend_id
        }

        self.headers.update(new_headers)

        data = json.dumps(input_)
        response = await self.http_request(
            method="POST",
            target_url="https://www.seaart.me/api/v1/task/create",
            headers=self.headers,
            content=data
        )

        if isinstance(response, dict) and 'error' in response:
            self.logger.warning(f"{response.get('details')}")
        else:
            task = response
            task_id = task.get('data', {}).get('id')

            if task_id:
                await self.heart_beat(task_id)

        await self.err_formating_to_sd_style()

