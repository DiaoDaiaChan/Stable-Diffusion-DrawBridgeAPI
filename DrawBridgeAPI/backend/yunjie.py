import asyncio
import json
import traceback

import aiohttp

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)
        # 需要更改
        self.model = f"YunJie - {self.config.yunjie_setting['model'][self.count]}"
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[YunJie]')
        token = self.config.yunjie[self.count]

        self.token = token
        self.backend_name = self.config.backend_name_list[7]
        self.workload_name = f"{self.backend_name}-{token}"

    async def heart_beat(self, id_):
        self.logger.info(f"{id_} 开始请求")
        for i in range(60):
            await asyncio.sleep(5)
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                        url="https://www.yunjie.art/rayvision/aigc/customer/task/progress",
                        data=json.dumps({"taskId": id_})) as resp:

                    if resp.status != 200:
                        raise RuntimeError(f"请求失败，状态码: {resp.status}")

                    resp_json = await resp.json()
                    if resp_json['code'] == "Account.Token.Expired":
                        error_text = f"""
                        后端：{self.config.yunjie_setting['note'][self.count]} token过期。
                        请前往https://www.yunjie.art/ 登录重新获取token
                        """
                        self.logger.warning("token过期")
                        raise RuntimeError(error_text)
                    items = resp_json.get('data', {}).get('data', [])
                    self.logger.info(f"第{i + 1}次心跳，未返回结果")

                    if not items:
                        continue

                    for item in items:
                        url = item.get("url")

                        if url:
                            self.img_url.append(url)
                            await self.set_backend_working_status(available=True)
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
            "genModel": "advance",
            "initImage": "",
            "modelUuid": "MGC-17d172ee37c1b000",
            "samplingMethod":
                self.sampler,
            "cfgScale": self.scale,
            "samplingSteps": self.steps,
            "plugins": [],
            "clipSkip": 2,
            "etaNoiseSeedDelta": 31337,
            "prompt": self.tags,
            "negativePrompt": self.ntags,
            "resolutionX": self.width,
            "resolutionY": self.height,
            "genCount": self.total_img_count,
            "seed": self.seed,
            "tags": []
        }

        if self.enable_hr:

            hr_payload = {
                "hires":
                    {"hrSecondPassSteps": self.hr_second_pass_steps,
                     "denoisingStrength": self.denoising_strength,
                     "hrScale": self.hr_scale,
                     "hrUpscaler": "R-ESRGAN 4x+"
                     }
            }

            input_.update(hr_payload)

        new_headers = {
            "Token": self.token
        }
        self.headers.update(new_headers)

        await self.set_backend_working_status(available=False)
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(
                    url="https://www.yunjie.art/rayvision/aigc/customer/task/imageGen",
                    data=json.dumps(input_)
            ) as resp:
                if resp.status not in [200, 201]:
                    pass
                else:
                    task = await resp.json()
                    task_id = task['data']['taskId']
                    await self.heart_beat(task_id)

        await self.formating_to_sd_style()

