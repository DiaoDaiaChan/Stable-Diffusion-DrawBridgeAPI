import asyncio
import json
import traceback

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = self.setup_logger('[TusiArt]')

    async def heart_beat(self, id_):
        self.logger.info(f"{id_}开始请求")
        self.headers['referer'] = "https://tusiart.com/models"
        del self.headers['sec-ch-ua']

        for i in range(60):
            await asyncio.sleep(5)
            self.logger.info(f"第{i + 1}次心跳")
            response = await self.http_request(
                method="GET",
                target_url='https://api.tusiart.cn/works/v1/works/tasks?size=20&cursor=0&returnAllTask=true',
                headers=self.headers
            )

            if isinstance(response, dict) and 'error' in response:
                raise RuntimeError(f"Request failed with error: {response.get('details')}")
            else:
                resp_json = response
                all_tasks = resp_json['data']['tasks']
                task_found = False
                for task in all_tasks:
                    if task['taskId'] == id_:
                        task_found = True
                        if task['status'] == 'WAITING':
                            break
                        elif task['status'] == 'FINISH':
                            matched = False
                            for img in task['items']:
                                if 'workspace.tusiassets.com' in img['url']:
                                    self.logger.img(f"图片url: {img['url']}")
                                    self.img_url.append(img['url'])
                                    matched = True

                            if matched:
                                return
                            else:
                                self.logger.info(f"第{i + 1}次心跳，FINISH状态下未找到符合条件的URL")
                                await asyncio.sleep(5)
                                break
                if not task_found:
                    self.logger.info(f"任务 {id_} 未找到")
                    await asyncio.sleep(5)
                    continue

        raise RuntimeError(f"任务 {id_} 在 {60} 次轮询后仍未完成")

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

        self.sampler = "Euler a"

        input_ = {
            "params":
                {
                    "baseModel":
                     {
                         "modelId": self.model_path or "758751795863586176",
                         "modelFileId": "708770380970509676"
                     },
                    "sdxl":
                        {"refiner": False},
                    "models": [],
                    "embeddingModels": [],
                    "sdVae": "Automatic",
                    "prompt": self.prompt,
                    "negativePrompt": self.negative_prompt,
                    "height": self.height,
                    "width": self.width,
                    "imageCount": self.total_img_count,
                    "steps": self.steps,
                    "images": [],
                    "cfgScale": self.scale,
                    "seed": str(self.seed),
                    "clipSkip": 2,
                    "etaNoiseSeedDelta": 31337,
                    "v1Clip": False,
                    "samplerName": self.sampler
                },
            "taskType": "TXT2IMG",
            "isRemix": False,
            "captchaType": "CLOUDFLARE_TURNSTILE"
        }

        if self.enable_hr:

            hr_payload = {
                "enableHr": True,
                "hrUpscaler": "R-ESRGAN 4x+ Anime6B",
                "hrSecondPassSteps": self.hr_second_pass_steps,
                "denoisingStrength": self.denoising_strength,
                "hrResizeX": int(self.width*self.hr_scale),
                "hrResizeY": int(self.height*self.hr_scale)
            }

            input_['params'].update(hr_payload)

        new_headers = {
            "Authorization": f"Bearer {self.backend_id}",
            "Token": self.backend_id,
            "referer": self.config.backends[self.backend_type]['referer'][self.count],
            "sec-ch-ua": 'Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127'
        }
        self.headers.update(new_headers)

        data = json.dumps(input_)

        response = await self.http_request(
            method="POST",
            target_url="https://api.tusiart.cn/works/v1/works/task",
            headers=self.headers,
            content=data
        )

        if isinstance(response, dict) and 'error' in response:
            pass
        else:
            task = response
            if task['code'] == '1300100':
                error_text = f"""
后端：{self.config.tusiart_setting['note'][self.count]} 遇到人机验证，需到验证。
请前往https://tusiart.com/使用一次生图来触发验证码。
后端已被标记为不可使用,如需继续使用请重启API"
"""
                self.logger.warning("遇到人机验证！")
                raise RuntimeError(error_text)
            task_id = task['data']['task']['taskId']
            await self.heart_beat(task_id)

        await self.err_formating_to_sd_style()


