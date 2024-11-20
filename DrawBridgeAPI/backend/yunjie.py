import asyncio
import json
import traceback

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = self.setup_logger('[YunJie]')

    async def heart_beat(self, id_):
        self.logger.info(f"{id_} 开始请求")
        for i in range(60):
            await asyncio.sleep(5)

            data = json.dumps({"taskId": id_})
            response = await self.http_request(
                method="POST",
                target_url="https://www.yunjie.art/rayvision/aigc/customer/task/progress",
                headers=self.headers,
                content=data
            )

            if isinstance(response, dict) and 'error' in response:
                raise RuntimeError(f"请求失败，错误信息: {response.get('details')}")
            else:
                resp_json = response
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
                        self.logger.img(f"图片url: {url}")
                        self.img_url.append(url)
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
            "genModel": "advance",
            "initImage": "",
            "modelUuid": self.model_path or "MGC-17d172ee37c1b000",
            "samplingMethod":
                self.sampler,
            "cfgScale": self.scale,
            "samplingSteps": self.steps,
            "plugins": [],
            "clipSkip": 2,
            "etaNoiseSeedDelta": 31337,
            "prompt": self.prompt,
            "negativePrompt": self.negative_prompt,
            "resolutionX": self.width,
            "resolutionY": self.height,
            "genCount": self.total_img_count,
            "seed": self.seed,
            "prompt": []
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
            "Token": self.backend_id
        }
        self.headers.update(new_headers)
        data = json.dumps(input_)

        response = await self.http_request(
            method="POST",
            target_url="https://www.yunjie.art/rayvision/aigc/customer/task/imageGen",
            headers=self.headers,
            content=data
        )

        if response.get("error", None):
            self.logger.error(f"请求失败，错误信息: {response.get('details')}")
        else:
            task = response
            task_id = task['data']['taskId']
            await self.heart_beat(task_id)
            await self.err_formating_to_sd_style()

