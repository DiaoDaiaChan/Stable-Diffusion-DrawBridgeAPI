import asyncio
import json
import traceback

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = self.setup_logger('[LiblibAI]')

    async def heart_beat(self, id_):
        self.logger.info(f"{id_}开始请求")
        for i in range(60):

            response = await self.http_request(
                method="POST",
                target_url=f"https://liblib-api.vibrou.com/gateway/sd-api/generate/progress/msg/v3/{id_}",
                headers=self.headers,
                content=json.dumps({"flag": 0}),
                verify=False
            )

            # 检查请求结果并处理
            if response.get('error') == "error":
                self.logger.warning(f"Failed to request: {response}")
                raise RuntimeError('服务器返回错误')
            if response['code'] != 0 or response['data']['statusMsg'] == '执行异常':
                raise RuntimeError('服务器返回错误')

            images = response['data']['images']

            if images is None:
                self.logger.info(f"第{i+1}次心跳，未返回结果")
                await asyncio.sleep(5)
                continue
            else:
                # await self.set_backend_working_status(available=True)
                for i in images:
                    if 'porn' in i['previewPath']:
                        self.nsfw_detected = True
                        self.logger.warning("API侧检测到NSFW图片")
                    else:
                        self.logger.img(f"图片url: {i['previewPath']}")
                        self.img_url.append(i['previewPath'])
                        self.comment = i['imageInfo']
                break

    async def update_progress(self):
        # 覆写函数
        pass

    async def check_backend_usability(self):
        pass

    async def err_formating_to_sd_style(self):

        if self.nsfw_detected:
            await self.return_build_image()
        else:
            await self.download_img()

        self.format_api_respond()

        self.result = self.build_respond

    async def posting(self):

        if self.flux:
            input_ = {
                "checkpointId": 2295774,
                "generateType": 17,
                "frontCustomerReq": {
                    "windowId": "",
                    "tabType": "txt2img",
                    "conAndSegAndGen": "gen"
                },
                "adetailerEnable": 0,
                "text2imgV3": {
                    "clipSkip": 2,
                    "checkPointName": 2295774,
                    "prompt": self.prompt,
                    "negPrompt": self.negative_prompt,
                    "seed": self.seed,
                    "randnSource": 0,
                    "samplingMethod": 31,
                    "imgCount": self.batch_size,
                    "samplingStep": self.steps,
                    "cfgScale": self.scale,
                    "width": self.width,
                    "height": self.height
                },
                "taskQueuePriority": 1
            }

        else:
            input_ = {
                "checkpointId": self.model_path,
                "generateType": 1,
                "frontCustomerReq": {
                    # "frontId": "f46f8e35-5728-4ded-b163-832c3b85009d",
                    "windowId": "",
                    "tabType": "txt2img",
                    "conAndSegAndGen": "gen"
                }
            ,
                "adetailerEnable": 0,
                "text2img": {
                    "prompt": self.prompt,
                    "negativePrompt": self.negative_prompt,
                    "extraNetwork": "",
                    "samplingMethod": 0,
                    "samplingStep": self.steps,
                    "width": self.width,
                    "height": self.height,
                    "imgCount": self.batch_size,
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
            "Token": self.backend_id
        }
        self.headers.update(new_headers)

        response = await self.http_request(
            method="POST",
            target_url="https://liblib-api.vibrou.com/gateway/sd-api/generate/image",
            headers=self.headers,
            content=json.dumps(input_),
            verify=False
        )

        # 检查请求结果
        if response.get('error') == "error":
            self.logger.warning(f"Failed to request: {response}")
        else:
            task = response
            if task.get('msg') == 'Insufficient power':
                self.logger.warning('费用不足!')
            self.logger.info(f"API返回{task}")
            task_id = task['data']
            await self.heart_beat(task_id)

        await self.err_formating_to_sd_style()


