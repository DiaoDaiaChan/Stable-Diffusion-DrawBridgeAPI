import asyncio
import json
import traceback

import aiohttp

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)

        self.xl = self.config.liblibai_setting['xl'][self.count]
        self.flux = self.config.liblibai_setting['flux'][self.count]
        site_name = 'LiblibAI_XL' if self.xl else 'LiblibAI'
        self.model = f"{site_name} - {self.config.liblibai_setting['model_name'][self.count]}"
        self.model_id = self.config.liblibai_setting['model'][self.count]
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[LiblibAI]')

        token = self.config.liblibai[self.count]
        self.token = token
        self.backend_name = self.config.backend_name_list[4]
        self.workload_name = f"{self.backend_name}-{token}"

    async def heart_beat(self, id_):
        self.logger.info(f"{id_}开始请求")
        for i in range(60):

            response = await self.http_request(
                method="POST",
                target_url=f"https://liblib-api.vibrou.com/gateway/sd-api/generate/progress/msg/v3/{id_}",
                headers=self.headers,
                content=json.dumps({"flag": 0})
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
                await self.set_backend_working_status(available=True)
                for i in images:
                    if 'porn' in i['previewPath']:
                        raise RuntimeError("API侧检测到NSFW图片")
                    self.logger.img(f"图片url: {i['previewPath']}")
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

        if self.xl or self.flux:
            if self.xl:
                pre_tag, pre_ntag = tuple(self.config.liblibai_setting.get('preference')[self.count]['pretags']['xl'])
            elif self.flux:
                pre_tag, pre_ntag = tuple(self.config.liblibai_setting.get('preference')[self.count]['pretags']['flux'])
            self.tags = pre_tag + self.tags
            self.ntags = pre_ntag + self.ntags
            if self.enable_hr:
                self.width = int(self.width * self.hr_scale)
                self.height = int(self.height * self.hr_scale)
                self.enable_hr = False
            elif self.width * self.height < 1048576:
                self.width = int(self.width * 1.5)
                self.height = int(self.height * 1.5)

        self.steps = self.config.liblibai_setting.get('preference')[self.count].get('steps', 12)

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
                    "prompt": self.tags,
                    "negPrompt": self.ntags,
                    "seed": self.seed,
                    "randnSource": 0,
                    "samplingMethod": 31,
                    "imgCount": self.total_img_count,
                    "samplingStep": self.steps,
                    "cfgScale": self.scale,
                    "width": self.width,
                    "height": self.height
                },
                "taskQueuePriority": 1
            }

        else:
            input_ = {
                "checkpointId": self.model_id,
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

        if self.enable_hr and self.flux is False and self.xl is False:

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
        response = await self.http_request(
            method="POST",
            target_url="https://liblib-api.vibrou.com/gateway/sd-api/generate/image",
            headers=self.headers,
            content=json.dumps(input_)
        )

        # 检查请求结果
        if response.get('error') == "error":
            self.logger.warning(f"Failed to request: {response}")
        else:
            task = response
            if task['msg'] == 'Insufficient power':
                self.logger.warning('费用不足!')
            self.logger.info(f"API返回{task}")
            task_id = task['data']
            await self.heart_beat(task_id)

        await self.formating_to_sd_style()

