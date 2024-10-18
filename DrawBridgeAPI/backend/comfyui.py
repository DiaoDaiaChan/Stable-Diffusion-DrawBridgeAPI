import asyncio
import copy
import json
import time
import traceback
import uuid
from pathlib import Path
from tqdm import tqdm
import os
import base64
import aiohttp

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)
        # 需要更改
        self.model = f"Comfyui - "
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[Comfyui]')
        backend = self.config.comfyui['name'][self.count]
        self.backend_name = self.config.backend_name_list[8]
        self.workload_name = f"{self.backend_name}-{backend}"

        self.current_config: dict = self.config.comfyui_setting
        self.backend_url = self.current_config['backend_url'][self.count]

        self.reflex_dict['sampler'] = {
            "DPM++ 2M": "dpmpp_2m",
            "DPM++ SDE": "dpmpp_sde",
            "DPM++ 2M SDE": "dpmpp_2m_sde",
            "DPM++ 2M SDE Heun": "dpmpp_2m_sde",
            "DPM++ 2S a": "dpmpp_2s_ancestral",
            "DPM++ 3M SDE": "dpmpp_3m_sde",
            "Euler a": "euler_ancestral",
            "Euler": "euler",
            "LMS": "lms",
            "Heun": "heun",
            "DPM2": "dpm_2",
            "DPM2 a": "dpm_2_ancestral",
            "DPM fast": "dpm_fast",
            "DPM adaptive": "dpm_adaptive",
            "Restart": "restart",
            "HeunPP2": "heunpp2",
            "IPNDM": "ipndm",
            "IPNDM_V": "ipndm_v",
            "DEIS": "deis",
            "DDIM": "ddim",
            "DDIM CFG++": "ddim",
            "PLMS": "plms",
            "UniPC": "uni_pc",
            "LCM": "lcm",
            "DDPM": "ddpm",
            # "[Forge] Flux Realistic": None,
            # "[Forge] Flux Realistic (Slow)": None,
        }
        self.reflex_dict['scheduler'] = {
            "Automatic": "normal",
            "Karras": "karras",
            "Exponential": "exponential",
            "SGM Uniform": "sgm_uniform",
            "Simple": "simple",
            "Normal": "normal",
            "DDIM": "ddim_uniform",
            "Beta": "beta"
        }

        self.scheduler = self.reflex_dict['scheduler'].get(self.scheduler, "normal")
        self.sampler = self.reflex_dict['sampler'].get(self.sampler, "euler")

        self.model_path = self.config.comfyui['model'][self.count]

        self.logger.info(f"选择工作流{self.comfyui_api_json}")
        if self.comfyui_api_json:

            with open(
                    Path(f"{os.path.dirname(os.path.abspath(__file__))}/../comfyui_workflows/{self.comfyui_api_json}.json").resolve(), 'r') as f:
                self.comfyui_api_json = json.load(f)

    async def heart_beat(self, id_):
        self.logger.info(f"{id_} 开始请求")

        async def get_images():

            response = await self.http_request(
                method="GET",
                target_url=f"{self.backend_url}/history/{id_}",
            )

            if response:
                for img in response[id_]['outputs']['9']['images']:
                    img_url = f"{self.backend_url}/view?filename={img['filename']}"
                    self.img_url.append(img_url)

        async with aiohttp.ClientSession() as session:
            ws_url = f'{self.backend_url}/ws?clientId={self.client_id}'
            async with session.ws_connect(ws_url) as ws:

                self.logger.info(f"WS连接成功: {ws_url}")
                progress_bar = None

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        ws_msg = json.loads(msg.data)

                        if ws_msg['type'] == 'progress':
                            value = ws_msg['data']['value']
                            max_value = ws_msg['data']['max']

                            if progress_bar is None:
                                progress_bar = await asyncio.to_thread(
                                    tqdm, total=max_value,
                                   desc=f"Prompt ID: {ws_msg['data']['prompt_id']}",
                                   unit="step"
                                )

                            delta = value - progress_bar.n
                            await asyncio.to_thread(progress_bar.update, delta)

                        if ws_msg['type'] == 'executing':
                            if ws_msg['data']['node'] is None:
                                self.logger.info(f"{id_}绘画完成!")
                                await get_images()
                                await ws.close()

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        self.logger.error(f"Error: {msg.data}")
                        await ws.close()
                        break

                if progress_bar is not None:
                    await asyncio.to_thread(progress_bar.close)

    async def update_progress(self):
        # 覆写函数
        pass

    async def get_backend_working_progress(self):

        self.get_backend_id()

        try:
            response = await self.http_request(
                method="GET",
                target_url=f"{self.backend_url}/queue",
            )
            if response.get("error", None):
                available = False
            else:
                available = True

            if len(response["queue_running"]) == 0:
                progress = 0
            else:
                progress = 0.99

            build_resp = self.format_progress_api_resp(progress, self.start_time)

            sc = 200 if available is True else 500
        except:
            traceback.print_exc()
        finally:
            return build_resp, sc, self.backend_url, sc

    async def check_backend_usability(self):
        pass

    async def err_formating_to_sd_style(self):

        await self.download_img()
        self.format_api_respond()
        self.result = self.build_respond

    async def posting(self):
        upload_img_resp_list = []

        if self.init_images:
            for image in self.init_images:
                resp = await self.upload_base64_image(image, uuid.uuid4().hex)
                upload_img_resp_list.append(resp)

        self.update_api_json(upload_img_resp_list)

        input_ = {
            "client_id": self.client_id,
            "prompt": self.comfyui_api_json
        }

        respone = await self.http_request(
            method="POST",
            target_url=f"{self.backend_url}/prompt",
            headers=self.headers,
            content=json.dumps(input_)
        )

        if respone.get("error", None):
            self.logger.error(respone)
            raise RuntimeError(respone["status_code"])

        self.task_id = respone['prompt_id']

        await self.heart_beat(self.task_id)
        await self.err_formating_to_sd_style()

    def update_api_json(self, init_images):
        api_json = copy.deepcopy(self.comfyui_api_json)

        self.logger.info(api_json)

        # sampler
        update_dict = api_json.get('3', None)
        if update_dict:
            update = {
                "seed": self.seed,
                "steps": self.steps,
                "cfg": self.scale,
                "sampler_name": self.sampler,
                "scheduler": self.scheduler,
                "denoise": self.denoising_strength
            }
            api_json['3']['inputs'].update(update)
        # update checkpoint
        update_dict = api_json.get('4', None)
        if update_dict:
            update = {
                "ckpt_name": self.model_path if self.model_path else None
            }
            api_json['4']['inputs'].update(update)
        # image size
        update_dict = api_json.get('5', None)
        if update_dict:
            update = {
                "width": self.width,
                "height": self.height,
                "batch_size": self.batch_size,
            }
            api_json['5']['inputs'].update(update)
        # prompt
        update_dict = api_json.get('6', None)
        if update_dict:
            update = {
                "text": self.tags
            }
            api_json['6']['inputs'].update(update)
        # negative prompt
        update_dict = api_json.get('7', None)
        if update_dict:
            update = {
                "text": self.ntags
            }
            api_json['7']['inputs'].update(update)
        # LatentUpscale / load image
        if self.init_images:
            update_dict = api_json.get('10', None)
            if update_dict:
                update = {
                    "image": init_images[0]['name']
                }
                api_json['10']['inputs'].update(update)
        else:
            update_dict = api_json.get('10', None)
            if update_dict:
                update = {
                    "width": int(self.width*self.hr_scale) if not self.hr_resize_x else self.hr_resize_x,
                    "height": int(self.height*self.hr_scale) if not self.hr_resize_y else self.hr_resize_y,
                }
                api_json['10']['inputs'].update(update)
        # image upscale
        # update_dict = api_json.get('12', None)
        # if update_dict:
        #     update = {
        #         "model_name": self.hr_upscaler if self.hr_upscaler else "RealESRGAN_x4plus.pth"
        #     }
        #     api_json['12']['inputs'].update(update)
        # resize image
        update_dict = api_json.get('14', None)
        if update_dict:
            update = {
                "ckpt_name": self.model_path if self.model_path else None
            }
            api_json['14']['inputs'].update(update)
        update_dict = api_json.get('15', None)
        if update_dict:
            update = {
                "width": int(self.width*self.hr_scale) if not self.hr_resize_x else self.hr_resize_x,
                "height": int(self.height*self.hr_scale) if not self.hr_resize_y else self.hr_resize_y,
            }
            api_json['15']['inputs'].update(update)
        # hr_steps
        update_dict = api_json.get('19', None)
        if update_dict:
            update = {
                "seed": self.seed,
                "steps": self.hr_second_pass_steps,
                "cfg": self.hr_scale,
                "sampler_name": self.sampler,
                "scheduler": self.scheduler,
                "denoise": self.denoising_strength,
            }
            api_json['19']['inputs'].update(update)

        if self.hr_prompt:
            update_dict = api_json.get('21', None)
            if update_dict:
                update = {
                    "text": self.hr_prompt
                }
                api_json['21']['inputs'].update(update)

        if self.hr_negative_prompt:
            update_dict = api_json.get('22', None)
            if update_dict:
                update = {
                    "text": self.hr_negative_prompt
                }
                api_json['22']['inputs'].update(update)

        self.logger.info(api_json)
        self.comfyui_api_json = api_json

    async def upload_base64_image(self, b64_image, name, image_type="input", overwrite=False):

        image_data = base64.b64decode(b64_image)

        data = aiohttp.FormData()
        data.add_field('image', image_data, filename=f"{name}.png", content_type='image/png')
        data.add_field('type', image_type)
        data.add_field('overwrite', str(overwrite).lower())

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.backend_url}/upload/image", data=data) as response:
                return json.loads(await response.read())
