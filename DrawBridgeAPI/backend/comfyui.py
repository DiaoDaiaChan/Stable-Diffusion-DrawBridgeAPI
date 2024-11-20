import asyncio
import copy
import json
import random
import time
import traceback
import uuid
from pathlib import Path
from tqdm import tqdm
import os
import base64
import aiohttp

from .base import Backend
from ..utils import run_later

global __ALL_SUPPORT_NODE__
MAX_SEED = 2 ** 32


class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 需要更改
        self.logger = self.setup_logger('[Comfyui]')

        self.comfyui_api_json = "sdbase_txt2img"
        self.comfyui_api_json_reflex = None

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

        self.reflex_dict['parameters'] = {}

        self.scheduler = self.reflex_dict['scheduler'].get(self.scheduler, "normal")
        self.sampler = self.reflex_dict['sampler'].get(self.sampler, "euler")

        self.logger.info(f"选择工作流{self.comfyui_api_json}")
        path_to_json = self.comfyui_api_json
        if self.comfyui_api_json:

            with open(
                    Path(f"{os.path.dirname(os.path.abspath(__file__))}/../comfyui_workflows/{self.comfyui_api_json}.json").resolve(), 'r', encoding='utf-8') as f:
                self.comfyui_api_json = json.load(f)
            with open(
                    Path(f"{os.path.dirname(os.path.abspath(__file__))}/../comfyui_workflows/{path_to_json}_reflex.json").resolve(), 'r', encoding='utf-8') as f:
                self.comfyui_api_json_reflex = json.load(f)

    async def heart_beat(self, id_):
        self.logger.info(f"{id_} 开始请求")

        async def get_images():

            response = await self.http_request(
                method="GET",
                target_url=f"{self.backend_id}/history/{id_}",
            )

            if response:
                for img in response[id_]['outputs'][str(self.comfyui_api_json_reflex.get('output', 9))]['images']:
                    if img['subfolder'] == "":
                        img_url = f"{self.backend_id}/view?filename={img['filename']}"
                    else:
                        img_url = f"{self.backend_id}/view?filename={img['filename']}&subfolder={img['subfolder']}"
                    self.img_url.append(img_url)

        async with aiohttp.ClientSession() as session:
            ws_url = f'{self.backend_id}/ws?clientId={self.client_id}'
            async with session.ws_connect(ws_url) as ws:

                self.logger.info(f"WS连接成功: {ws_url}")
                progress_bar = None

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        ws_msg = json.loads(msg.data)
                        #
                        # current_node = ws_msg['data']['node']

                        if ws_msg['type'] == 'progress':
                            value = ws_msg['data']['value']
                            max_value = ws_msg['data']['max']

                            if progress_bar is None:
                                progress_bar = await asyncio.to_thread(
                                    tqdm, total=max_value,
                                   desc=f"Prompt ID: {ws_msg['data']['prompt_id']}",
                                   unit="steps"
                                )

                            delta = value - progress_bar.n
                            await asyncio.to_thread(progress_bar.update, delta)

                        if ws_msg['type'] == 'executing':
                            if ws_msg['data']['node'] is None:
                                self.logger.info(f"{id_}绘画完成!")
                                await get_images()
                                await ws.close()
                    #
                    # elif msg.type == aiohttp.WSMsgType.BINARY:
                    #     if current_node == 'save_image_websocket_node':
                    #         bytes_msg = msg.data
                    #         images_output = output_images.get(current_node, [])
                    #         images_output.append(bytes_msg[8:])
                    #         output_images[current_node] = images_output

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

        try:
            response = await self.http_request(
                method="GET",
                target_url=f"{self.backend_id}/queue",
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
            return build_resp, sc, self.backend_id, sc

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

        await self.update_api_json(upload_img_resp_list)

        input_ = {
            "client_id": self.client_id,
            "prompt": self.comfyui_api_json
        }

        respone = await self.http_request(
            method="POST",
            target_url=f"{self.backend_id}/prompt",
            headers=self.headers,
            content=json.dumps(input_)
        )

        if respone.get("error", None):
            self.logger.error(respone)
            raise RuntimeError(respone["status_code"])

        self.task_id = respone['prompt_id']

        await self.heart_beat(self.task_id)
        await self.err_formating_to_sd_style()

    async def update_api_json(self, init_images):
        api_json = copy.deepcopy(self.comfyui_api_json)
        raw_api_json = copy.deepcopy(self.comfyui_api_json)

        update_mapping = {
            "sampler": {
                "seed": self.seed,
                "steps": self.steps,
                "cfg": self.scale,
                "sampler_name": self.sampler,
                "scheduler": self.scheduler,
                "denoise": self.denoising_strength
            },
            "seed": {
                "seed": self.seed,
                "noise_seed": self.seed
            },
            "image_size": {
                "width": self.width,
                "height": self.height,
                "batch_size": self.batch_size
            },
            "prompt": {
                "text": self.prompt
            },
            "negative_prompt": {
                "text": self.negative_prompt
            },
            "checkpoint": {
                "ckpt_name": self.model_path if self.model_path else None
            },
            "latentupscale": {
                    "width": int(self.width*self.hr_scale) if not self.hr_resize_x else self.hr_resize_x,
                    "height": int(self.height*self.hr_scale) if not self.hr_resize_y else self.hr_resize_y,
            },
            "load_image": {
                    "image": init_images[0]['name'] if self.init_images else None
            },
            "resize": {
                "width": int(self.width*self.hr_scale) if not self.hr_resize_x else self.hr_resize_x,
                "height": int(self.height*self.hr_scale) if not self.hr_resize_y else self.hr_resize_y,
            },
            "hr_steps": {
                "seed": self.seed,
                "steps": self.hr_second_pass_steps,
                "cfg": self.hr_scale,
                "sampler_name": self.sampler,
                "scheduler": self.scheduler,
                "denoise": self.denoising_strength,
            },
            "hr_prompt": {
                    "text": self.hr_prompt
            },
            "hr_negative_prompt": {
                    "text": self.hr_negative_prompt
            },
            "tipo": {
                "width": self.width,
                "height": self.height,
                "seed": self.seed,
                "prompt": self.prompt,
            },
            "append_prompt": {

            }
        }

        __OVERRIDE_SUPPORT_KEYS__ = {
            'keep',
            'value',
            'append_prompt',
            'append_negative_prompt',
            'remove',
            "randint",
            "get_text",
            "upscale",
            'image'

        }
        __ALL_SUPPORT_NODE__ = set(update_mapping.keys())

        for item, node_id in self.comfyui_api_json_reflex.items():

            if node_id and item not in ("override", "note"):

                org_node_id = node_id

                if isinstance(node_id, list):
                    node_id = node_id
                elif isinstance(node_id, int or str):
                    node_id = [node_id]
                elif isinstance(node_id, dict):
                    node_id = list(node_id.keys())

                for id_ in node_id:
                    id_ = str(id_)
                    update_dict = api_json.get(id_, None)
                    if update_dict and item in update_mapping:
                        api_json[id_]['inputs'].update(update_mapping[item])

                if isinstance(org_node_id, dict):
                    for node, override_dict in org_node_id.items():
                        single_node_or = override_dict.get("override", {})

                        if single_node_or:
                            for key, override_action in single_node_or.items():

                                if override_action == "randint":
                                    api_json[node]['inputs'][key] = random.randint(0, MAX_SEED)

                                elif override_action == "keep":
                                    org_cons = raw_api_json[node]['inputs'][key]

                                elif override_action == "append_prompt":
                                    prompt = raw_api_json[node]['inputs'][key]
                                    prompt = self.prompt + prompt
                                    api_json[node]['inputs'][key] = prompt

                                elif override_action == "append_negative_prompt":
                                    prompt = raw_api_json[node]['inputs'][key]
                                    prompt = self.negative_prompt + prompt
                                    api_json[node]['inputs'][key] = prompt

                                elif "upscale" in override_action:
                                    scale = 1.5
                                    if "_" in override_action:
                                        scale = override_action.split("_")[1]

                                    if key == 'width':
                                        res = self.width
                                    elif key == 'height':
                                        res = self.height

                                    upscale_size = int(res * scale)
                                    api_json[node]['inputs'][key] = upscale_size

                                elif "value" in override_action:
                                    override_value = raw_api_json[node]['inputs'][key]
                                    if "_" in override_action:
                                        override_value = override_action.split("_")[1]
                                        override_type = override_action.split("_")[2]
                                        if override_type == "int":
                                            override_value = int(override_value)
                                        elif override_type == "float":
                                            override_value = float(override_value)
                                        elif override_type == "str":
                                            override_value = str(override_value)

                                    api_json[node]['inputs'][key] = override_value

                                elif "image" in override_action:
                                    image_id = int(override_action.split("_")[1])
                                    api_json[node]['inputs'][key] = init_images[image_id]['name']

                        else:
                            update_dict = api_json.get(node, None)
                            if update_dict and item in update_mapping:
                                api_json[node]['inputs'].update(update_mapping[item])

        await run_later(self.compare_dicts(api_json, self.comfyui_api_json), 0.5)
        self.comfyui_api_json = api_json

    async def compare_dicts(self, dict1, dict2):

        modified_keys = {k for k in dict1.keys() & dict2.keys() if dict1[k] != dict2[k]}
        build_info = "节点映射情况: \n"
        for key in modified_keys:
            build_info += f"节点ID: {key} -> \n"
            for (key1, value1), (key2, value2) in zip(dict1[key].items(), dict2[key].items()):
                if value1 == value2:
                    pass
                else:
                    build_info += f"新的值: {key1} -> {value1}\n旧的值: {key2} -> {value2}\n"

        self.logger.info(build_info)

    async def upload_base64_image(self, b64_image, name, image_type="input", overwrite=False):

        if b64_image.startswith("data:image"):
            header, b64_image = b64_image.split(",", 1)
            file_type = header.split(";")[0].split(":")[1].split("/")[1]
        else:
            raise ValueError("Invalid base64 image format.")

        image_data = base64.b64decode(b64_image)

        data = aiohttp.FormData()
        data.add_field('image', image_data, filename=f"{name}.{file_type}", content_type=f'image/{file_type}')
        data.add_field('type', image_type)
        data.add_field('overwrite', str(overwrite).lower())

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.backend_id}/upload/image", data=data) as response:
                return json.loads(await response.read())
