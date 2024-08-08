import random

import aiohttp
import base64
import json
import asyncio
import traceback
import time
import httpx
from PIL import Image

from tqdm import tqdm
from fastapi import Request
from fastapi.responses import JSONResponse
from io import BytesIO
from copy import deepcopy

from base_config import setup_logger
from base_config import redis_client, config
from utils import http_request


class Backend:

    def __init__(
        self,
        login: bool = False,
        backend_url: str = None,
        token: str = None,
        count: int = None,
        payload: dict = {},
        input_img: str = None,
        request: Request = None,
        path: str = None,
        **kwargs,
    ):

        self.tags: str = payload.get('prompt', '1girl')
        self.ntags: str = payload.get('negative_prompt', '')
        self.seed: int = payload.get('seed', -1)
        self.steps: int = payload.get('steps', 20)
        self.scale: float = payload.get('cfg_scale', 7.0)
        self.width: int = payload.get('width', 512)
        self.height: int = payload.get('height', 512)
        self.sampler: str = payload.get('sampler_name', None)

        self.batch_size: int = payload.get('batch_size', 1)
        self.batch_count: int = payload.get('n_iter', 1)
        self.total_img_count: int = self.batch_size * self.batch_count

        self.enable_hr: bool = payload.get('enable_hr', False)
        self.hr_scale: float = payload.get('hr_scale', 1.5)
        self.hr_second_pass_steps: int = payload.get('hr_second_pass_steps', 7)
        self.denoising_strength: float = payload.get('denoising_strength', 0.6)

        self.init_images: list = payload.get('init_images', None)
        if self.init_images is not None and len(self.init_images) == 0:
            self.init_images = None

        self.clip_skip = 2
        self.final_width = None
        self.final_height = None
        self.model = "DiaoDaia"
        self.model_hash = "c7352c5d2f"
        self.model_list: list = []

        self.result: list = []
        self.time = time.strftime("%Y-%m-%d %H:%M:%S")

        self.backend_url = backend_url  # 后端url
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
        }  # 后端headers
        self.login = login  # 是否需要登录后端
        self.token = token  # 后端token
        self.count = count  # 适用于后端的负载均衡中遍历的后端编号
        self.config = config  # 配置文件
        self.backend_name = ''  # 后端名称
        self.current_config = None  # 当前后端的配置

        self.fail_on_login = None
        self.fail_on_requesting = None

        self.result = None  # api返回的结果
        self.img = []  # 返回的图片
        self.img_url = []
        self.img_btyes = []
        self.input_img = input_img

        self.payload = payload  # post时使用的负载
        self.request = request
        self.path = path

        self.logger = None
        self.setup_logger = setup_logger
        self.redis_client = redis_client

        self.parameters = None  # 图片元数据
        self.post_event = None
        self.task_id = None
        self.workload_name = None

        self.start_time = None
        self.end_time = None
        self.comment = None

        self.current_process = None

        self.build_info: dict = None
        self.build_respond: dict = None

    def build_api_respond(self):

        self.build_info = {
            "prompt": self.tags,
            "all_prompts": self.repeat(self.tags)
        ,
            "negative_prompt": self.ntags,
            "all_negative_prompts": self.repeat(self.ntags)
        ,
            "seed": self.seed,
            "all_seeds": self.repeat(self.seed),
            "subseed": self.seed,
            "all_subseeds": self.repeat(self.seed),
            "subseed_strength": 0,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler,
            "cfg_scale": self.scale,
            "steps": self.steps,
            "batch_size": 1,
            "restore_faces": False,
            "face_restoration_model": None,
            "sd_model_name": self.model,
            "sd_model_hash": self.model_hash,
            "sd_vae_name": 'no vae',
            "sd_vae_hash": self.model_hash,
            "seed_resize_from_w": -1,
            "seed_resize_from_h": -1,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": {

            },
            "index_of_first_image": 0,
            "infotexts": self.repeat(
                f"{self.tags}\\nNegative prompt: {self.ntags}\\nSteps: {self.steps}, Sampler: {self.sampler}, CFG scale: {self.scale}, Seed: {self.seed}, Size: {self.final_width}x{self.final_height}, Model hash: c7352c5d2f, Model: {self.model}, Denoising strength: {self.denoising_strength}, Clip skip: {self.clip_skip}, Version: 1.1.4"
            )
        ,
            "styles": [

            ],
            "job_timestamp": "0",
            "clip_skip": self.clip_skip,
            "is_using_inpainting_conditioning": False
        }

        self.build_respond = {
            "images": self.img,
            "parameters": {
                "prompt": self.tags,
                "negative_prompt": self.ntags,
                "seed": self.seed,
                "subseed": -1,
                "subseed_strength": 0,
                "seed_resize_from_h": -1,
                "seed_resize_from_w": -1,
                "sampler_name": '',
                "batch_size": 1,
                "n_iter": self.total_img_count,
                "steps": self.steps,
                "cfg_scale": self.scale,
                "width": self.width,
                "height": self.height,
                "restore_faces": None,
                "tiling": None,
                "do_not_save_samples": None,
                "do_not_save_grid": None,
                "eta": None,
                "denoising_strength": 0,
                "s_min_uncond": None,
                "s_churn": None,
                "s_tmax": None,
                "s_tmin": None,
                "s_noise": None,
                "override_settings": None,
                "override_settings_restore_afterwards": True,
                "refiner_checkpoint": None,
                "refiner_switch_at": None,
                "disable_extra_networks": False,
                "comments": None,
                "enable_hr": True if self.enable_hr else False,
                "firstphase_width": 0,
                "firstphase_height": 0,
                "hr_scale": self.hr_scale,
                "hr_upscaler": None,
                "hr_second_pass_steps": self.hr_second_pass_steps,
                "hr_resize_x": 0,
                "hr_resize_y": 0,
                "hr_checkpoint_name": None,
                "hr_sampler_name": None,
                "hr_prompt": "",
                "hr_negative_prompt": "",
                "sampler_index": "Euler",
                "script_name": None,
                "script_args": [],
                "send_images": True,
                "save_images": False,
                "alwayson_scripts": {}
            },

            "info": ''
        }
        image = Image.open(BytesIO(self.img_btyes[0]))
        self.final_width, self.final_height = image.size

        str_info = json.dumps(self.build_info)
        self.build_respond['info'] = str_info

    def format_models_resp(self, input_list=None):
        models_resp_list = []
        input_list = input_list if input_list else [self.model]
        for i in input_list:
            built_reps = {
                "title": f"{i} [{self.model_hash}]",
                "model_name": i,
                "hash": f"{self.model_hash}",
                "sha256": "03f33720f33b67634b5da3a8bf2e374ef90ea03e85ab157fcf89bf48213eee4e",
                "filename": self.backend_name,
                "config": None
            }
            models_resp_list.append(built_reps)

        return models_resp_list

    def repeat(self, input_):
        # 使用列表推导式生成重复的tag列表
        repeated_ = [input_ for _ in range(self.total_img_count)]
        return repeated_

    async def exec_login(self):
        pass

    async def check_backend_usability(self):
        pass

    async def get_backend_working_progress(self):
        pass

    async def send_result_to_api(self) -> JSONResponse:
        """
        获取生图结果的函数
        :return: 类A1111 webui返回值
        """
        total_retry = config.retry_times

        for retry_times in range(total_retry):
            self.start_time = time.time()
            await self.set_backend_working_status(self.start_time, True)
            try:
                await self.set_backend_working_status(idle=False)
                # 如果传入了Request对象/转发请求
                if self.request:

                    target_url = f"{self.backend_url}/{self.path}"
                    self.logger.info(f"已转发请求 - {target_url}")

                    method = self.request.method
                    headers = self.request.headers
                    params = self.request.query_params
                    content = await self.request.body()

                    response = await http_request(method, target_url, headers, params, content, False)

                    resp = response.json()
                    if self.path == 'sdapi/v1/progress':
                        resp["current_image"] = "data:image/jpeg;base64,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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAf/2Q=="

                    self.result = JSONResponse(content=resp, status_code=response.status_code)
                else:
                    await self.posting()

            except Exception as e:

                self.logger.info(f"第{retry_times + 1}次尝试")
                self.logger.error(traceback.format_exc())

                if retry_times >= (total_retry - 1):
                    await asyncio.sleep(30)

                if retry_times > total_retry:
                    raise RuntimeError(f"重试{total_retry}次后仍然发生错误, 请检查服务器")

            finally:
                self.end_time = time.time()
                self.logger.info(f"请求完成，共耗时{self.end_time - self.start_time}")
                await self.set_backend_working_status(idle=True)

        return self.result

    async def post_request(self):
        try:
            post_api = f"{self.backend_url}/sdapi/v1/txt2img"
            if self.input_img:
                post_api = f"{self.backend_url}/sdapi/v1/img2img"

            async with aiohttp.ClientSession(
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=1800)
            ) as session:
                # 向服务器发送请求
                async with session.post(post_api, json=self.payload) as resp:
                    resp_dict = json.loads(await resp.text())
                    if resp.status not in [200, 201]:
                        self.post_event.is_set()
                        self.logger.error(resp_dict)
                        if resp_dict["error"] == "OutOfMemoryError":
                            self.logger.info("检测到爆显存，执行自动模型释放并加载")
                            await self.unload_and_reload(self.backend_url)
                    else:
                        self.result = resp_dict
                        self.logger.info(f"获取到返回图片，正在处理")
                        self.post_event.set()
            return True

        except:
            traceback.print_exc()

    async def posting(self):
        
        """
        默认为a1111webui posting
        :return:
        """

        self.post_event = asyncio.Event()
        post_task = asyncio.create_task(self.post_request())
        # 此处为显示进度条
        while not self.post_event.is_set():
            await self.show_progress_bar()
            await asyncio.sleep(2)

        ok = await post_task

    async def download_img(self, image_list=None):
        """
        使用aiohttp下载图片
        :return:
        """
        for url in self.img_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:

                    if response.status == 200:
                        img_data = await response.read()
                        self.logger.info("图片下载成功")
                        self.img.append(base64.b64encode(img_data).decode('utf-8'))
                        self.img_btyes.append(img_data)
                    else:
                        self.logger.error(f"图片下载失败！{response.status}")
                        raise ConnectionError("图片下载失败")

    async def unload_and_reload(self, backend_url=None):
        """
        释放a1111后端的显存
        :param backend_url: 后端url地址
        :return:
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(url=f"{backend_url}/sdapi/v1/unload-checkpoint") as resp:
                if resp.status not in [200, 201]:
                    self.logger.error(f"释放模型失败，可能是webui版本太旧，未支持此API，错误:{await resp.text()}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url=f"{backend_url}/sdapi/v1/reload-checkpoint") as resp:
                if resp.status not in [200, 201]:
                    self.logger.error(f"重载模型失败，错误:{await resp.text()}")
                self.logger.info("重载模型成功")

    async def get_backend_status(self):
        """
        共有函数, 用于获取各种类型的后端的工作状态
        :return:
        """
        await self.check_backend_usability()
        resp_json, resp_status = await self.get_backend_working_progress()

        return resp_json, resp_status

    async def show_progress_bar(self):
        """
        在控制台实时打印后端工作进度进度条
        :return:
        """
        show_str = f"[SD-A1111] [{self.time}] : {self.seed}"
        show_str = show_str.ljust(25, "-")
        with tqdm(total=1, desc=show_str + "-->", bar_format="{l_bar}{bar}|{postfix}\n") as pbar:
            while not self.post_event.is_set():
                self.current_process, eta = await self.update_progress()
                increment = self.current_process - pbar.n
                pbar.update(increment)
                pbar.set_postfix({"eta": f"{int(eta)}秒"})
                await asyncio.sleep(2)

    async def update_progress(self):
        """
        更新后端工作进度
        :return:
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url=f"{self.backend_url}/sdapi/v1/progress") as resp:
                    resp_json = await resp.json()
                    return resp_json["progress"], resp_json["eta_relative"]
        except:
            traceback.print_exc()
            return 0.404

    async def set_backend_working_status(
            self,
            start_time=None,
            idle=None,
            available=None,
            get=False,
            key=None,
    ) -> bool or None:
        """
        :param start_time : 任务开始时间
        :param idle : 后端是否待机
        :param available: 后端是否可以使用
        :param get: 是否只读取
        :param key: 要获取的键
        :return:
        """
        current_backend_workload: bytes = self.redis_client.get('workload')
        backend_workload: dict = json.loads(current_backend_workload.decode('utf-8'))
        current_backend_workload = backend_workload.get(self.workload_name)

        if get:
            if key is None:
                return current_backend_workload
            return current_backend_workload[key]

        if start_time:
            current_backend_workload['start_time'] = start_time

        if idle is not None:
            current_backend_workload['idle'] = idle

        if available is not None:
            current_backend_workload['available'] = available

        backend_workload[self.workload_name] = current_backend_workload

        self.redis_client.set(f"workload", json.dumps(backend_workload))
        #
        # elif redis_key == 'models':
        #     models: bytes = self.redis_client.get(redis_key)
        #     models: dict = json.loads(models.decode('utf-8'))
        #     models[self.workload_name] = self.model_list
        #     rp = self.redis_client.pipeline()
        #     self.redis_client.set()
        #     current_backend_workload = models.get(self.workload_name)
        #     current_backend_workload

    async def get_models(self) -> dict:

        if self.backend_name != self.config.backend_name_list[1]:
            respond = self.format_models_resp()

            backend_to_models_dict = {
                self.workload_name: respond
            }

            return backend_to_models_dict

        else:

            self.backend_url = self.config.a1111webui_setting['backend_url'][self.count]
            respond = await http_request(
                "GET",
                f"{self.backend_url}/sdapi/v1/sd-models"
            )

            backend_to_models_dict = {
                self.workload_name: respond
            }

            return backend_to_models_dict









