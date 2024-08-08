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

    def format_api_respond(self):

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

    @staticmethod
    def format_progress_api_resp(progress, start_time) -> dict:
        build_resp = {
            "progress": progress,
            "eta_relative": 0.0,
            "state": {
                "skipped": False,
                "interrupted": False,
                "job": "",
                "job_count": 0,
                "job_timestamp": start_time,
                "job_no": 0,
                "sampling_step": 0,
                "sampling_steps": 0
            },
            "current_image": "data:image/png;base64,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",
            "textinfo": None
        }

        return build_resp

    @staticmethod
    def format_vram_api_resp():

        build_resp = {
          "ram": {
            "free": 61582063428.50122,
            "used": 2704183296,
            "total": 64286246724.50122
          },
          "cuda": {
            "system": {
              "free": 4281335808,
              "used": 2160787456,
              "total": 85899345920
            },
            "active": {
              "current": 699560960,
              "peak": 3680867328
            },
            "allocated": {
              "current": 699560960,
              "peak": 3680867328
            },
            "reserved": {
              "current": 713031680,
              "peak": 3751804928
            },
            "inactive": {
              "current": 13470720,
              "peak": 650977280
            },
            "events": {
              "retries": 0,
              "oom": 0
            }
          }
        }
        return build_resp


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









