import random
import uuid

import aiofiles
import aiohttp
import json
import asyncio
import traceback
import time
import httpx

from tqdm import tqdm
from fastapi import Request
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from typing import Union

from ..base_config import setup_logger
from ..base_config import init_instance
from ..utils import exceptions
from ..locales import _

import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


class Backend:

    queues = {}
    locks = {}
    task_count = 0
    queue_logger = setup_logger('[QueueManager]')

    @classmethod
    def get_queue(cls, token):
        if token not in cls.queues:
            cls.queues[token] = asyncio.Queue()
        return cls.queues[token]

    @classmethod
    def get_lock(cls, token):
        if token not in cls.locks:
            cls.locks[token] = asyncio.Lock()
        return cls.locks[token]

    @classmethod
    async def add_to_queue(cls, token, request_func, *args, **kwargs):
        queue = cls.get_queue(token)
        future = asyncio.get_event_loop().create_future()

        await queue.put((request_func, args, kwargs, future))

        lock = cls.get_lock(token)

        if not lock.locked():
            asyncio.create_task(cls.process_queue(token))

        return await future

    @classmethod
    async def process_queue(cls, token):
        queue = cls.get_queue(token)
        lock = cls.get_lock(token)

        async with lock:
            while not queue.empty():

                request_func, args, kwargs, future = await queue.get()
                try:
                    result = await request_func(*args, **kwargs)
                    if not future.done():
                        future.set_result(result)
                    cls.queue_logger.info(f"Token: {token}, {_('Task completed successfully')}")
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                    cls.queue_logger.info(f"Token: {token}, {_('Task failed')}: {e}")
                finally:
                    queue.task_done()

                cls.queue_logger.info(f"Token: {token}, {_('Remaining tasks in the queue')}")
            cls.queue_logger.info(f"Token: {token}, {_('No remaining tasks in the queue')}")

    def __init__(
        self,
        login: bool = False,
        backend_url: str = None,
        token: str = "",
        count: int = None,
        payload: dict = {},
        input_img: str = None,
        request: Request = None,
        path: str = None,
        comfyui_api_json: str = None,
        **kwargs,
    ):

        self.tags: str = payload.get('prompt', '1girl')
        self.ntags: str = payload.get('negative_prompt', '')
        self.seed: int = payload.get('seed', random.randint(0, 4294967295))
        self.seed_list: list[int] = [self.seed]
        self.steps: int = payload.get('steps', 20)
        self.scale: float = payload.get('cfg_scale', 7.0)
        self.width: int = payload.get('width', 512)
        self.height: int = payload.get('height', 512)
        self.sampler: str = payload.get('sampler_name', "Euler")
        self.restore_faces: bool = payload.get('restore_faces', False)
        self.scheduler: str = payload.get('scheduler', 'Normal')

        self.batch_size: int = payload.get('batch_size', 1)
        self.batch_count: int = payload.get('n_iter', 1)
        self.total_img_count: int = self.batch_size * self.batch_count

        self.enable_hr: bool = payload.get('enable_hr', False)
        self.hr_scale: float = payload.get('hr_scale', 1.5)
        self.hr_second_pass_steps: int = payload.get('hr_second_pass_steps', self.steps)
        self.hr_upscaler: str = payload.get('hr_upscaler', "")
        self.denoising_strength: float = payload.get('denoising_strength', 0.6)
        self.hr_resize_x: int = payload.get('hr_resize_x', 0)
        self.hr_resize_y: int = payload.get('hr_resize_y', 0)
        self.hr_sampiler: str = payload.get('hr_sampler_name', "Euler")
        self.hr_scheduler: str = payload.get('hr_scheduler', 'Normal')
        self.hr_prompt: str = payload.get('hr_prompt', '')
        self.hr_negative_prompt: str = payload.get('hr_negative_prompt', '')
        self.hr_distilled_cfg: float = payload.get('hr_distilled_cfg', 3.5)

        self.init_images: list = payload.get('init_images', [])

        self.xl = False
        self.flux = False
        self.clip_skip = 2
        self.final_width = None
        self.final_height = None
        self.model = "DiaoDaia"
        self.model_id = '20204'
        self.model_hash = "c7352c5d2f"
        self.model_list: list = []
        self.model_path = "models\\1053-S.ckpt"
        self.client_id = uuid.uuid4().hex

        self.comfyui_api_json = comfyui_api_json

        self.result: list = []
        self.time = time.strftime("%Y-%m-%d %H:%M:%S")

        self.backend_url = backend_url  # 后端url
        self.backend_id = None  # 用于区别后端, token或者ulr
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
        }  # 后端headers
        self.login = login  # 是否需要登录后端
        self.token = token  # 后端token
        self.count = count  # 适用于后端的负载均衡中遍历的后端编号
        self.config = init_instance.config  # 配置文件
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
        self.redis_client = init_instance.redis_client

        self.parameters = None  # 图片元数据
        self.post_event = None
        self.task_id = uuid.uuid4().hex
        self.task_type = 'txt2img'
        self.workload_name = None
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.save_path = ''

        self.start_time = None
        self.end_time = None
        self.spend_time = None
        self.comment = None

        self.current_process = None

        self.build_info: dict = None
        self.build_respond: dict = None

        self.nsfw_detected = False
        self.DBAPIExceptions = exceptions.DrawBridgeAPIException

        self.reflex_dict = {}

    def format_api_respond(self):

        self.build_info = {
            "prompt": self.tags,
            "all_prompts": self.repeat(self.tags)
        ,
            "negative_prompt": self.ntags,
            "all_negative_prompts": self.repeat(self.ntags)
        ,
            "seed": self.seed_list,
            "all_seeds": self.seed_list,
            "subseed": self.seed,
            "all_subseeds": self.seed_list,
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
                f"{self.tags}\\nNegative prompt: {self.ntags}\\nSteps: {self.steps}, Sampler: {self.sampler}, CFG scale: {self.scale}, Seed: {self.seed_list}, Size: {self.final_width}x{self.final_height}, Model hash: c7352c5d2f, Model: {self.model}, Denoising strength: {self.denoising_strength}, Clip skip: {self.clip_skip}, Version: 1.1.4"
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
                "seed": self.seed_list,
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
    async def write_image(img_data, save_path):
        """
        异步保存图片数据到指定路径。
        :param img_data: 图片的字节数据
        :param save_path: 保存图片的完整路径
        """
        if "view?filename=" in str(save_path):
            save_path = Path(str(save_path).replace("view?filename=", ""))
        async with aiofiles.open(save_path, 'wb') as img_file:
            await img_file.write(img_data)

    @staticmethod
    async def run_later(func, delay=1):
        loop = asyncio.get_running_loop()
        loop.call_later(
            delay,
            lambda: loop.create_task(
                func
            )
        )

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

    @staticmethod
    async def http_request(
            method,
            target_url,
            headers=None,
            params=None,
            content=None,
            format=True,
            timeout=300,
            verify=True,
            http2=False,
            use_aiohttp=False,
            proxy=False
    ) -> Union[dict, httpx.Response, bytes]:

        logger = setup_logger("[HTTP_REQUEST]")

        if use_aiohttp:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(
                        method,
                        target_url,
                        headers=headers,
                        params=params,
                        data=content,
                        ssl=verify,
                        proxy=init_instance.config.server_settings['proxy'] if proxy else None
                ) as response:
                    if format:
                        return await response.json()
                    else:
                        return await response.read()

        proxies = {
            "http://": init_instance.config.server_settings['proxy'] if proxy else None,
            "https://": init_instance.config.server_settings['proxy'] if proxy else None,
        }

        async with httpx.AsyncClient(
                verify=verify,
                http2=http2,
                proxies=proxies
        ) as client:
            try:
                response = await client.request(
                    method,
                    target_url,
                    headers=headers,
                    params=params,
                    content=content,
                    timeout=timeout,
                )
                response.raise_for_status()
            except httpx.RequestError as e:
                error_info = {"error": "Request error", "details": traceback.format_exc()}
                logger.warning(error_info)
                return error_info
            except httpx.HTTPStatusError as e:
                error_info = {"error": "HTTP error", "status_code": e.response.status_code, "details": traceback.format_exc()}
                logger.warning(error_info)
                return error_info
            if format:
                return response.json()
            else:
                return response

    def repeat(self, input_):
        # 使用列表推导式生成重复的tag列表
        repeated_ = [input_ for _ in range(self.total_img_count)]
        return repeated_

    async def exec_login(self):
        pass

    async def check_backend_usability(self):
        pass

    async def get_backend_working_progress(self):

        self.get_backend_id()

        avg_time = 0
        try:
            if self.redis_client.exists("backend_avg_time"):
                backend_avg_dict = json.loads(self.redis_client.get("backend_avg_time"))
                spend_time_list = backend_avg_dict.get(self.backend_id, [])
                if spend_time_list and len(spend_time_list) >= 10:
                    sorted_list = sorted(spend_time_list)
                    trimmed_list = sorted_list[1:-1]
                    avg_time = sum(trimmed_list) / len(trimmed_list) if trimmed_list else None

            workload_dict = await self.set_backend_working_status(get=True)
            start_time = workload_dict.get('start_time', None)
            end_time = workload_dict.get('end_time', None)
            current_time = time.time()

            if end_time:
                progress = 0.0
            else:
                if start_time:
                    spend_time = current_time - start_time
                    self.logger.info(f"当前耗时: {spend_time}")

                    if avg_time:
                        progress = 0.99 if spend_time > avg_time else spend_time / avg_time
                    else:
                        progress = 0.99
                else:
                    progress = 0.0

            available = await self.set_backend_working_status(get=True, key="available")
            sc = 200 if available is True else 500
            build_resp = self.format_progress_api_resp(progress, self.start_time)

        except:
            traceback.print_exc()

        return build_resp, sc, self.backend_id, sc

    async def send_result_to_api(self):
        """
        获取生图结果的函数
        :return: 类A1111 webui返回值
        """
        if self.backend_id is None:
            self.get_backend_id()
        total_retry = self.config.retry_times

        for retry_times in range(total_retry):
            self.start_time = time.time()

            try:
                await self.set_backend_working_status(
                    params={"start_time": self.start_time, "idle": False, "end_time": None}
                )
                # 如果传入了Request对象/转发请求
                if self.request:
                    target_url = f"{self.backend_url}/{self.path}"

                    self.logger.info(f"{_('Forwarding request')} - {target_url}")

                    method = self.request.method
                    headers = self.request.headers
                    params = self.request.query_params
                    content = await self.request.body()

                    response = await self.http_request(method, target_url, headers, params, content, False)

                    try:
                        resp = response.json()
                    except json.JSONDecodeError:
                        self.logger.error(str(response.text))
                        raise RuntimeError(_('Backend returned error'))

                    self.result = JSONResponse(content=resp, status_code=response.status_code)
                else:

                    if "comfyui" in self.backend_name:
                        await self.add_to_queue(self.backend_id[:24], self.posting)
                        self.logger.info(_('Comfyui Backend, not using built-in multi-image generation management'))
                    elif "a1111" in self.backend_name:
                        await self.add_to_queue(self.backend_id[:24], self.posting)
                        self.logger.info(_('A1111 Backend, not using built-in multi-image generation management'))
                    else:
                        self.logger.info(f"{self.backend_name}: {self.backend_id[:24]} total {self.total_img_count} images")
                        for i in range(self.total_img_count):
                            if i > 0:
                                self.seed += 1
                                self.seed_list.append(self.seed)

                            await self.add_to_queue(self.backend_id[:24], self.posting)

                    if self.config.server_settings['enable_nsfw_check']:
                        await self.pic_audit()
                break

            except Exception as e:

                self.logger.info(f"{retry_times + 1} retries")
                self.logger.error(traceback.format_exc())

                # if retry_times >= (total_retry - 1):
                #     await asyncio.sleep(30)

                if retry_times == (total_retry - 1):

                    err = traceback.format_exc()
                    self.logger.error(f"{_('Over maximum retry times, posting still failed')}: {err}")
                    await self.return_build_image(text=f"Exception: {e}", title="FATAL")
                    await self.err_formating_to_sd_style()
                    return self

            finally:
                self.end_time = time.time()
                self.spend_time = self.end_time - self.start_time
                self.logger.info(_("Request completed, took %s seconds") % int(self.spend_time))
                await self.set_backend_working_status(params={"end_time": self.end_time, "idle": True})

        return self

    async def post_request(self):
        try:
            post_api = f"{self.backend_url}/sdapi/v1/txt2img"
            if self.init_images:
                post_api = f"{self.backend_url}/sdapi/v1/img2img"

            response = await self.http_request(
                method="POST",
                target_url=post_api,
                headers=self.headers,
                content=json.dumps(self.payload),
                format=False,

            )

            if isinstance(response, httpx.Response):
                resp_dict = response.json()

                if response.status_code not in [200, 201]:
                    self.logger.error(resp_dict)
                    if resp_dict.get("error") == "OutOfMemoryError":
                        self.logger.info(_("VRAM OOM detected, auto model unload and reload"))
                        await self.unload_and_reload(self.backend_url)
                else:
                    self.result = resp_dict
                    self.logger.info(_("Get a respond image, processing"))
            else:
                self.logger.error(f"{_('Request failed, error message:')} {response.get('details')}")
            return True

        except:
            traceback.print_exc()

    async def posting(self):
        
        """
        默认为a1111webui posting
        :return:
        """
        await self.post_request()

        # self.post_event = asyncio.Event()
        # post_task = asyncio.create_task(self.post_request())
        # # 此处为显示进度条
        # while not self.post_event.is_set():
        #     await self.show_progress_bar()
        #     await asyncio.sleep(2)
        #
        # ok = await post_task

    async def download_img(self, image_list=None):
        """
        使用aiohttp下载图片并保存到指定路径。
        """

        for url in self.img_url:
            response = await self.http_request(
                method="GET",
                target_url=url,
                headers=None,
                format=False,
                verify=False,
                proxy=True
            )

            if isinstance(response, httpx.Response):
                if response.status_code == 200:
                    img_data = response.read()
                    self.logger.info(_("Downloading image successful"))
                    self.img.append(base64.b64encode(img_data).decode('utf-8'))
                    self.img_btyes.append(img_data)
                    await self.save_image(img_data)
                else:
                    self.logger.error(f"{_('Image download failed!')}: {response.status_code}")
                    raise ConnectionError(_('Image download failed!'))
            else:
                self.logger.error(f"{_('Request failed, error message:')} {response.get('details')}")

    async def save_image(self, img_data, base_path="txt2img"):

        self.save_path = Path(f'saved_images/{self.task_type}/{self.current_date}/{self.workload_name[:12]}')
        self.save_path.mkdir(parents=True, exist_ok=True)

        img_filename = self.save_path / Path(self.task_id).name
        await self.run_later(self.write_image(img_data, img_filename), 1)

    async def unload_and_reload(self, backend_url=None):
        """
        释放a1111后端的显存
        :param backend_url: 后端url地址
        :return:
        """
        # 释放模型
        response = await self.http_request(
            method="POST",
            target_url=f"{backend_url}/sdapi/v1/unload-checkpoint",
            headers=None
        )

        if isinstance(response, httpx.Response):
            if response.status_code not in [200, 201]:
                error_message = await response.text()
                self.logger.error(f"释放模型失败，可能是webui版本太旧，未支持此API，错误: {error_message}")
        else:
            self.logger.error(f"{_('Request failed, error message:')} {response.get('details')}")

        # 重载模型
        response = await self.http_request(
            method="POST",
            target_url=f"{backend_url}/sdapi/v1/reload-checkpoint",
            headers=None
        )

        if isinstance(response, httpx.Response):
            if response.status_code not in [200, 201]:
                error_message = await response.text()
                self.logger.error(f"重载模型失败，错误: {error_message}")
            else:
                self.logger.info("重载模型成功")
        else:
            self.logger.error(f"{_('Request failed, error message:')} {response.get('details')}")

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
            response = await self.http_request(
                method="GET",
                target_url=f"{self.backend_url}/sdapi/v1/progress",
                headers=None
            )

            if isinstance(response, httpx.Response):
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json.get("progress"), resp_json.get("eta_relative")
                else:
                    self.logger.error(f"获取进度失败，状态码: {response.status_code}")
                    raise RuntimeError(f"获取进度失败，状态码: {response.status_code}")
            else:
                self.logger.error(f"请求失败，错误信息: {response.get('details')}")
                raise RuntimeError(f"请求失败，错误信息: {response.get('details')}")
        except:
            traceback.print_exc()
            return 0.404

    async def set_backend_working_status(
            self,
            params: dict = None,
            get: bool = False,
            key: str = None,
    ) -> bool or None:
        """
        设置或获取后端工作状态

        :param params: 包含要更新的参数的字典 (如 {'start_time': xxx, 'idle': True})
        :param get: 是否只读取
        :param key: 要获取的键
        :return: 获取或设置结果
        """
        current_backend_workload = self.redis_client.get('workload')
        backend_workload: dict = json.loads(current_backend_workload.decode('utf-8'))
        current_backend_workload: dict = backend_workload.get(self.workload_name)

        if get:
            if key is None:
                return current_backend_workload
            return current_backend_workload.get(key, None)

        if params:
            for param_key, param_value in params.items():
                if param_key in current_backend_workload:
                    current_backend_workload[param_key] = param_value

        backend_workload[self.workload_name] = current_backend_workload
        self.redis_client.set('workload', json.dumps(backend_workload))

        return True

    async def get_models(self) -> dict:

        if self.backend_name != self.config.backend_name_list[1]:
            respond = self.format_models_resp()

            backend_to_models_dict = {
                self.workload_name: respond
            }

            return backend_to_models_dict

        else:

            self.backend_url = self.config.a1111webui_setting['backend_url'][self.count]
            try:
                respond = await self.http_request(
                    "GET",
                    f"{self.backend_url}/sdapi/v1/sd-models",
                )
            except Exception:
                self.logger.warning(f"获取模型失败")
                respond = self.format_models_resp()

            backend_to_models_dict = {
                self.workload_name: respond
            }

            return backend_to_models_dict

    async def pic_audit(self):
        from ..utils.tagger import wd_tagger_handler
        new_image_list = []
        for i in self.result['images']:
            is_nsfw = await wd_tagger_handler.tagger_main(i, 0.35, [], True)

            if is_nsfw:
                img_base64 = await self.return_build_image()
                new_image_list.append(img_base64)
            else:
                new_image_list.append(i)

        self.result['images'] = new_image_list

    async def return_build_image(self, title='Warning', text='NSFW Detected'):

        def draw_rounded_rectangle(draw, xy, radius, fill):
            x0, y0, x1, y1 = xy
            draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)  # 中间部分
            draw.rectangle([x0, y0 + radius, x0 + radius, y1 - radius], fill=fill)  # 左上角
            draw.rectangle([x1 - radius, y0 + radius, x1, y1 - radius], fill=fill)  # 右上角
            draw.pieslice([x0, y0, x0 + 2 * radius, y0 + 2 * radius], 180, 270, fill=fill)  # 左上圆角
            draw.pieslice([x1 - 2 * radius, y0, x1, y0 + 2 * radius], 270, 360, fill=fill)  # 右上圆角
            draw.pieslice([x0, y1 - 2 * radius, x0 + 2 * radius, y1], 90, 180, fill=fill)  # 左下圆角
            draw.pieslice([x1 - 2 * radius, y1 - 2 * radius, x1, y1], 0, 90, fill=fill)  # 右下圆角

        # 创建一个新的图像
        img = Image.new("RGB", (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 设置字体大小
        title_font_size = 24
        text_font_size = 16

        # 加载默认字体
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

        # 绘制标题背景
        title_x = 20  # 左对齐，设置标题的横坐标
        title_y = 20  # 设置标题的纵坐标
        title_bbox = draw.textbbox((0, 0), title, font=title_font)

        # 绘制以标题为边界的背景
        draw_rounded_rectangle(draw,
                               (title_x - 10, title_y - 10, title_x + title_bbox[2] + 10, title_y + title_bbox[3] + 10),
                               radius=10, fill=(0, 0, 0))

        # 绘制标题
        draw.text((title_x, title_y), title, fill=(255, 255, 255), font=title_font)  # 白色标题

        # 准备绘制文本，设置最大宽度
        max_text_width = img.width - 40
        wrapped_text = []
        words = text.split(' ')
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_bbox = draw.textbbox((0, 0), test_line, font=text_font)
            if text_bbox[2] - text_bbox[0] <= max_text_width:
                current_line = test_line
            else:
                wrapped_text.append(current_line)
                current_line = word

        wrapped_text.append(current_line)  # 添加最后一行

        # 绘制内容文字
        text_y = title_y + 40  # 让内容文字与标题有间距
        for line in wrapped_text:
            text_x = 20  # 左对齐，设置内容文字的横坐标

            # 绘制内容文字背景
            text_bbox = draw.textbbox((0, 0), line, font=text_font)
            draw_rounded_rectangle(draw,
                                   (text_x - 10, text_y - 5, text_x + text_bbox[2] + 10, text_y + text_bbox[3] + 5),
                                   radius=10, fill=(0, 0, 0))

            # 绘制内容文字
            draw.text((text_x, text_y), line, fill=(255, 255, 255), font=text_font)  # 白色内容文字
            text_y += text_bbox[3] - text_bbox[1] + 5  # 行间距

        # 将图像保存到内存字节流中
        img_byte_array = BytesIO()
        img.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)

        # 编码为base64
        base64_image = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
        self.img_btyes.append(img_byte_array.getvalue())
        self.img.append(base64_image)

        return base64_image

    def get_backend_id(self):
        self.backend_id = self.token or self.backend_url

    async def err_formating_to_sd_style(self):

        self.format_api_respond()

        self.result = self.build_respond
