import traceback
import piexif
import aiohttp
import civitai
import os

from civitai import Civitai
from io import BytesIO

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)

        self.model = "Civitai - urn:air:sd1:checkpoint:civitai:4201@130072"
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[Civitai]')

        token = self.config.civitai[self.count]
        self.token = token
        self.backend_name = self.config.backend_name_list[0]
        self.workload_name = f"{self.backend_name}-{token}"

    async def get_models(self) -> list:
        pass
    async def update_progress(self):
        # 覆写函数
        pass

    async def get_backend_working_progress(self):
        try:
            self.logger = self.setup_logger('[Civitai]')

            token = self.config.civitai[self.count]
            self.token = token
            self.backend_name = self.config.backend_name_list[0]
            self.workload_name = f"{self.backend_name}-{token}"

            resp = await self.set_backend_working_status(get=True)

            progress = resp['idle']
            available = resp['available']
            progress = 0.99 if progress is False else 0.0

            build_resp = {
                "progress": progress,
                "eta_relative": 0.0,
                "state": {
                "skipped": False,
                "interrupted": False,
                "job": "",
                "job_count": 0,
                "job_timestamp": self.start_time,
                "job_no": 0,
                "sampling_step": 0,
                "sampling_steps": 0
                },
                "current_image": None,
                "textinfo": None
            }

            sc = 200 if available is True else 500
        except:
            traceback.print_exc()

        return build_resp, sc, self.token, sc

    async def get_img_comment(self):

        image_data = self.img_btyes[0]
        image_file = BytesIO(image_data)
        image_bytes = image_file.getvalue()
        exif_dict = piexif.load(image_bytes)
        try:
            user_comment = exif_dict['Exif'].get(piexif.ExifIFD.UserComment)
        except KeyError:
            return 'No Raw Data'

        return user_comment.decode('utf-8', errors='ignore')

    async def check_backend_usability(self):

        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get('https://civitai.com/api/v1/models') as resp:
                if resp.status != 200:
                    self.fail_on_login = True
                    return False
                else:
                    resp_json = await resp.json()
                    return True, (resp_json, resp.status)

    async def formating_to_sd_style(self):
        await self.download_img()
        try:
            comment = await self.get_img_comment()
        except:
            comment = ''

        build_respond = {
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
                "enable_hr": False,
                "firstphase_width": 0,
                "firstphase_height": 0,
                "hr_scale": 2.0,
                "hr_upscaler": None,
                "hr_second_pass_steps": 0,
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
            "info": comment
        }

        self.result = build_respond

    async def posting(self):

        self.logger.info(f"开始使用{self.token}获取图片")

        class CustomCivitai(Civitai):
            def __init__(self, api_token, env="prod"):
                self.api_token = api_token
                if not self.api_token:
                    raise ValueError("API token not provided.")

                self.base_path = "https://orchestration-dev.civitai.com" if env == "dev" else "https://orchestration.civitai.com"
                self.verify = True
                self.headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json"
                }

                self.image = self.Image(self)
                self.jobs = self.Jobs(self)

        civiai_ = CustomCivitai(api_token=self.token)
        os.environ['CIVITAI_API_TOKEN'] = self.token
        await self.check_backend_usability()
        input_ = {
            "model": "urn:air:sd1:checkpoint:civitai:4201@130072",
            "params": {
                "prompt": self.tags,
                "negativePrompt": self.ntags,
                "scheduler": None,
                "steps": self.steps,
                "cfgScale": self.scale,
                "width": self.width,
                "height": self.height,
                "clipSkip": 2,
                "seed": self.seed
            }
        }

        self.logger.info(f"任务已经发送!本次生图{self.total_img_count}张")

        for i in range(self.total_img_count):

            try:
                response = await civitai.image.create(input_, wait=True)
                if response['jobs'][0]['result'].get('available'):
                    self.img_url.append(response['jobs'][0]['result'].get('blobUrl'))
                else:
                    raise ValueError("图片没有被生成,可能是图片没有完成或者结果不可用")
            except Exception as e:
                self.fail_on_requesting = True
                self.logger.error(f"请求API失败: {e}\n{traceback.format_exc()}")

        await self.formating_to_sd_style()




