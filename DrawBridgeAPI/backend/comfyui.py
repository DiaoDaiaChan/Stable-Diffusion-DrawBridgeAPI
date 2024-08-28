import asyncio
import copy
import json
import traceback

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

        if self.comfyui_api_json:
            with open(f"comfyui_workflows/{self.comfyui_api_json}.json", 'r') as f:
                self.comfyui_api_json = json.load(f)

    async def heart_beat(self, id_):
        self.logger.info(f"{id_} 开始请求")
        for i in range(600):
            await asyncio.sleep(1)
            response = await self.http_request(
                method="GET",
                target_url=f"{self.backend_url}/history/{id_}",
            )

            if response:
                for img in response[id_]['outputs']['9']['images']:
                    img_url = f"{self.backend_url}/view?filename={img['filename']}"
                    self.img_url.append(img_url)
                break
            else:
                self.logger.info(f"第{i + 1}次心跳，未返回结果")

            if response is None:
                raise RuntimeError("请求失败，未返回有效响应")


    async def update_progress(self):
        # 覆写函数
        pass

    async def get_backend_working_progress(self):

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

    async def formating_to_sd_style(self):

        await self.download_img()
        self.format_api_respond()
        self.result = self.build_respond

    async def posting(self):

        self.update_api_json()

        input_ = {
            "client_id": "114514",
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

        await self.heart_beat(respone['prompt_id'])

        await self.formating_to_sd_style()

    def update_api_json(self):
        api_json = copy.deepcopy(self.comfyui_api_json)
        # sampler
        update_dict = api_json.get('3', None)
        if update_dict:
            update = {
                "seed": self.seed,
                "steps": self.steps,
                "cfg": self.scale,
                "sampler_name": "euler",
                "scheduler": self.scheduler,
                "denoise": 1
            }
            api_json['3']['inputs'].update(update)
        # update checkpoint
        update_dict = api_json.get('4', None)
        if update_dict:
            update = {
                "ckpt_name": self.model_path
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
        self.comfyui_api_json['5']['inputs'].update(update)
        # prompt
        update_dict = api_json.get('6', None)
        if update_dict:
            update = {
                "text": self.tags
            }
        self.comfyui_api_json['6']['inputs'].update(update)
        update_dict = api_json.get('7', None)
        if update_dict:
            update = {
                "text": self.tags
            }
        self.comfyui_api_json['7']['inputs'].update(update)

        self.comfyui_api_json = api_json



