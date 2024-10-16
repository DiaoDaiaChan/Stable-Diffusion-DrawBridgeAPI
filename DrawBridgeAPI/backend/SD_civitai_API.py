import traceback
import piexif
import os
import civitai

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


    async def update_progress(self):
        # 覆写函数
        pass

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

        self.headers['Authorization'] = f"Bearer {self.token}"
        response = await self.http_request(
            method="GET",
            target_url='https://civitai.com/api/v1/models',
            headers=self.headers,
            params=None,
            format=True
        )

        if isinstance(response, dict) and 'error' in response:
            self.fail_on_login = True
            return False
        else:
            resp_json = response
            return True, (resp_json, 200)

    async def err_formating_to_sd_style(self):

        await self.download_img()
        self.format_api_respond()
        self.result = self.build_respond

    async def posting(self):

        self.logger.info(f"开始使用{self.token}获取图片")

        os.environ['CIVITAI_API_TOKEN'] = self.token
        os.environ['HTTP_PROXY'] = self.config.civitai_setting['proxy'][self.count]
        os.environ['HTTPS_PROXY'] = self.config.civitai_setting['proxy'][self.count]
        await self.check_backend_usability()

        input_ = {
            "model": "urn:air:sd1:checkpoint:civitai:4201@130072",
            "params": {
                "prompt": self.tags,
                "negativePrompt": self.ntags,
                "scheduler": self.sampler,
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

        await self.err_formating_to_sd_style()




