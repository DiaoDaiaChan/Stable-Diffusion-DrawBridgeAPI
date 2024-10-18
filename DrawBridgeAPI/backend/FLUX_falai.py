import traceback
import piexif
import fal_client
import os

from io import BytesIO
from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)

        self.model = "Fal-AI - FLUX.1 [schnell]"
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[FLUX-FalAI]')

        token = self.config.fal_ai[self.count]
        self.token = token
        self.backend_name = self.config.backend_name_list[2]
        self.workload_name = f"{self.backend_name}-{token}"

    async def get_shape(self):

        aspect_ratio = self.width / self.height
        tolerance = 0.05

        def is_close_to_ratio(ratio):
            return abs(aspect_ratio - ratio) < tolerance

        if self.width == self.height:
            return "square"
        elif is_close_to_ratio(4 / 3):
            return "portrait_4_3" if self.height > self.width else "landscape_4_3"
        elif is_close_to_ratio(16 / 9):
            return "portrait_16_9" if self.height > self.width else "landscape_16_9"
        else:
            return "portrait_4_3"

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
        except Exception:
            return 'No Raw Data'

        return user_comment.decode('utf-8', errors='ignore')

    async def check_backend_usability(self):
        pass

    async def err_formating_to_sd_style(self):

        await self.download_img()

        self.format_api_respond()

        self.result = self.build_respond

    async def posting(self):

        os.environ['FAL_KEY'] = self.token
        image_shape = await self.get_shape()
        self.steps = int(self.steps / 3)

        handler = await fal_client.submit_async(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": self.tags,
                "image_size": image_shape,
                "seed": self.seed,
                "num_inference_steps": self.steps,  # FLUX不需要很高的步数
                "num_images": self.total_img_count,
                "enable_safety_checker": True
            },
        )

        response = await handler.get()

        try:
            if response['images']:
                images_list = response['images']
                for i in images_list:
                    self.img_url.append(i['url'])
            else:
                raise ValueError("图片没有被生成,可能是图片没有完成或者结果不可用")
        except Exception as e:
            self.fail_on_requesting = True
            self.logger.error(f"请求API失败: {e}\n{traceback.format_exc()}")

        await self.err_formating_to_sd_style()
