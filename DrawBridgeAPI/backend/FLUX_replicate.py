import traceback
import piexif
import os
import replicate

from io import BytesIO
from PIL import Image

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)

        self.model = "Replicate - FLUX.1 [schnell]"
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[FLUX-Replicate]')

        token = self.config.replicate[self.count]
        self.token = token
        self.backend_name = self.config.backend_name_list[3]
        self.workload_name = f"{self.backend_name}-{token}"


    async def get_shape(self):

        aspect_ratio = self.width / self.height
        tolerance = 0.05

        def is_close_to_ratio(ratio):
            return abs(aspect_ratio - ratio) < tolerance

        if self.width == self.height:
            return "1:1"
        elif is_close_to_ratio(16 / 9):
            return "16:9"
        elif is_close_to_ratio(21 / 9):
            return "21:9"
        elif is_close_to_ratio(2 / 3):
            return "2:3"
        elif is_close_to_ratio(3 / 2):
            return "3:2"
        elif is_close_to_ratio(4 / 5):
            return "4:5"
        elif is_close_to_ratio(5 / 4):
            return "5:4"
        elif is_close_to_ratio(9 / 16):
            return "9:16"
        elif is_close_to_ratio(9 / 21):
            return "9:21"
        else:
            return "2:3"

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

    async def formating_to_sd_style(self):

        await self.download_img()

        self.format_api_respond()

        self.result = self.build_respond

    async def posting(self):

        os.environ['REPLICATE_API_TOKEN'] = self.token
        image_shape = await self.get_shape()

        input_ = {
            "prompt": self.tags,
            "seed": self.seed,
            "num_outputs": self.total_img_count,
            "aspect_ratio": image_shape,
            "output_format": 'png',
            "output_quality": 90
        }

        output = await replicate.async_run(
            "black-forest-labs/flux-schnell",
            input=input_
        )

        try:
            if output:
                for i in output:
                    self.img_url.append(i)
            else:
                raise ValueError("图片没有被生成,可能是图片没有完成或者结果不可用")
        except Exception as e:
            self.fail_on_requesting = True
            self.logger.error(f"请求API失败: {e}\n{traceback.format_exc()}")

        await self.formating_to_sd_style()

