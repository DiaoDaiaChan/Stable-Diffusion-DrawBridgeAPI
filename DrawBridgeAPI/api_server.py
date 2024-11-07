import base64
import os
import httpx
import asyncio
import time
import traceback
import json
import itertools
import argparse
import uvicorn
import logging
import warnings
import uuid
import aiofiles
import gradio
import threading

os.environ['CIVITAI_API_TOKEN'] = 'kunkun'
os.environ['FAL_KEY'] = 'Daisuki'
path_env = os.getenv("CONF_PATH")

from .utils import request_model, topaz, run_later
from .base_config import setup_logger, init_instance

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import HTTPException
from pathlib import Path

from .locales import _

app = FastAPI()

parser = argparse.ArgumentParser(description='Run the FastAPI application.')
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='The host IP address to listen on (default: 0.0.0.0).')
parser.add_argument('--port', type=int, default=8000,
                    help='The port number to listen on (default: 8000).')
parser.add_argument('--conf', '-c', type=str, default='./config.yaml',
                    help='配置文件路径', dest='conf')

args = parser.parse_args()
port = args.port
host = args.host
config_file_path = path_env or args.conf

init_instance.init(config_file_path)
config = init_instance.config
redis_client = init_instance.redis_client

from .backend import TaskHandler, Backend, StaticHandler

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = setup_logger("[API]")
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").disabled = True
logging.getLogger("fastapi").disabled = True


class Api:
    def __init__(self):
        self.app = app
        self.backend_instance = Backend()

        self.add_api_route(
            "/sdapi/v1/txt2img",
            self.txt2img_api,
            methods=["POST"],
            # response_model=request_model.Txt2ImgRequest
        )
        self.add_api_route(
            "/sdapi/v1/img2img",
            self.img2img_api,
            methods=["POST"],
            # response_model=request_model.Img2ImgRequest
        )
        self.add_api_route(
            "/sdapi/v1/sd-models",
            self.get_sd_models,
            methods=["GET"]
        )
        self.add_api_route(
            "/sdapi/v1/progress",
            self.get_progress,
            methods=["GET"]
        )
        self.add_api_route(
            "/sdapi/v1/memory",
            self.get_memory,
            methods=["GET"]
        )
        self.add_api_route(
            "/sdapi/v1/options",
            self.get_options,
            methods=["GET"]
        )
        self.add_api_route(
            "/sdapi/v1/options",
            self.set_options,
            methods=["POST"]
        )
        self.add_api_route(
            "/sdapi/v1/prompt-styles",
            self.get_prompt_styles,
            methods=["GET"]
        )

        if config.server_settings['build_in_tagger']:

            from .utils.tagger import wd_tagger_handler, wd_logger
            self.add_api_route(
                "/tagger/v1/interrogate",
                self.tagger,
                methods=["POST"],
                response_model=request_model.TaggerRequest
            )

            if config.server_settings['llm_caption']['enable']:
                from .utils.llm_captions import llm_logger, joy_caption_handler
                self.add_api_route(
                    "/llm/caption",
                    self.llm_caption,
                    methods=["POST"],
                    response_model=request_model.TaggerRequest
                )

        if config.server_settings['build_in_photoai']['exec_path']:
            self.add_api_route(
                "/topazai/image",
                self.topaz_ai,
                methods=["POST"]
            )

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    @staticmethod
    async def generate_handle(data) -> TaskHandler:

        model_to_backend = None
        if data['override_settings'].get("sd_model_checkpoint", None):
            model_to_backend = data['override_settings'].get("sd_model_checkpoint", None)

        styles = data.get('styles', [])
        selected_style = []
        selected_comfyui_style = []

        logger.error(styles)

        if styles:
            api_styles = StaticHandler.get_prompt_style()

            for index, i in enumerate(api_styles):
                for style in styles:
                    if style in i['name']:
                        if 'comfyui' in i['name']:
                            logger.info(f"{_('Selected ComfyUI style')} - {i['name']}")
                            selected_comfyui_style.append(i['name'])
                        else:
                            selected_style.append(i['name'])

        if selected_style:
            for i in selected_style:
                data['prompt'] = data.get('prompt', '') + i['prompt']
                data['negative_prompt'] = data.get('negative_prompt', '') + i['negative_prompt']

        task_handler = TaskHandler(
            data,
            model_to_backend=model_to_backend,
            comfyui_json=selected_comfyui_style[0].replace('comfyui-work-flows-', '') if selected_comfyui_style else None
        )

        return task_handler

    @staticmethod
    async def txt2img_api(request: request_model.Txt2ImgRequest, api: Request):

        data = request.model_dump()
        client_host = api.client.host

        task_handler = await Api.generate_handle(data)

        try:
            logger.info(f"{_('Exec TXT2IMG')} - {client_host}")
            result = await task_handler.txt2img()
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        if result is None:
            raise HTTPException(500, detail='Result not found')

        return result

    @staticmethod
    async def img2img_api(request: request_model.Img2ImgRequest, api: Request):
        data = request.model_dump()
        client_host = api.client.host

        if len(data['init_images']) == 0:
            raise HTTPException(status_code=400, detail=_('IMG2IMG Requires image to start'))

        task_handler = await Api.generate_handle(data)

        try:
            logger.info(f"{_('Exec IMG2IMG')} - {client_host}")
            result = await task_handler.img2img()
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        if result is None:
            raise HTTPException(500, detail='Result not found')

        return result

    @staticmethod
    async def get_sd_models():

        task_list = []
        path = '/sdapi/v1/sd-models'

        task_handler = TaskHandler({}, None, path, reutrn_instance=True, override_model_select=True)
        instance_list: list[Backend] = await task_handler.txt2img()

        for i in instance_list:
            task_list.append(i.get_models())
        resp = await asyncio.gather(*task_list)

        models_dict = {}
        api_respond = []
        for i in resp:
            models_dict = models_dict | i
            api_respond = api_respond + list(i.values())

        api_respond = list(itertools.chain.from_iterable(api_respond))

        redis_resp: bytes = redis_client.get('models')
        redis_resp: dict = json.loads(redis_resp.decode('utf-8'))
        redis_resp.update(models_dict)
        redis_client.set('models', json.dumps(redis_resp))
        return api_respond

    async def tagger(self, request: request_model.TaggerRequest):
        from .utils.tagger import wd_tagger_handler, wd_logger

        data = request.model_dump()
        base64_image = await self.download_img_from_url(data)
        caption = await wd_tagger_handler.tagger_main(base64_image, data['threshold'], data['exclude_tags'])
        resp = {}

        resp['caption'] = caption
        wd_logger.info(f"{_('Caption Successful')}, {caption}")
        return JSONResponse(resp)

    async def llm_caption(self, request: request_model.TaggerRequest):

        from .utils.llm_captions import llm_logger, joy_caption_handler
        from .utils.tagger import wd_tagger_handler, wd_logger

        data = request.model_dump()
        base64_image = await self.download_img_from_url(data)

        try:
            caption = await joy_caption_handler.get_caption(base64_image, data['exclude_tags'])
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

        resp = {}

        resp['llm'] = caption
        llm_logger.info(f"{_('Caption Successful')}, {caption}")
        # caption = await wd_tagger_handler.tagger_main(
        #     base64_image,
        #     data['threshold'],
        #     data['exclude_tags']
        # )
        #
        # resp['caption'] = caption
        # wd_logger.info(f"打标成功,{caption}")
        return JSONResponse(resp)

    async def get_progress(self):
        return JSONResponse(self.backend_instance.format_progress_api_resp(0.0, time.time()))

    async def get_memory(self):
        return JSONResponse(self.backend_instance.format_vram_api_resp())

    @staticmethod
    async def get_options():
        return JSONResponse(StaticHandler.get_backend_options())

    @staticmethod
    async def set_options(request: request_model.SetConfigRequest):

        data = request.model_dump()
        if data.get('sd_model_checkpoint', None):
            logger.info(_("Lock to backend has configured"))
            StaticHandler.set_lock_to_backend(data.get('sd_model_checkpoint'))

        return

    @staticmethod
    async def topaz_ai(request: request_model.TopazAiRequest):
        data = request.model_dump()

        unique_id = str(uuid.uuid4())
        save_dir = Path("saved_images") / unique_id
        processed_dir = save_dir / 'processed'
        save_dir.mkdir(parents=True, exist_ok=True)
        del data['output_folder']

        try:

            if data['image']:
                base64_image = data['image']
                input_image_path = save_dir / f"{unique_id}_image.png"
                async with aiofiles.open(input_image_path, "wb") as image_file:
                    await image_file.write(base64.b64decode(base64_image))
                output, error, return_code = await asyncio.get_running_loop().run_in_executor(
                    None, topaz.run_tpai(
                        input_folder=str(save_dir.resolve()),
                        output_folder=str(processed_dir.resolve()),
                        **data
                    )
                )
            elif data['input_folder']:
                output, error, return_code = await asyncio.get_running_loop().run_in_executor(
                    None, topaz.run_tpai(
                        output_folder=str(processed_dir.resolve()),
                        **data
                    )
                )
        except:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Error occurred while processing the image.")

        if return_code == 0:
            files = list(processed_dir.glob("*"))

            processed_image_path = files[0]
            if processed_image_path.exists():
                async with aiofiles.open(processed_image_path, "rb") as img_file:
                    encoded_image = base64.b64encode(await img_file.read()).decode('utf-8')
                processed_dir.rmdir()
                return {"status": "success", "image": encoded_image}
            else:
                raise HTTPException(status_code=500, detail="Processed image not found.")
        else:
            raise HTTPException(status_code=500, detail=f"Error: {error}")

    async def download_img_from_url(self, data):

        base64_image = data['image']

        if data['image'].startswith("http"):
            image_url = data['image']
            logger.info(f"{_('URL detected')}: {image_url}")
            response = await self.backend_instance.http_request(
                "GET",
                image_url,
                format=False
            )

            if response.status_code != 200:
                logger.warning(_("Image download failed!"))

            base64_image = base64.b64encode(response.read())

        return base64_image

    @staticmethod
    async def get_prompt_styles():

        task_list = []
        path = '/sdapi/v1/prompt-styles'

        task_handler = TaskHandler({}, None, path, reutrn_instance=True, override_model_select=True)
        instance_list: list[Backend] = await task_handler.txt2img()

        for i in instance_list:
            task_list.append(i.get_all_prompt_style())
        resp = await asyncio.gather(*task_list)

        api_respond = []
        for i in resp:
            api_respond += i

        StaticHandler.set_prompt_style(api_respond)

        return api_respond

    async def init_api(self):
        await self.get_sd_models()
        await self.get_prompt_styles()


api_instance = Api()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    client_host = request.client.host

    task_handler = TaskHandler({}, request, path)

    try:
        logger.info(f"{_('Exec forwarding')} - {client_host}")
        result = await task_handler.sd_api()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, detail=str(e))

    if result is None:
        raise HTTPException(500, detail='Result not found')

    return result


@app.get("/backend-control")
async def get_backend_control(backend: str, key: str, value: bool):
    pass


@app.on_event("startup")
async def startup_event():
    logger.info(_('Waiting for API initialization'))
    await api_instance.init_api()
    logger.info(_('API initialization completed'))


if __name__ == "__main__":

    # if config.server_settings['start_gradio']:
    #     demo = create_gradio_interface(host, port)
    #     app = gradio.mount_gradio_app(api_instance.app, demo, path="/")

    uvicorn.run(api_instance.app, host=host, port=port)
