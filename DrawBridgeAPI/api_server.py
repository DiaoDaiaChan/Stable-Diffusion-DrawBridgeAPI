import base64
import os
import httpx

os.environ['CIVITAI_API_TOKEN'] = 'kunkun'
os.environ['FAL_KEY'] = 'Daisuki'
from backend import TaskHandler, Backend
from base_config import setup_logger, redis_client, config
from utils import request_model

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

import asyncio
import time
import traceback
import json
import ui
import itertools
import argparse
import uvicorn
import threading
import logging
import warnings


app = FastAPI()

parser = argparse.ArgumentParser(description='Run the FastAPI application.')
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='The host IP address to listen on (default: 0.0.0.0).')
parser.add_argument('--port', type=int, default=8000,
                    help='The port number to listen on (default: 8000).')

args = parser.parse_args()
port = args.port
host = args.host

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = setup_logger("[API]")
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").disabled = True
logging.getLogger("fastapi").disabled = True


class Api:
    def __init__(self):
        self.app = app
        self.wd_instance = None
        self.pipeline = None
        self.joy_caption_instance = None
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

        if config.server_settings['build_in_tagger']:
            from utils.tagger import WaifuDiffusionInterrogator

            wd_instance = WaifuDiffusionInterrogator(
                name='WaifuDiffusion',
                repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
                revision='v2.0',
                model_path='model.onnx',
                tags_path='selected_tags.csv'
            )

            wd_instance.load()

            self.wd_instance = wd_instance

            self.add_api_route(
                "/tagger/v1/interrogate",
                self.llm_caption if config.server_settings['llm_caption']['enable'] else self.tagger,
                methods=["POST"],
                response_model=request_model.TaggerRequest
            )

            if config.server_settings['llm_caption']['enable']:
                from utils.llm_captions import (
                    pipeline,
                    joy_caption_instance,
                    llm_logger,
                    get_caption
                )

                self.add_api_route(
                    "/llm/caption",
                    self.llm_caption,
                    methods=["POST"],
                    response_model=request_model.TaggerRequest
                )

                llm_logger.info("LLM加载中")

                self.pipeline = pipeline
                self.joy_caption_instance = joy_caption_instance
                llm_logger.info("LLM加载完成,等待命令")

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    @staticmethod
    async def txt2img_api(request: request_model.Txt2ImgRequest, api: Request):

        data = request.dict()
        client_host = api.client.host
        model_to_backend = None

        if data['override_settings'].get("sd_model_checkpoint", None):
            model_to_backend = data['override_settings'].get("sd_model_checkpoint", None)

        task_handler = TaskHandler(data, model_to_backend=model_to_backend)
        try:
            logger.info(f"开始进行文生图 - {client_host}")
            result = await task_handler.txt2img()
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

        if result is None:
            raise HTTPException(500, detail='Result not found')

        return result

    @staticmethod
    async def img2img_api(request: request_model.Txt2ImgRequest, api: Request):
        data = request.dict()
        client_host = api.client.host

        if len(data['init_images']) == 0:
            raise HTTPException(status_code=400, detail='图生图需要图片来启动')

        task_handler = TaskHandler(data)

        try:
            logger.info(f"开始进行图生图 - {client_host}")
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

        task_handler = TaskHandler({}, None, path, reutrn_instance=True)
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
        from utils.tagger import tagger_main, wd_logger
        data = request.model_dump()
        caption = await asyncio.get_event_loop().run_in_executor(
            None,
            tagger_main,
            data['image'],
            data['threshold'],
            self.wd_instance,
            data['exclude_tags']
        )

        resp = {}
        resp['caption'] = caption
        wd_logger.info(f"打标成功,{caption}")
        return JSONResponse(resp)

    async def llm_caption(self, request: request_model.TaggerRequest):
        from utils.llm_captions import (
            llm_logger,
            get_caption
        )
        data = request.model_dump()
        base64_image = data['image']
        if data['image'].startswith("http"):
            image_url = data['image']
            llm_logger.info(f"检测到url{image_url}")
            response: httpx.Response = await self.backend_instance.http_request(
                "POST",
                image_url,
                format=False
            )
            if response.status_code != 200:
                llm_logger.warning("图片下载失败!")
            base64_image = base64.b64encode(response.read())
        try:
            caption = await asyncio.get_event_loop().run_in_executor(
            None,
            get_caption,
            self.pipeline,
            self.joy_caption_instance,
            base64_image,
            data['exclude_tags']
        )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

        resp = {}
        resp['llm'] = caption
        llm_logger.info(f"打标成功,{caption}")

        from utils.tagger import tagger_main, wd_logger
        caption = await asyncio.get_event_loop().run_in_executor(
            None,
            tagger_main,
            base64_image,
            data['threshold'],
            self.wd_instance
        )

        resp['caption'] = caption
        wd_logger.info(f"打标成功,{caption}")
        return JSONResponse(resp)

    async def get_progress(self):
        return JSONResponse(self.backend_instance.format_progress_api_resp(0.0, time.time()))

    async def get_memory(self):
        return JSONResponse(self.backend_instance.format_vram_api_resp())

    async def get_options(self):
        return JSONResponse(self.backend_instance.format_options_api_resp())


api_instance = Api()


@app.post('/sdapi/v1/prompt-styles')
async def _(request):
    pass


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    client_host = request.client.host

    task_handler = TaskHandler({}, request, path)

    try:
        logger.info(f"开始进行转发 - {client_host}")
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


threading.Thread(
    target=uvicorn.run,
    args=(app,),
    kwargs={
        "host": host,
        "port": port,
        "log_level": "critical"
    }
).start()

asyncio.run(ui.run_gradio(host, port))


