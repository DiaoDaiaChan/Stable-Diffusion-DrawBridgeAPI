import asyncio
import time
import traceback
import json
import os
import warnings
import itertools
import requests
import argparse
import uvicorn
import threading
import logging
import warnings

os.environ['CIVITAI_API_TOKEN'] = 'kunkun'
os.environ['FAL_KEY'] = 'Daisuki'

from backend import Task_Handler, Backend
from base_config import setup_logger, redis_client, config

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# 忽略 RuntimeWarning
warnings.simplefilter("ignore", RuntimeWarning)

app = FastAPI()

parser = argparse.ArgumentParser(description='Run the FastAPI application.')
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='The host IP address to listen on (default: 0.0.0.0).')
parser.add_argument('--port', type=int, default=8000,
                    help='The port number to listen on (default: 8000).')

args = parser.parse_args()
port = args.port

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = setup_logger("[API]")
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").disabled = True
logging.getLogger("fastapi").disabled = True


class Txt2ImgRequest(BaseModel):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    styles: List[str] = []
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    sampler_name: str = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7
    width: int = 512
    height: int = 512
    restore_faces: bool = True
    tiling: bool = True
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    eta: float = 0
    denoising_strength: float = 0
    s_min_uncond: float = 0
    s_churn: float = 0
    s_tmax: float = 0
    s_tmin: float = 0
    s_noise: float = 0
    override_settings: Dict[str, Any] = {}
    override_settings_restore_afterwards: bool = True
    refiner_checkpoint: str = None
    refiner_switch_at: int = 0
    disable_extra_networks: bool = False
    comments: Dict[str, Any] = {}
    enable_hr: bool = False
    firstphase_width: int = 0
    firstphase_height: int = 0
    hr_scale: float = 2
    hr_upscaler: str = None
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_checkpoint_name: str = None
    hr_sampler_name: str = None
    hr_prompt: str = ""
    hr_negative_prompt: str = ""
    sampler_index: str = "Euler"
    script_name: str = None
    script_args: List[Any] = []
    send_images: bool = True
    save_images: bool = False
    alwayson_scripts: Dict[str, Any] = {}


class Img2ImgRequest(BaseModel):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    styles: List[str] = []
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    sampler_name: str = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7
    width: int = 512
    height: int = 512
    restore_faces: bool = True
    tiling: bool = True
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    eta: float = 0
    denoising_strength: float = 0.75
    s_min_uncond: float = 0
    s_churn: float = 0
    s_tmax: float = 0
    s_tmin: float = 0
    s_noise: float = 0
    override_settings: Dict[str, Any] = {}
    override_settings_restore_afterwards: bool = True
    refiner_checkpoint: str = None
    refiner_switch_at: int = 0
    disable_extra_networks: bool = False
    comments: Dict[str, Any] = {}
    init_images: List[str] = [None]
    resize_mode: int = 0
    image_cfg_scale: float = 0
    mask: str = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = 0
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    initial_noise_multiplier: float = 0
    latent_mask: str = None
    sampler_index: str = "Euler"
    include_init_images: bool = False
    script_name: str = None
    script_args: List[Any] = []
    send_images: bool = True
    save_images: bool = False
    alwayson_scripts: Dict[str, Any] = {}
    # 以下为拓展


class TaggerRequest(BaseModel):
    image: str = '',
    model: str = 'wd14-vit-v2'
    threshold: float = 0.35


@app.on_event("startup")
def startup_event():
    threading.Thread(target=make_request).start()


def make_request():
    response = requests.get(f"http://localhost:{port}/sdapi/v1/sd-models")


# 创建两个 POST endpoint
@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Txt2ImgRequest, api: Request):

    data = request.dict()
    client_host = api.client.host
    model_to_backend = None

    if data['override_settings'].get("sd_model_checkpoint", None):
        model_to_backend = data['override_settings'].get("sd_model_checkpoint", None)

    task_handler = Task_Handler(data, model_to_backend=model_to_backend)
    try:
        logger.info(f"开始进行文生图 - {client_host}")
        result = await task_handler.txt2img()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(500, detail='Result not found')

    return result


@app.post("/sdapi/v1/img2img")
async def img2img(request: Img2ImgRequest, api: Request):

    data = request.dict()
    client_host = api.client.host

    if len(data['init_images']) == 0:
        raise HTTPException(status_code=400, detail='图生图需要图片来启动')

    task_handler = Task_Handler(data)

    try:
        logger.info(f"开始进行图生图 - {client_host}")
        result = await task_handler.img2img()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(500, detail='Result not found')

    return result


@app.get("/sdapi/v1/sd-models")
async def get_models(request: Request):

    task_list = []
    path = '/sdapi/v1/sd-models'

    task_handler = Task_Handler({}, request, path, reutrn_instance=True)
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


    @app.post("/tagger/v1/interrogate")
    async def _(request: TaggerRequest, api: Request):
        from utils.tagger import tagger_main, wd_logger
        data = request.model_dump()
        caption = await asyncio.get_event_loop().run_in_executor(
            None,
            tagger_main,
            data['image'],
            data['threshold'],
            wd_instance
        )
        wd_logger.info(f"打标成功,{caption}")
        return JSONResponse(caption)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    client_host = request.client.host
    backend_instance = Backend()

    if path == 'sdapi/v1/progress':
        return JSONResponse(backend_instance.format_progress_api_resp(0.0, time.time()))
    elif path == 'sdapi/v1/memory':
        return JSONResponse(backend_instance.format_vram_api_resp())
    elif path == 'sdapi/v1/options':
        return JSONResponse(backend_instance.format_options_api_resp())

    task_handler = Task_Handler({}, request, path)

    try:
        logger.info(f"开始进行转发 - {client_host}")
        result = await task_handler.sd_api()
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, detail=str(e))

    if result is None:
        raise HTTPException(500, detail='Result not found')

    return result


uvicorn.run(app, host=args.host, port=args.port, log_level='critical')
# uvicorn.run(app, host=args.host, port=args.port)


