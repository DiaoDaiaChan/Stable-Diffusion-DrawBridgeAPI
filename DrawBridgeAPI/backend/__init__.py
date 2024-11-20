import asyncio
import random
import json
import time

import aiofiles

from tqdm import tqdm
from pathlib import Path
from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Union
from colorama import Fore, Style
from colorama import init
init()

from ..base_config import setup_logger, init_instance
from .base import Backend

from DrawBridgeAPI.locales import _ as i18n

class BaseHandler:

    def __init__(
        self,
        payload,
        request: Request = None,
        path: str = None,
        comfyui_task=None,
    ):
        self.task_list = []
        self.instance_list: list[Backend] = []
        self.payload = payload
        self.request = request
        self.path = path
        self.config = init_instance.config
        self.all_task_list = None
        self.enable_backend: dict = {}
        self.comfyui_task: str = comfyui_task

    async def get_enable_task(
        self,
        task_type,
    ):
        """
        此函数的作用是获取示例并且只保留选择了的后端
        :param enable_task:
        :return:
        """
        enable_backend_list: dict[str, list[int]] = self.config.enable_backends.get(task_type, {})

        instance_list = []

        for enable_backend, backend_setting in enable_backend_list.items():

            def create_and_append_instances(
                    enable_backend_type,
                    AIDRAW_class,
                    backend_setting,
                    payload,
                    instance_list,
                    extra_args: dict=None
            ):

                enable_queue = False

                for counter in backend_setting:
                    if isinstance(counter, int):
                        enable_queue = False

                    elif isinstance(counter, dict):
                        operation = list(counter.values())[0]
                        counter = int(list(counter.keys())[0])
                        enable_queue = True if operation == "queue" else False

                    aidraw_instance = AIDRAW_class(
                        count=counter,
                        payload=payload,
                        enable_queue=enable_queue,
                        backend_type=enable_backend_type,
                        **(extra_args if extra_args else {})
                    )
                    self.enable_backend.update(
                        {
                            self.config.backends[enable_backend_type]["name"][counter]:
                            self.config.backends[enable_backend_type]["api"][counter]
                        }
                    )
                    aidraw_instance.init_backend_info()
                    instance_list.append(aidraw_instance)

            if "civitai" in enable_backend:
                from .SD_civitai_API import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "a1111webui" in enable_backend:
                from .SD_A1111_webui import AIDRAW

                create_and_append_instances(
                    enable_backend,
                    AIDRAW,
                    backend_setting,
                    self.payload,
                    instance_list,
                    extra_args={"request": self.request, "path": self.path}
                )

            elif "fal_ai" in enable_backend:
                from FLUX_falai import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "replicate" in enable_backend:
                from FLUX_replicate import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "liblibai" in enable_backend:
                from liblibai import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "tusiart" in enable_backend:
                from .tusiart import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "seaart" in enable_backend:
                from .seaart import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "yunjie" in enable_backend:
                from .yunjie import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "comfyui" in enable_backend:
                from .comfyui import AIDRAW

                create_and_append_instances(
                    enable_backend,
                    AIDRAW,
                    backend_setting,
                    self.payload,
                    instance_list,
                    extra_args={
                        "request": self.request,
                        "path": self.path
                    }
                )

            elif "novelai" in enable_backend:
                from .novelai import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

            elif "midjourney" in enable_backend:
                from .midjourney import AIDRAW
                create_and_append_instances(enable_backend, AIDRAW, backend_setting, self.payload, instance_list)

        self.instance_list = instance_list


class TXT2IMGHandler(BaseHandler):

    def __init__(self, payload=None, comfyui_task: str = None):
        super().__init__(comfyui_task=comfyui_task, payload=payload)

    async def get_all_instance(self) -> tuple[list[Backend], dict]:

        await self.get_enable_task("enable_txt2img_backends")
        return self.instance_list, self.enable_backend


class IMG2IMGHandler(BaseHandler):

    def __init__(self, payload=None, comfyui_task: str = None):
        super().__init__(comfyui_task=comfyui_task, payload=payload)

    async def get_all_instance(self) -> tuple[list[Backend], dict]:

        await self.get_enable_task("enable_img2img_backends")
        return self.instance_list, self.enable_backend


class A1111WebuiHandler(BaseHandler):

    async def get_all_instance(self) -> tuple[list[Backend], dict]:

        await self.get_enable_task("enable_a1111_backend")
        return self.instance_list, self.enable_backend


class A1111WebuiHandlerAPI(BaseHandler):
    async def get_all_instance(self) -> tuple[list[Backend], dict]:

        await self.get_enable_task("enable_sdapi_backends")
        return self.instance_list, self.enable_backend


class ComfyUIHandler(BaseHandler):

    async def get_all_instance(self) -> tuple[list[Backend], dict]:
        await self.get_enable_task("comfyui")
        return self.instance_list, self.enable_backend


class StaticHandler:
    lock_to_backend = None
    prompt_style: list = None

    @classmethod
    def set_lock_to_backend(cls, selected_model: str):
        cls.lock_to_backend = selected_model

    @classmethod
    def get_lock_to_backend(cls):
        return cls.lock_to_backend
    
    @classmethod
    def get_prompt_style(cls):
        return cls.prompt_style
    
    @classmethod
    def set_prompt_style(cls, prompt_style: list):
        cls.prompt_style = prompt_style

    @classmethod
    def get_backend_options(cls):
        build_resp = {
            "samples_save": True,
            "samples_format": "png",
            "samples_filename_pattern": "",
            "save_images_add_number": True,
            "grid_save": True,
            "grid_format": "png",
            "grid_extended_filename": False,
            "grid_only_if_multiple": True,
            "grid_prevent_empty_spots": False,
            "grid_zip_filename_pattern": "",
            "n_rows": -1.0,
            "font": "",
            "grid_text_active_color": "#000000",
            "grid_text_inactive_color": "#999999",
            "grid_background_color": "#ffffff",
            "enable_pnginfo": True,
            "save_txt": False,
            "save_images_before_face_restoration": False,
            "save_images_before_highres_fix": False,
            "save_images_before_color_correction": False,
            "save_mask": False,
            "save_mask_composite": False,
            "jpeg_quality": 80.0,
            "webp_lossless": False,
            "export_for_4chan": True,
            "img_downscale_threshold": 4.0,
            "target_side_length": 4000.0,
            "img_max_size_mp": 200.0,
            "use_original_name_batch": True,
            "use_upscaler_name_as_suffix": False,
            "save_selected_only": True,
            "save_init_img": False,
            "temp_dir": "",
            "clean_temp_dir_at_start": False,
            "save_incomplete_images": False,
            "outdir_samples": "",
            "outdir_txt2img_samples": "outputs/txt2img-images",
            "outdir_img2img_samples": "outputs/img2img-images",
            "outdir_extras_samples": "outputs/extras-images",
            "outdir_grids": "",
            "outdir_txt2img_grids": "outputs/txt2img-grids",
            "outdir_img2img_grids": "outputs/img2img-grids",
            "outdir_save": "log/images",
            "outdir_init_images": "outputs/init-images",
            "save_to_dirs": True,
            "grid_save_to_dirs": True,
            "use_save_to_dirs_for_ui": False,
            "directories_filename_pattern": "[date]",
            "directories_max_prompt_words": 8.0,
            "ESRGAN_tile": 192.0,
            "ESRGAN_tile_overlap": 8.0,
            "realesrgan_enabled_models": [
                "R-ESRGAN 4x+",
                "R-ESRGAN 4x+ Anime6B"
            ],
            "upscaler_for_img2img": None,
            "face_restoration": False,
            "face_restoration_model": "CodeFormer",
            "code_former_weight": 0.5,
            "face_restoration_unload": False,
            "auto_launch_browser": "Local",
            "show_warnings": False,
            "show_gradio_deprecation_warnings": True,
            "memmon_poll_rate": 8.0,
            "samples_log_stdout": False,
            "multiple_tqdm": True,
            "print_hypernet_extra": False,
            "list_hidden_files": True,
            "disable_mmap_load_safetensors": False,
            "hide_ldm_prints": True,
            "api_enable_requests": True,
            "api_forbid_local_requests": True,
            "api_useragent": "",
            "unload_models_when_training": False,
            "pin_memory": False,
            "save_optimizer_state": False,
            "save_training_settings_to_txt": True,
            "dataset_filename_word_regex": "",
            "dataset_filename_join_string": " ",
            "training_image_repeats_per_epoch": 1.0,
            "training_write_csv_every": 500.0,
            "training_xattention_optimizations": False,
            "training_enable_tensorboard": False,
            "training_tensorboard_save_images": False,
            "training_tensorboard_flush_every": 120.0,
            "sd_model_checkpoint": cls.lock_to_backend if cls.lock_to_backend else 'DrawBridgeAPI-Auto-Select',
            "sd_checkpoints_limit": 1.0,
            "sd_checkpoints_keep_in_cpu": True,
            "sd_checkpoint_cache": 3,
            "sd_unet": "None",
            "enable_quantization": False,
            "enable_emphasis": True,
            "enable_batch_seeds": True,
            "comma_padding_backtrack": 20.0,
            "CLIP_stop_at_last_layers": 3.0,
            "upcast_attn": False,
            "randn_source": "GPU",
            "tiling": False,
            "hires_fix_refiner_pass": "second pass",
            "sdxl_crop_top": 0.0,
            "sdxl_crop_left": 0.0,
            "sdxl_refiner_low_aesthetic_score": 2.5,
            "sdxl_refiner_high_aesthetic_score": 6.0,
            "sd_vae_explanation": "<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>\nimage into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling\n(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.\nFor img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling.",
            "sd_vae_checkpoint_cache": 0,
            "sd_vae": "None",
            "sd_vae_overrides_per_model_preferences": False,
            "auto_vae_precision": True,
            "sd_vae_encode_method": "Full",
            "sd_vae_decode_method": "Full",
            "inpainting_mask_weight": 1.0,
            "initial_noise_multiplier": 1.0,
            "img2img_extra_noise": 0,
            "img2img_color_correction": False,
            "img2img_fix_steps": False,
            "img2img_background_color": "#ffffff",
            "img2img_editor_height": 720.0,
            "img2img_sketch_default_brush_color": "#ffffff",
            "img2img_inpaint_mask_brush_color": "#ffffff",
            "img2img_inpaint_sketch_default_brush_color": "#ffffff",
            "return_mask": False,
            "return_mask_composite": False,
            "cross_attention_optimization": "Automatic",
            "s_min_uncond": 0.0,
            "token_merging_ratio": 0.0,
            "token_merging_ratio_img2img": 0.0,
            "token_merging_ratio_hr": 0.0,
            "pad_cond_uncond": False,
            "persistent_cond_cache": True,
            "batch_cond_uncond": True,
            "use_old_emphasis_implementation": False,
            "use_old_karras_scheduler_sigmas": False,
            "no_dpmpp_sde_batch_determinism": False,
            "use_old_hires_fix_width_height": False,
            "dont_fix_second_order_samplers_schedule": False,
            "hires_fix_use_firstpass_conds": False,
            "use_old_scheduling": False,
            "interrogate_keep_models_in_memory": False,
            "interrogate_return_ranks": False,
            "interrogate_clip_num_beams": 1.0,
            "interrogate_clip_min_length": 24.0,
            "interrogate_clip_max_length": 48.0,
            "interrogate_clip_dict_limit": 1500.0,
            "interrogate_clip_skip_categories": [],
            "interrogate_deepbooru_score_threshold": 0.5,
            "deepbooru_sort_alpha": True,
            "deepbooru_use_spaces": True,
            "deepbooru_escape": True,
            "deepbooru_filter_tags": "",
            "extra_networks_show_hidden_directories": True,
            "extra_networks_hidden_models": "When searched",
            "extra_networks_default_multiplier": 1.0,
            "extra_networks_card_width": 0,
            "extra_networks_card_height": 0,
            "extra_networks_card_text_scale": 1.0,
            "extra_networks_card_show_desc": True,
            "extra_networks_add_text_separator": " ",
            "ui_extra_networks_tab_reorder": "",
            "textual_inversion_print_at_load": False,
            "textual_inversion_add_hashes_to_infotext": True,
            "sd_hypernetwork": "None",
            "localization": "None",
            "gradio_theme": "Default",
            "gradio_themes_cache": True,
            "gallery_height": "",
            "return_grid": True,
            "do_not_show_images": False,
            "send_seed": True,
            "send_size": True,
            "js_modal_lightbox": True,
            "js_modal_lightbox_initially_zoomed": True,
            "js_modal_lightbox_gamepad": False,
            "js_modal_lightbox_gamepad_repeat": 250.0,
            "show_progress_in_title": True,
            "samplers_in_dropdown": True,
            "dimensions_and_batch_together": True,
            "keyedit_precision_attention": 0.1,
            "keyedit_precision_extra": 0.05,
            "keyedit_delimiters": ".,\\/!?%^*;:{}=`~()",
            "keyedit_move": True,
            "quicksettings_list": [
                "sd_model_checkpoint",
                "sd_unet",
                "sd_vae",
                "CLIP_stop_at_last_layers"
            ],
            "ui_tab_order": [],
            "hidden_tabs": [],
            "ui_reorder_list": [],
            "hires_fix_show_sampler": False,
            "hires_fix_show_prompts": False,
            "disable_token_counters": False,
            "add_model_hash_to_info": True,
            "add_model_name_to_info": True,
            "add_user_name_to_info": False,
            "add_version_to_infotext": True,
            "disable_weights_auto_swap": True,
            "infotext_styles": "Apply if any",
            "show_progressbar": True,
            "live_previews_enable": True,
            "live_previews_image_format": "png",
            "show_progress_grid": True,
            "show_progress_every_n_steps": 10.0,
            "show_progress_type": "Approx NN",
            "live_preview_allow_lowvram_full": False,
            "live_preview_content": "Prompt",
            "live_preview_refresh_period": 1000.0,
            "live_preview_fast_interrupt": False,
            "hide_samplers": [],
            "eta_ddim": 0.0,
            "eta_ancestral": 1.0,
            "ddim_discretize": "uniform",
            "s_churn": 0.0,
            "s_tmin": 0.0,
            "s_tmax": 0,
            "s_noise": 1.0,
            "k_sched_type": "Automatic",
            "sigma_min": 0.0,
            "sigma_max": 0.0,
            "rho": 0.0,
            "eta_noise_seed_delta": 0,
            "always_discard_next_to_last_sigma": False,
            "sgm_noise_multiplier": False,
            "uni_pc_variant": "bh1",
            "uni_pc_skip_type": "time_uniform",
            "uni_pc_order": 3.0,
            "uni_pc_lower_order_final": True,
            "postprocessing_enable_in_main_ui": [],
            "postprocessing_operation_order": [],
            "upscaling_max_images_in_cache": 5.0,
            "disabled_extensions": [],
            "disable_all_extensions": "none",
            "restore_config_state_file": "",
            "sd_checkpoint_hash": "91e0f7cbaf70676153810c231e8703bf26b3208c116a3d1f2481cbc666905471"
        }

        return build_resp


class TaskHandler(StaticHandler):

    backend_avg_dict: dict = {}
    write_count: dict = {}
    backend_images: dict = {}

    backend_site_list = None
    load_balance_logger = setup_logger('[AvgTimeCalculator]')
    load_balance_sample = 10

    redis_client = None
    backend_status = None

    @classmethod
    def update_backend_status(cls):
        cls.backend_status = json.loads(cls.redis_client.get("workload"))

    @classmethod
    def get_redis_client(cls):
        cls.redis_client = init_instance.redis_client

    @classmethod
    async def get_backend_avg_work_time(cls) -> dict:
        backend_sites = cls.backend_site_list

        avg_time_key = ""

        avg_time_data = cls.redis_client.get("backend_avg_time")
        if avg_time_data is None:
            cls.redis_client.set(avg_time_key, json.dumps(cls.backend_avg_dict))
        else:
            new_data = json.loads(avg_time_data)
            for key, values in new_data.items():
                if key in cls.backend_avg_dict:
                    cls.backend_avg_dict[key].extend(
                        values[-cls.load_balance_sample:] if len(values) >= cls.load_balance_sample else
                        values
                    )
                else:
                    cls.backend_avg_dict[key] = (values[-cls.load_balance_sample:] if
                                                 len(values) >= cls.load_balance_sample else values)

                cls.backend_avg_dict[key] = cls.backend_avg_dict[key][-10:]

        avg_time_dict = {}
        for backend_site in backend_sites:
            spend_time_list = cls.backend_avg_dict.get(backend_site, [])
            if spend_time_list and len(spend_time_list) >= cls.load_balance_sample:
                sorted_list = sorted(spend_time_list)
                trimmed_list = sorted_list[1:-1]
                avg_time = sum(trimmed_list) / len(trimmed_list) if trimmed_list else None
                avg_time_dict[backend_site] = avg_time
            else:
                avg_time_dict[backend_site] = None

        return avg_time_dict

    @classmethod
    async def set_backend_work_time(cls, spend_time, backend_site, total_images=1):
        spend_time_list = cls.backend_avg_dict.get(backend_site, [])
        spend_time_list.append(int(spend_time/total_images))

        if len(spend_time_list) >= cls.load_balance_sample:
            spend_time_list = spend_time_list[-cls.load_balance_sample:]

        cls.backend_avg_dict[backend_site] = spend_time_list

        cls.write_count[backend_site] = cls.write_count.get(backend_site, 0) + 1

        if cls.write_count.get(backend_site, 0) >= cls.load_balance_sample:
            cls.redis_client.set("backend_avg_time", json.dumps(cls.backend_avg_dict))
            cls.write_count[backend_site] = 0

        # info_str = ''

        # for key, values in cls.backend_avg_dict.items():
        #     info_str += f"{key}: 最近10次生成时间{values}\n"
        #
        # cls.load_balance_logger.info(info_str)

    @classmethod
    def set_backend_image(cls, num=0, backend_site=None, get=False) -> Union[None, dict]:
        all_backend_dict = {}

        if backend_site:
            working_images = cls.backend_images.get(backend_site, 1)
            working_images += num
            cls.backend_images[backend_site] = working_images

        if get:
            for site in cls.backend_site_list:
                all_backend_dict[site] = cls.backend_images.get(site, 1)
            return all_backend_dict

    @classmethod
    def set_backend_list(cls, backend_dict):
        cls.backend_site_list = list(backend_dict.values())

    def __init__(
        self,
        payload=None,
        request: Request = None,
        path: str = None,
        select_backend: int = None,
        reutrn_instance: bool = False,
        model_to_backend: str = None,
        disable_loadbalance: bool = False,
        comfyui_json: str = "",
        override_model_select: bool = False,
    ):
        self.payload = payload
        self.instance_list = []
        self.result = None
        self.request = request
        self.path = path
        self.enable_backend = None
        self.reutrn_instance = reutrn_instance
        self.select_backend = select_backend
        self.model_to_backend = model_to_backend  # 模型的名称
        self.disable_loadbalance = disable_loadbalance
        self.lock_to_backend = self.get_lock_to_backend() if override_model_select is False else None
        self.comfyui_json: str = comfyui_json

        self.total_images = (self.payload.get("batch_size", 1) * self.payload.get("n_iter", 1)) or 1

        self.ava_backend_url = None
        self.ava_backend_index = None

    @staticmethod
    def get_backend_name(model_name) -> str:
        all_model: bytes = init_instance.redis_client.get('models')
        all_model: dict = json.loads(all_model.decode('utf-8'))
        for key, models in all_model.items():
            if isinstance(models, list):
                for model in models:
                    if model.get("title") == model_name or model.get("model_name") == model_name:
                        return key

    @staticmethod
    def get_backend_index(mapping_dict, key_to_find) -> int:
        keys = list(mapping_dict.keys())
        if key_to_find in keys:
            return keys.index(key_to_find)
        return None

    async def txt2img(self):

        self.instance_list, self.enable_backend = await TXT2IMGHandler(
            self.payload,
            comfyui_task=self.comfyui_json
        ).get_all_instance()

        await self.choice_backend()
        return self.result

    async def img2img(self):

        self.instance_list, self.enable_backend = await IMG2IMGHandler(
            self.payload,
            comfyui_task=self.comfyui_json
        ).get_all_instance()

        await self.choice_backend()
        return self.result

    async def sd_api(self) -> JSONResponse or list[Backend]:

        self.instance_list, self.enable_backend = await A1111WebuiHandlerAPI(
            self.payload,
            self.request,
            self.path
        ).get_all_instance()

        await self.choice_backend()
        return self.result

    async def comfyui_api(self) -> JSONResponse or list[Backend]:

        self.instance_list, self.enable_backend = await ComfyUIHandler(
            self.payload,
            self.request,
            self.path
        ).get_all_instance()

        await self.choice_backend()
        return self.result

    async def choice_backend(self):

        from DrawBridgeAPI.locales import _ as i18n

        if self.disable_loadbalance:
            return
        backend_url_dict = self.enable_backend
        self.set_backend_list(backend_url_dict)
        self.get_redis_client()
        reverse_dict = {value: key for key, value in backend_url_dict.items()}

        tasks = []
        is_avaiable = 0
        status_dict = {}
        ava_url = None
        n = -1
        e = -1
        normal_backend = []
        idle_backend = []

        logger = setup_logger(custom_prefix='[LOAD_BALANCE]')

        if self.reutrn_instance:
            self.result = self.instance_list
            return
        for i in self.instance_list:
            task = i.get_backend_working_progress()
            tasks.append(task)
        # 获取api队列状态
        key = self.get_backend_name(self.model_to_backend or self.lock_to_backend)
        if self.model_to_backend and key is not None:

            backend_index = self.get_backend_index(backend_url_dict, key)
            logger.info(f"{i18n('Manually select model')}: {self.model_to_backend}, {i18n('Backend select')}{key[:24]}")

            self.ava_backend_url = backend_url_dict[key]
            self.ava_backend_index = backend_index

            await self.exec_generate()

        elif self.lock_to_backend:
            if self.lock_to_backend and key is not None:
                backend_index = self.get_backend_index(backend_url_dict, key)
                logger.info(f"{i18n('Backend locked')}: {key[:24]}")

                self.ava_backend_url = backend_url_dict[key]
                self.ava_backend_index = backend_index

                await self.exec_generate()

        else:
            all_resp = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(i18n('Starting backend selection'))
            for resp_tuple in all_resp:
                e += 1
                if isinstance(resp_tuple, None or Exception):
                    logger.warning(i18n('Backend %s is down') % self.instance_list[e].workload_name[:24])
                else:
                    try:
                        if resp_tuple[3] in [200, 201]:
                            n += 1
                            status_dict[resp_tuple[2]] = resp_tuple[0]["eta_relative"]
                            normal_backend = (list(status_dict.keys()))
                        else:
                            raise RuntimeError
                    except RuntimeError or TypeError:
                        logger.warning(i18n('Backend %s is failed or locked') % self.instance_list[e].workload_name[:24])
                        continue
                    else:
                        # 更改判断逻辑
                        if resp_tuple[0]["progress"] in [0, 0.0]:
                            is_avaiable += 1
                            idle_backend.append(normal_backend[n])
                        else:
                            pass
                    # 显示进度
                    total = 100
                    progress = int(resp_tuple[0]["progress"] * 100)
                    show_str = f"{list(backend_url_dict.keys())[e][:24]}"
                    show_str = show_str.ljust(50, "-")

                    bar_format = f"{Fore.CYAN}[Progress] {{l_bar}}{{bar}}|{Style.RESET_ALL}"

                    with tqdm(
                            total=total,
                            desc=show_str + "-->",
                            bar_format=bar_format
                    ) as pbar:
                        pbar.update(progress)
            if len(normal_backend) == 0:
                logger.error(i18n('No available backend'))
                raise RuntimeError(i18n('No available backend'))

            backend_total_work_time = {}
            avg_time_dict = await self.get_backend_avg_work_time()
            backend_image = self.set_backend_image(get=True)

            eta = 0

            for (site, time_), (_, image_count) in zip(avg_time_dict.items(), backend_image.items()):
                self.load_balance_logger.info(
                    i18n('Backend: %s Average work time: %s seconds, Current tasks: %s') % (site, time_, image_count - 1)
                )
                if site in normal_backend:
                    self.update_backend_status()
                    for key in self.backend_status:
                        if site in key:
                            end_time = self.backend_status[key].get('end_time', None)
                            start_time = self.backend_status[key].get('start_time', None)
                            if start_time:
                                if end_time:
                                    eta = 0
                                else:
                                    current_time = time.time()
                                    eta = int(current_time - start_time)

                    effective_time = 1 if time_ is None else time_
                    total_work_time = effective_time * int(image_count)

                    eta = eta if time_ else 0
                    self.load_balance_logger.info(f"{i18n('Extra time weight')}{eta}")

                    backend_total_work_time[site] = total_work_time - eta if (total_work_time - eta) >= 0 else total_work_time

            total_time_dict = list(backend_total_work_time.values())
            rev_dict = {}
            for key, value in backend_total_work_time.items():
                if value in rev_dict:
                    rev_dict[(value, key)] = value
                else:
                    rev_dict[value] = key

            sorted_list = sorted(total_time_dict)
            fastest_backend = sorted_list[0]
            ava_url = rev_dict[fastest_backend]
            self.load_balance_logger.info(i18n('Backend %s is the fastest, has been selected') % ava_url[:24])
            ava_url_index = list(backend_url_dict.values()).index(ava_url)

            self.ava_backend_url = ava_url
            self.ava_backend_index = ava_url_index

            await self.exec_generate()
            # ava_url_tuple = (ava_url, reverse_dict[ava_url], all_resp, len(normal_backend), vram_dict[ava_url])

    async def exec_generate(self):
        self.set_backend_image(self.total_images, self.ava_backend_url)
        fifo = None
        try:
            fifo = await self.instance_list[self.ava_backend_index].send_result_to_api()
        except:
            pass
        finally:
            self.set_backend_image(-self.total_images, self.ava_backend_url)
            self.result = fifo.result if fifo is not None else None
            await self.set_backend_work_time(fifo.spend_time, self.ava_backend_url, fifo.total_img_count)




