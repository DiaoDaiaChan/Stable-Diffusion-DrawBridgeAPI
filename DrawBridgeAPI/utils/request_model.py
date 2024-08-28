from pydantic import BaseModel
from dataclasses import field
from typing import Optional, List, Dict, Any
import random


class RequetModelClass(BaseModel):
    pass

class Txt2ImgRequest(RequetModelClass):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    styles: List[str] = []
    seed: int = random.randint(0, 4294967295)
    subseed: int = random.randint(0, 4294967295)
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    sampler_name: str = "euler"
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    tiling: bool = False
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
    override_settings_restore_afterwards: bool = False
    refiner_checkpoint: str = ""
    refiner_switch_at: int = 0
    disable_extra_networks: bool = False
    comments: Dict[str, Any] = {}
    enable_hr: bool = False
    firstphase_width: int = 0
    firstphase_height: int = 0
    hr_scale: float = 2
    hr_upscaler: str = ""
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_checkpoint_name: str = ""
    hr_sampler_name: str = ""
    hr_prompt: str = ""
    hr_negative_prompt: str = ""
    sampler_index: str = "euler"
    script_name: str = ""
    script_args: List[Any] = []
    send_images: bool = True
    save_images: bool = True
    alwayson_scripts: Dict[str, Any] = {}


class Img2ImgRequest(RequetModelClass):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    styles: List[str] = []
    seed: int = random.randint(0, 4294967295)
    subseed: int = random.randint(0, 4294967295)
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    sampler_name: str = "euler"
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    tiling: bool = False
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
    override_settings_restore_afterwards: bool = False
    refiner_checkpoint: str = ""
    refiner_switch_at: int = 0
    disable_extra_networks: bool = False
    comments: Dict[str, Any] = {}
    init_images: List[str] = [""]
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
    latent_mask: str = ""
    sampler_index: str = "euler"
    include_init_images: bool = False
    script_name: str = ""
    script_args: List[Any] = []
    send_images: bool = True
    save_images: bool = True
    alwayson_scripts: Dict[str, Any] = {}
    # 以下为拓展


class TaggerRequest(RequetModelClass):
    image: str = '',
    model: str = 'wd14-vit-v2'
    threshold: float = 0.35