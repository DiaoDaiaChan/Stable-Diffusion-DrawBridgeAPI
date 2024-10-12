import yaml as yaml_
import shutil
import redis
import json
import logging
import os
import traceback
import sys

import pydantic
from packaging import version

pyd_version = pydantic.__version__

if version.parse(pyd_version) < version.parse("2.0"):
    from pydantic import BaseSettings
else:
    try:
        from pydantic_settings import BaseSettings
    except:
        traceback.print_exc()
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pydantic_settings"])
        print("请重启 / please restart")

        sys.exit()

from pathlib import Path

redis_client = None

api_current_dir = os.path.dirname(os.path.abspath(__file__))


class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', prefix="[MAIN]"):
        super().__init__(fmt, datefmt, style)
        self.prefix = prefix

    def format(self, record):
        original_msg = record.msg
        record.msg = f"{self.prefix} {original_msg}"
        formatted_msg = super().format(record)
        record.msg = original_msg  # 恢复原始消息
        return formatted_msg


# 字典用于跟踪已创建的日志记录器

empty_dict = {"token": None}

import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter to add a fixed color for the prefix and variable colors for the log levels."""

    def __init__(self, prefix="", img_prefix="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = f"\033[94m{prefix}\033[0m"  # 固定蓝色前缀
        self.img_prefix = f"\033[93m{img_prefix}\033[0m"  # 固定黄色前缀
        self.FORMATS = {
            logging.DEBUG: f"{self.prefix} \033[94m[DEBUG]\033[0m %(message)s",
            logging.INFO: f"{self.prefix} \033[92m[INFO]\033[0m %(message)s",
            logging.WARNING: f"{self.prefix} \033[93m[WARNING]\033[0m %(message)s",
            logging.ERROR: f"{self.prefix} \033[91m[ERROR]\033[0m %(message)s",
            logging.CRITICAL: f"{self.prefix} \033[95m[CRITICAL]\033[0m %(message)s",
            "IMG": f"{self.img_prefix} \033[93m[IMG]\033[0m %(message)s"  # 黄色前缀的 IMG 日志
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS.get("IMG"))
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger(logging.Logger):
    """Custom logger class to add an img method."""

    def __init__(self, name, level=logging.DEBUG):
        super().__init__(name, level)
        self.img_level = 25  # 自定义日志等级
        logging.addLevelName(self.img_level, "IMG")

    def img(self, msg, *args, **kwargs):
        if self.isEnabledFor(self.img_level):
            self._log(self.img_level, msg, args, **kwargs)


loggers = {}


def setup_logger(custom_prefix="[MAIN]"):
    # 检查是否已经存在具有相同前缀的 logger
    if custom_prefix in loggers:
        return loggers[custom_prefix]

    # 使用自定义的 Logger 类
    logger = CustomLogger(custom_prefix)
    logger.setLevel(logging.DEBUG)

    # 创建一个控制台处理器并设置日志级别为DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建一个文件处理器来保存所有日志到 log.txt
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)

    # 创建一个错误文件处理器来保存错误日志到 log_error.txt
    error_file_handler = logging.FileHandler('log_error.log')
    error_file_handler.setLevel(logging.ERROR)

    # 创建一个文件处理器来保存IMG日志到 log_img.log
    img_file_handler = logging.FileHandler('log_img.log')
    img_file_handler.setLevel(logger.img_level)

    # 创建格式器并将其添加到处理器
    formatter = CustomFormatter(prefix=custom_prefix, img_prefix=custom_prefix)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    error_file_handler.setFormatter(formatter)
    img_file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(img_file_handler)

    # 将创建的 logger 存储在字典中
    loggers[custom_prefix] = logger

    return logger


class Config(BaseSettings):
    backend_name_list: list = []

    server_settings: dict = None

    civitai_setting: dict = empty_dict
    a1111webui_setting: dict = {"backend_url": None}
    fal_ai_setting: dict = empty_dict
    replicate_setting: dict = empty_dict
    liblibai_setting: dict = empty_dict
    tusiart_setting: dict = empty_dict
    seaart_setting: dict = empty_dict
    yunjie_setting: dict = empty_dict
    comfyui_setting: dict = empty_dict
    novelai_setting: dict = empty_dict

    civitai: list or None = []
    a1111webui: list = []
    fal_ai: list = []
    replicate: list = []
    liblibai: list = []
    tusiart: list = []
    seaart: list = []
    yunjie: list = []
    comfyui: list = []
    novelai: list = []

    civitai_name: dict = {}
    a1111webui_name: dict = {}
    fal_ai_name: dict = {}
    replicate_name: dict = {}
    liblibai_name: dict = {}
    tusiart_name: dict = {}
    seaart_name: dict = {}
    yunjie_name: dict = {}
    comfyui_name: dict = {}
    novelai_name: dict = {}

    server_settings: dict = {}
    retry_times: int = 3
    proxy: str = ''

    workload_dict: dict = {}

    base_workload_dict: dict = {
        "start_time": None,
        "idle": True,
        "available": True,
        "fault": False
    }

    models_list: list = []

    name_url: dict = {}


def package_import(copy_to_config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_template = Path(os.path.join(current_dir, "config_example.yaml")).resolve()
    shutil.copy(source_template, copy_to_config_path)


class ConfigInit:

    def __init__(self):
        self.config = None
        self.config_file_path = None
        self.logger = setup_logger(custom_prefix="[INIT]")
        self.redis_client = None

    def load_config(self):

        with open(self.config_file_path, "r", encoding="utf-8") as f:
            yaml_config = yaml_.load(f, Loader=yaml_.FullLoader)
            config = Config(**yaml_config)
            self.logger.info('加载配置文件完成')

            return config

    def init(self, config_file_path):

        self.config_file_path = config_file_path
        config = self.load_config()

        welcome_txt = '''
欢迎使用 
_____                              ____           _       _                               _____    _____ 
|  __ \                            |  _ \         (_)     | |                      /\     |  __ \  |_   _|
| |  | |  _ __    __ _  __      __ | |_) |  _ __   _    __| |   __ _    ___       /  \    | |__) |   | |  
| |  | | | '__|  / _` | \ \ /\ / / |  _ <  | '__| | |  / _` |  / _` |  / _ \     / /\ \   |  ___/    | |  
| |__| | | |    | (_| |  \ V  V /  | |_) | | |    | | | (_| | | (_| | |  __/    / ____ \  | |       _| |_ 
|_____/  |_|     \__,_|   \_/\_/   |____/  |_|    |_|  \__,_|  \__, |  \___|   /_/    \_\ |_|      |_____|
                                                                __/ |                                     
                                                                |___/
关注雕雕, 关注雕雕喵
项目地址: https://github.com/DiaoDaiaChan/Stable-Diffusion-DrawBridgeAPI                                                                                                      
        '''

        print(welcome_txt)

        config.civitai = config.civitai_setting['token']
        config.a1111webui = config.a1111webui_setting
        config.fal_ai = config.fal_ai_setting['token']
        config.replicate = config.replicate_setting['token']
        config.liblibai = config.liblibai_setting['token']
        config.tusiart = config.tusiart_setting['token']
        config.seaart = config.seaart_setting['token']
        config.yunjie = config.yunjie_setting['token']
        config.comfyui = config.comfyui_setting
        config.novelai = config.novelai_setting['token']

        sources_list = [
            (config.civitai, 0, config.civitai_name),
            (config.a1111webui, 1, config.a1111webui_name),
            (config.fal_ai, 2, config.fal_ai_name),
            (config.replicate, 3, config.replicate_name),
            (config.liblibai, 4, config.liblibai_name),
            (config.tusiart, 5, config.tusiart_name),
            (config.seaart, 6, config.seaart_name),
            (config.yunjie, 7, config.yunjie_name),
            (config.comfyui, 8, config.comfyui_name),
            (config.novelai, 9, config.novelai_name),
        ]

        def process_items(config, items, backend_index, name_dict):
            if backend_index == 1:  # 特殊处理 config.a1111webui
                for i in range(len(items['name'])):
                    key = f"{config.backend_name_list[backend_index]}-{items['name'][i]}"
                    config.workload_dict[key] = config.base_workload_dict
                    name_dict[f"a1111-{items['name'][i]}"] = items['backend_url'][i]
            elif backend_index == 8:
                for i in range(len(items['name'])):
                    key = f"{config.backend_name_list[backend_index]}-{items['name'][i]}"
                    config.workload_dict[key] = config.base_workload_dict
                    name_dict[f"comfyui-{items['name'][i]}"] = items['backend_url'][i]
            else:
                for n in items:
                    key = f"{config.backend_name_list[backend_index]}-{n}"
                    config.workload_dict[key] = config.base_workload_dict
                    name_dict[key] = n

        for items, backend_index, name_dict in sources_list:
            process_items(config, items, backend_index, name_dict)

        def merge_and_count(*args):
            merged_dict = {}
            lengths = []
            for arg in args:
                merged_dict |= arg[2]
                lengths.append(len(arg[0]))
            return merged_dict, tuple(lengths)

        config.name_url = merge_and_count(*sources_list)

        models_dict = {}
        models_dict['is_loaded'] = False
        for back_name in list(config.workload_dict.keys()):
            models_dict[back_name] = config.models_list

        self.redis_client = redis.Redis(
            host=config.server_settings['redis_server'][0],
            port=config.server_settings['redis_server'][1],
            password=config.server_settings['redis_server'][2],
            db=config.server_settings['redis_server'][3]
        )

        self.logger.info('redis连接成功')

        workload_json = json.dumps(config.workload_dict)

        rp = self.redis_client.pipeline()
        rp.set('workload', workload_json)
        rp.set('models', json.dumps(models_dict))
        rp.execute()

        self.config = config


init_instance = ConfigInit()
