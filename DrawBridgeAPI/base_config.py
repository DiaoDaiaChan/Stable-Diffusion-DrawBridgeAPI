import yaml as yaml_
import shutil
import redis
import json
import logging
import os

from pydantic import BaseModel
from typing import Dict, List
from pathlib import Path

from .locales import _


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


class Config(BaseModel):

    backend_name_list: list = []

    server_settings: dict = {}
    enable_backends: Dict[str, Dict[str, List[int | Dict]]] = {}
    backends: dict = {}

    retry_times: int = 3
    proxy: str = ''

    workload_dict: dict = {}

    base_workload_dict: dict = {
        "start_time": None,
        "end_time": None,
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
            self.logger.info(_('Loading config file completed'))

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
项目地址/Project Re: https://github.com/DiaoDaiaChan/Stable-Diffusion-DrawBridgeAPI                                                                                                      
        '''

        print(welcome_txt)

        for backend_type, api in config.backends.items():
            if api:
                for name in api['name']:
                    key = f"{backend_type}-{name}"
                    config.workload_dict[key] = config.base_workload_dict

        models_dict = {}
        models_dict['is_loaded'] = False
        for back_name in list(config.workload_dict.keys()):
            models_dict[back_name] = config.models_list

        try:
            db_index = config.server_settings['redis_server'][3]
        except IndexError:
            db_index = 15

        self.redis_client = redis.Redis(
            host=config.server_settings['redis_server'][0],
            port=config.server_settings['redis_server'][1],
            password=config.server_settings['redis_server'][2],
            db=db_index
        )

        self.logger.info(_('Redis connection successful'))

        workload_json = json.dumps(config.workload_dict)

        rp = self.redis_client.pipeline()
        rp.set('workload', workload_json)
        rp.set('models', json.dumps(models_dict))
        rp.set('styles', json.dumps([]))
        rp.execute()

        self.config = config


init_instance = ConfigInit()
