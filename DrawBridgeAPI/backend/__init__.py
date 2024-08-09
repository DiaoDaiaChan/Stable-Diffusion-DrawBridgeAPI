import asyncio
import random
import json

from tqdm import tqdm
from fastapi import Request
from fastapi.responses import JSONResponse

from base_config import setup_logger, config, redis_client
from .SD_civitai_API import AIDRAW
from .SD_A1111_webui import AIDRAW as AIDRAW2
from .FLUX_falai import AIDRAW as AIDRAW3
from .FLUX_replicate import AIDRAW as AIDRAW4
from .liblibai import AIDRAW as AIDRAW5
from .tusiart import AIDRAW as AIDRAW6
from .base import Backend


class Base_Handler:

    def __init__(
        self,
        payload,
        request: Request = None,
        path: str = None
    ):
        self.task_list = []
        self.instance_list: list[Backend] = []
        self.payload = payload
        self.request = request
        self.path = path
        self.config = config
        self.all_task_list = list(range(len(list(config.name_url[0].keys()))))
        self.enable_backend: dict = {}

    async def get_enable_task(
        self,
        enable_task
    ):
        """
        此函数的作用是获取示例并且只保留选择了的后端
        :param enable_task:
        :return:
        """
        tasks = [
            self.get_civitai_task(),
            self.get_a1111_task(),
            self.get_falai_task(),
            self.get_replicate_task(),
            self.get_liblibai_task(),
            self.get_tusiart_task()
        ]

        all_backend_instance = await asyncio.gather(*tasks)
        all_backend_instance_list = [item for sublist in all_backend_instance for item in sublist]

        # 获取启动的后端字典
        all_backend_dict: dict = self.config.name_url[0]
        items = list(all_backend_dict.items())
        self.enable_backend = dict([items[i] for i in enable_task])

        self.instance_list = [all_backend_instance_list[i] for i in enable_task]

    async def get_civitai_task(self):
        instance_list = []
        counter = 0
        for i in config.civitai:
            if i is not None:
                aidraw_instance = AIDRAW(count=counter, payload=self.payload)
                counter += 1
                instance_list.append(aidraw_instance)

        return instance_list

    async def get_a1111_task(self):

        instance_list = []
        counter = 0
        for i in config.a1111webui['name']:
            aidraw_instance = AIDRAW2(
                count=counter,
                payload=self.payload,
                request=self.request,
                path=self.path
            )
            counter += 1
            instance_list.append(aidraw_instance)

        return instance_list

    async def get_falai_task(self):

        instance_list = []
        counter = 0
        for i in config.fal_ai:
            if i is not None:
                aidraw_instance = AIDRAW3(count=counter, payload=self.payload)
                counter += 1
                instance_list.append(aidraw_instance)

        return instance_list

    async def get_replicate_task(self):

        instance_list = []
        counter = 0
        for i in config.replicate:
            if i is not None:
                aidraw_instance = AIDRAW4(count=counter, payload=self.payload)
                counter += 1
                instance_list.append(aidraw_instance)

        return instance_list

    async def get_liblibai_task(self):
        instance_list = []
        counter = 0
        for i in config.liblibai:
            if i is not None:
                aidraw_instance = AIDRAW5(count=counter, payload=self.payload)
                counter += 1
                instance_list.append(aidraw_instance)

        return instance_list

    async def get_tusiart_task(self):
        instance_list = []
        counter = 0
        for i in config.tusiart:
            if i is not None:
                aidraw_instance = AIDRAW6(count=counter, payload=self.payload)
                counter += 1
                instance_list.append(aidraw_instance)

        return instance_list


class TXT2IMG_Handler(Base_Handler):

    async def get_all_instance(self) -> tuple[list[Backend], dict]:
        # 手动选择启动的后端
        man_enable_task = self.config.server_settings['enable_txt2img_backends']
        if len(man_enable_task) != 0:
            man_enable_task = man_enable_task
        else:
            man_enable_task = self.all_task_list

        await self.get_enable_task(man_enable_task)

        return self.instance_list, self.enable_backend


class IMG2IMG2_Handler(Base_Handler):

    async def get_all_instance(self) -> tuple[list[Backend], dict]:
        # 手动选择启动的后端
        man_enable_task = self.config.server_settings['enable_img2img_backends']
        if len(man_enable_task) != 0:
            man_enable_task = man_enable_task
        else:
            man_enable_task = self.all_task_list

        await self.get_enable_task(man_enable_task)

        return self.instance_list, self.enable_backend


class A1111Webui_Handler(Base_Handler):

    async def get_all_instance(self) -> tuple[list[Backend], dict]:

        await self.get_enable_task([1])

        return self.instance_list, self.enable_backend


class Task_Handler:

    def __init__(
        self,
        payload,
        request: Request = None,
        path: str = None,
        select_backend: int = None,
        reutrn_instance: bool = False,
        model_to_backend: str = None,
        disable_loadbalance: bool = False
    ):
        self.payload = payload
        self.instance_list = []
        self.result = None
        self.request = request
        self.path = path
        self.enable_backend = None
        self.reutrn_instance = reutrn_instance
        self.select_backend = select_backend
        self.model_to_backend = model_to_backend
        self.disable_loadbalance = disable_loadbalance

    def get_backend_name(self, model_name) -> str:
        all_model: bytes = redis_client.get('models')
        all_model: dict = json.loads(all_model.decode('utf-8'))
        for key, models in all_model.items():
            if isinstance(models, list):
                for model in models:
                    if model.get("title") == model_name:
                        return key

    def get_backend_index(self, mapping_dict, key_to_find) -> int:
        keys = list(mapping_dict.keys())
        if key_to_find in keys:
            return keys.index(key_to_find)
        return None


    async def txt2img(self):
        self.instance_list, self.enable_backend = await TXT2IMG_Handler(self.payload).get_all_instance()
        await self.choice_backend()
        return self.result

    async def img2img(self):
        self.instance_list, self.enable_backend = await IMG2IMG2_Handler(self.payload).get_all_instance()
        await self.choice_backend()
        return self.result

    async def sd_api(self) -> JSONResponse or list[Backend]:

        self.instance_list, self.enable_backend = await A1111Webui_Handler(
            self.payload,
            self.request,
            self.path
        ).get_all_instance()

        await self.choice_backend()
        return self.result

    async def choice_backend(self):
        if self.disable_loadbalance:
            return
        backend_url_dict = self.enable_backend
        reverse_dict = {value: key for key, value in backend_url_dict.items()}

        tasks = []
        is_avaiable = 0
        status_dict = {}
        vram_dict = {}
        ava_url = None
        n = -1
        e = -1
        defult_eta = 20
        normal_backend = None
        idle_backend = []
        slice = [24]

        logger = setup_logger(custom_prefix='[LOAD_BALANCE]')
        if self.reutrn_instance:
            self.result = self.instance_list
            return
        for i in self.instance_list:
            task = i.get_backend_working_progress()
            tasks.append(task)
        # 获取api队列状态
        if self.model_to_backend:

            key = self.get_backend_name(self.model_to_backend)
            backend_index = self.get_backend_index(backend_url_dict, key)
            logger.info(f"手动选择模型{self.model_to_backend}, 已选择后端{key[:24]}")
            self.result = await self.instance_list[backend_index].send_result_to_api()

        else:
            all_resp = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info('开始进行后端选择')
            for resp_tuple in all_resp:
                e += 1
                if isinstance(resp_tuple, None or Exception):
                    logger.warning(f"后端{self.instance_list[e].workload_name[:24]}掉线")
                else:
                    try:
                        if resp_tuple[3] in [200, 201]:
                            n += 1
                            status_dict[resp_tuple[2]] = resp_tuple[0]["eta_relative"]
                            normal_backend = (list(status_dict.keys()))
                        else:
                            raise RuntimeError
                    except RuntimeError or TypeError:
                        logger.warning(f"后端{self.instance_list[e].backend_name[:24]}出错或者锁定中")
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
                    with tqdm(
                            total=total,
                            desc=show_str + "-->",
                            bar_format="{l_bar}{bar}|"
                    ) as pbar:
                        pbar.update(progress)

            if is_avaiable == 0:
                n = -1
                y = 0
                normal_backend = list(status_dict.keys())
                logger.info("没有空闲后端")
                if len(normal_backend) == 0:
                    raise RuntimeError("没有可用后端")
                else:
                    eta_list = list(status_dict.values())
                    for t, b in zip(eta_list, normal_backend):
                        if int(t) < defult_eta:
                            y += 1
                            ava_url = b
                            logger.info(f"已选择后端{reverse_dict[ava_url][:24]}")
                            break
                        else:
                            y += 0
                    if y == 0:
                        reverse_sta_dict = {value: key for key, value in status_dict.items()}
                        eta_list.sort()
                        ava_url = reverse_sta_dict[eta_list[0]]

            if len(idle_backend) >= 1:
                ava_url = random.choice(idle_backend)

            logger.info(f"已选择后端{reverse_dict[ava_url][:24]}")
            ava_url_index = list(backend_url_dict.values()).index(ava_url)
            # ava_url_tuple = (ava_url, reverse_dict[ava_url], all_resp, len(normal_backend), vram_dict[ava_url])
            self.result = await self.instance_list[ava_url_index].send_result_to_api()

