import aiohttp

from urllib.parse import urlencode

from .base import Backend, http_request

import traceback


class AIDRAW(Backend):

    def __init__(self, count, payload, **kwargs):
        super().__init__(count=count, payload=payload, **kwargs)

        self.model = "StableDiffusion"
        self.model_hash = "c7352c5d2f"
        self.logger = self.setup_logger('[SD-A1111]')
        self.current_config: dict = self.config.a1111webui_setting

        self.backend_url = self.current_config['backend_url'][self.count]
        name = self.current_config['name'][self.count]
        self.backend_name = self.config.backend_name_list[1]
        self.workload_name = f"{self.backend_name}-{name}"

    async def exec_login(self):

        login_data = {
            'username': self.current_config['username'][self.count],
            'password': self.current_config['password'][self.count]
        }
        encoded_data = urlencode(login_data)
        async with aiohttp.ClientSession(headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "accept": "application/json"}
        ) as session:
            async with session.post(
                    url=f"{self.backend_url}/login",
                    data=encoded_data
            ) as resp:
                resp_code = resp.status
                if resp_code != 200:
                    self.logger.info(f"后端{self.backend_name}登录失败")
                    self.fail_on_login = True
                    return False, resp_code
                else:
                    self.logger.info(f"后端{self.backend_name}登录成功")
                    return True, resp_code


    async def check_backend_usability(self):

        if self.login:
            resp = await self.exec_login()
            if resp[0] is None:
                self.fail_on_login = True
                self.logger.warning(f"后端{self.backend_name}登陆失败")
                return False, resp

        # api_url = f"{self.backend_url}/sdapi/v1/progress"
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(url=api_url) as resp:
        #         resp_json = await resp.json()
        #         return True, (resp_json, resp.status)
    async def get_backend_working_progress(self):
        """
        获取后端工作进度, 默认A1111
        :return:
        """
        respond = await http_request(
            "GET",
            f"{self.backend_url}/sdapi/v1/options"
        )

        self.model = respond['sd_model_checkpoint']
        self.model_hash = respond
        try:

            if self.current_config['auth'][self.count]:
                self.login = True
                await self.exec_login()

            api_url = f"{self.backend_url}/sdapi/v1/progress"

            async with aiohttp.ClientSession() as session:
                async with session.get(url=api_url) as resp:
                    resp_json = await resp.json()
        except Exception:
            raise RuntimeError
        else:
            return resp_json, resp.status, self.backend_url, resp.status

