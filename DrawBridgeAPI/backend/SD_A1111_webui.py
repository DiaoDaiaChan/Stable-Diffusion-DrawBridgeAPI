from urllib.parse import urlencode

from .base import Backend


class AIDRAW(Backend):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.logger = self.setup_logger('[SD-A1111]')

    async def exec_login(self):
        login_data = {
            'username': self.current_config['username'][self.count],
            'password': self.current_config['password'][self.count]
        }
        encoded_data = urlencode(login_data)

        response = await self.http_request(
            method="POST",
            target_url=f"{self.backend_id}/login",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "accept": "application/json"
            },
            content=encoded_data,
        )
        if response.get('error') == "error":
            self.logger.warning(f"后端{self.backend_name}登录失败")
            self.fail_on_login = True
            return False, 500
        else:
            self.logger.info(f"后端{self.backend_name}登录成功")
            return True, 200

    async def check_backend_usability(self):

        if self.login:
            resp = await self.exec_login()
            if resp[0] is None:
                self.fail_on_login = True
                self.logger.warning(f"后端{self.backend_name}登陆失败")
                return False, resp

    async def get_backend_working_progress(self):
        """
        获取后端工作进度, 默认A1111
        :return:
        """
        respond = await self.http_request(
            "GET",
            f"{self.backend_id}/sdapi/v1/options",
            verify=False,
            proxy=False,
            use_aiohttp=False
        )

        print(respond)

        self.model = respond['sd_model_checkpoint']
        self.model_hash = respond

        if self.current_config['auth'][self.count]:
            self.login = True
            await self.exec_login()

        api_url = f"{self.backend_id}/sdapi/v1/progress"

        resp = await self.http_request(
            method="GET",
            target_url=api_url,
            format=False
        )

        resp_json = resp.json()
        return resp_json, resp.status_code, self.backend_id, resp.status_code


