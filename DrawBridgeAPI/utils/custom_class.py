# custom_client.py
from fal_client.client import AsyncClient
from fal_client.auth import fetch_credentials
from dataclasses import field
import httpx
import os

USER_AGENT = "fal-client/0.2.2 (python)"


class CustomAsyncClient(AsyncClient):
    def __init__(self, key=None, default_timeout=120.0):
        if key is None:
            key = os.getenv("FAL_KEY")
        super().__init__(key=key, default_timeout=default_timeout)

    @property
    def _client(self):
        key = self.key
        if key is None:
            key = fetch_credentials()

        return httpx.AsyncClient(
            headers={
                "Authorization": f"Key {key}",
                "User-Agent": USER_AGENT,
            },
            timeout=self.default_timeout,
        )
