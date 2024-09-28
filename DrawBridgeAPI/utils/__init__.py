import json

import httpx
from fastapi.exceptions import HTTPException
from ..base_config import init_instance
config = init_instance.config
import asyncio


async def http_request(
        method,
        target_url,
        headers=None,
        params=None,
        content=None,
        format=True
):
    async with httpx.AsyncClient() as client:

        response = await client.request(
            method,
            target_url,
            headers=headers,
            params=params,
            content=content
        )

        if response.status_code != 200:
            raise HTTPException(500)
        if format:
            return response.json()
        else:
            return response


async def run_later(func, delay=1):
    loop = asyncio.get_running_loop()
    loop.call_later(
        delay,
        lambda: loop.create_task(
            func
        )
    )


async def txt_audit(
        msg,
        prompt='''
        接下来请你对一些聊天内容进行审核,
        如果内容出现政治/暴恐内容（特别是我国的政治人物/或者和我国相关的政治）则请你输出<yes>, 
        如果没有则输出<no>
        '''
):

    from ..backend import Backend

    system = [
        {"role": "system",
         "content": prompt}
    ]
    prompt = [{"role": "user", "content": msg}]

    try:
        resp = Backend.http_request(
        "POST",
        f"http://{config['prompt_audit']['site']}/v1/chat/completions",
        {"Authorization": config['prompt_audit']['api_key']},
        timeout=300,
        format=True,
        content= json.dumps(
            {
            "model": "gpt-3.5-turbo",
            "messages": system + prompt,
            "max_tokens": 4000,
        }
        )
    )
    except:
        return "yes"
    else:
        res: str = remove_punctuation(resp['choices'][0]['message']['content'].strip())
        return res

def remove_punctuation(text):
    import string
    for i in range(len(text)):
        if text[i] not in string.punctuation:
            return text[i:]
    return ""