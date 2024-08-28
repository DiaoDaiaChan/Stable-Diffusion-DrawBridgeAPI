import gradio as gr
from PIL import Image

import io
import base64
import httpx
from base_config import logger
from backend import TaskHandler


async def get_all_models(host, port):
    host = '127.0.0.1' if host == '0.0.0.0' else host
    all_models_list = httpx.get(f"http://{host}:{port}/sdapi/v1/sd-models").json()
    return [i['title'] for i in all_models_list]


async def create_gradio_interface(host, port):

    all_models = await get_all_models(host, port)
    logger.info(f"服务器准备就绪! Listen on {host}:{port}")

    async def get_image(model, prompt, negative_prompt, width, height, cfg_scale, steps):

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale
        }

        task_handler = TaskHandler(payload, model_to_backend=model)
        result = await task_handler.txt2img()
        image_data = result.get("images")[0]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        return image

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(label="Model", choices=all_models)
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here...")
                width = gr.Slider(label="Width", minimum=64, maximum=2048, step=1, value=512)
                height = gr.Slider(label="Height", minimum=64, maximum=2048, step=1, value=512)
                cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=0.1, value=7.5)
                steps = gr.Slider(label="Steps", minimum=1, maximum=200, step=1, value=20)
                generate_button = gr.Button("Generate Image")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")

        generate_button.click(get_image, [model, prompt, negative_prompt, width, height, cfg_scale, steps], output_image)

    return demo


async def run_gradio(host, port):
    interface = await create_gradio_interface(host, port)
    interface.launch(server_name=host, server_port=port + 1)
