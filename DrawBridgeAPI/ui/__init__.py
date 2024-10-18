import json
import asyncio
import gradio as gr
from PIL import Image

import io
import base64
import httpx
from ..base_config import init_instance
from ..backend import TaskHandler
from ..locales import _


class Gradio:
    def __init__(self, host, port):
        self.host = '127.0.0.1' if host == '0.0.0.0' else host
        self.port = port


    def get_caption(self, image):
        caption = httpx.post(
            f"http://{self.host}:{self.port}/tagger/v1/interrogate",
             json=json.loads({"image": image}), timeout=600).json()
        return caption


def format_caption_output(caption_result):
    llm_text = caption_result.get("llm", '')
    word_scores = "\n".join([f"{word}: {score}" for word, score in caption_result["caption"].items()])
    word_ = ",".join([f"{word}" for word in caption_result["caption"].keys()])
    return llm_text, word_scores, word_


def create_gradio_interface(host, port):

    gradio_api = Gradio(host, port)
    from ..api_server import api_instance
    all_models = [i['title'] for i in asyncio.run(api_instance.get_sd_models())]
    init_instance.logger.info(f"{_('Server is ready!')} Listen on {host}:{port}")

    def get_image(model, prompt, negative_prompt, width, height, cfg_scale, steps):

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale
        }

        task_handler = TaskHandler(payload, model_to_backend=model)
        result = asyncio.get_running_loop().run_in_executor(None, task_handler.txt2img)
        image_data = result.get("images")[0]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        return image

    with gr.Blocks() as demo:
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(label="Model", choices=all_models)
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                    negative_prompt = gr.Textbox(label="Negative Prompt",
                                                 placeholder="Enter your negative prompt here...")
                    width = gr.Slider(label="Width", minimum=64, maximum=2048, step=1, value=512)
                    height = gr.Slider(label="Height", minimum=64, maximum=2048, step=1, value=512)
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=0.1, value=7.5)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=200, step=1, value=20)
                    generate_button = gr.Button("Generate Image")

                with gr.Column():
                    output_image = gr.Image(label="Generated Image")

            generate_button.click(get_image, [model, prompt, negative_prompt, width, height, cfg_scale, steps],
                                  output_image)

        with gr.Tab("Caption"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image")
                    caption_button = gr.Button("Get Caption")

                with gr.Column():
                    llm_output = gr.Textbox(label="Natural Language Description")
                    word_output_ = gr.Textbox(label="Keywords", lines=6)
                    word_output = gr.Textbox(label="Keywords with Scores", lines=6)

                caption_button.click(
                    lambda image: format_caption_output(gradio_api.get_caption(image)),
                    inputs=[input_image],
                    outputs=[llm_output, word_output, word_output_]
                )

    return demo


def run_gradio(host, port):
    interface = create_gradio_interface(host, port)
    interface.launch(server_name=host, server_port=port+1)
