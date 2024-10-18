import asyncio
import base64
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
from torch import nn
from io import BytesIO
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, \
    AutoModelForCausalLM
import torch
import torch.amp.autocast_mode
from PIL import Image
import numpy as np
from io import BytesIO

from ..base_config import init_instance , setup_logger
from ..locales import _

llm_logger = setup_logger('[LLM-Caption]')

class JoyPipeline:
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None
        self.parent = None

    def clearCache(self):
        self.clip_model = None
        self.clip_processor = None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None


class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Joy_caption_load:

    def __init__(self):
        self.model = None
        self.pipeline = JoyPipeline()
        self.pipeline.parent = self
        self.config = init_instance.config
        pass

    def loadCheckPoint(self):
        # 清除一波
        if self.pipeline != None:
            self.pipeline.clearCache()

            # clip
        model_id = self.config.server_settings['llm_caption']['clip']

        model = AutoModel.from_pretrained(model_id)
        clip_processor = AutoProcessor.from_pretrained(model_id)
        clip_model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        clip_model = clip_model.vision_model
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to("cuda")

        # LLM
        model_path_llm = self.config.server_settings['llm_caption']['llm']
        tokenizer = AutoTokenizer.from_pretrained(model_path_llm, use_fast=False)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer,
                                                                        PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        text_model = AutoModelForCausalLM.from_pretrained(model_path_llm, device_map="auto", trust_remote_code=True)
        text_model.eval()

        # Image Adapte

        image_adapter = ImageAdapter(clip_model.config.hidden_size,
                                     text_model.config.hidden_size)  # ImageAdapter(clip_model.config.hidden_size, 4096)
        image_adapter.load_state_dict(torch.load(self.config.server_settings['llm_caption']['image_adapter'], map_location="cpu", weights_only=True))
        adjusted_adapter = image_adapter  # AdjustedImageAdapter(image_adapter, text_model.config.hidden_size)
        adjusted_adapter.eval()
        adjusted_adapter.to("cuda")

        self.pipeline.clip_model = clip_model
        self.pipeline.clip_processor = clip_processor
        self.pipeline.tokenizer = tokenizer
        self.pipeline.text_model = text_model
        self.pipeline.image_adapter = adjusted_adapter

    def clearCache(self):
        if self.pipeline != None:
            self.pipeline.clearCache()

    def gen(self, model):
        if self.model == None or self.model != model or self.pipeline == None:
            self.model = model
            self.loadCheckPoint()
        return (self.pipeline,)


class Joy_caption:

    def __init__(self):
        pass

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image:
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def gen(
            self,
            joy_pipeline=JoyPipeline,
            image=Image,
            prompt="A descriptive caption for this image",
            max_new_tokens=300,
            temperature=0.5,
            cache=False
    ):

        if joy_pipeline.clip_processor == None:
            joy_pipeline.parent.loadCheckPoint()

        clip_processor = joy_pipeline.clip_processor
        tokenizer = joy_pipeline.tokenizer
        clip_model = joy_pipeline.clip_model
        image_adapter = joy_pipeline.image_adapter
        text_model = joy_pipeline.text_model

        input_image = image

        # Preprocess image
        pImge = clip_processor(images=input_image, return_tensors='pt').pixel_values
        pImge = pImge.to('cuda')

        # Tokenize the prompt
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False,
                                  add_special_tokens=False)
        # Embed image
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pImge, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to('cuda')

        # Embed prompt
        prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda'))
        assert prompt_embeds.shape == (1, prompt.shape[1],
                                       text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
        embedded_bos = text_model.model.embed_tokens(
            torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            prompt,
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)

        generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                           max_new_tokens=max_new_tokens, do_sample=True, top_k=10,
                                           temperature=temperature, suppress_tokens=None)

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        r = caption.strip()

        if cache == False:
            joy_pipeline.parent.clearCache()

        return (r,)


class JoyCaptionHandler:
    def __init__(self, config):
        self.config = config
        self.pipeline, self.joy_caption = self._initialize()

    def _initialize(self):
        llm_logger.info(_("Loading LLM"))
        joy_caption_load = Joy_caption_load()
        model_path = self.config.server_settings['llm_caption']['llm']
        pipeline, = joy_caption_load.gen(model_path)
        joy_caption = Joy_caption()
        llm_logger.info(_("LLM loading completed, waiting for command"))
        return pipeline, joy_caption

    async def get_caption(self, image, ntags=[]):
        if image.startswith(b"data:image/png;base64,"):
            image = image.replace("data:image/png;base64,", "")
        image = Image.open(BytesIO(base64.b64decode(image))).convert(mode="RGB")

        extra_ = f"do not describe {','.join(ntags)} if it exist" if ntags else ''
        loop = asyncio.get_event_loop()

        caption = await loop.run_in_executor(
            None,
            self.joy_caption.gen,
            self.pipeline,
            image,
            f"A descriptive caption for this image, do not describe a signature or text in the image,{extra_}",
            300,
            0.5,
            True
        )

        return caption[0]


config = init_instance.config
if config.server_settings['llm_caption']['enable']:
    joy_caption_handler = JoyCaptionHandler(config)
