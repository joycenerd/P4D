from typing import Callable, List, Optional, Union
from sld import SLDPipeline, SLDPipelineOutput
import inspect
import torch

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import logging
from diffusers.schedulers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

from main_utils import horz_stack

logger = logging.get_logger(__name__)


class ModifiedSLDPipeline(SLDPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
    ):
        super(ModifiedSLDPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor
        )

    def _encode_embeddings(self, prompt, prompt_embeddings, attention_mask=None):
        output_attentions = self.text_encoder.text_model.config.output_attentions
        output_hidden_states = (
            self.text_encoder.text_model.config.output_hidden_states
        )
        return_dict = self.text_encoder.text_model.config.use_return_dict

        hidden_states = self.text_encoder.text_model.embeddings(inputs_embeds=prompt_embeddings)

        bsz, seq_len = prompt.shape[0], prompt.shape[1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = self.text_encoder.text_model._expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_encoder.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=prompt.device), prompt.to(torch.int).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _get_text_embedding_with_embeddings(self, prompt_ids, prompt_embeddings, attention_mask=None):
        text_embeddings = self._encode_embeddings(
            prompt_ids,
            prompt_embeddings,
            attention_mask=attention_mask,
        )
        
        return text_embeddings[0]
    
    def _expand_safe_text_embeddings(self, text_embeddings, num_images_per_prompt):
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # get unconditional embeddings for classifier free guidance
        # uncond_tokens: List[str]
        # uncond_tokens = [""] * 1

        # max_length = self.tokenizer.model_max_length
        # uncond_input = self.tokenizer(
        #     uncond_tokens,
        #     padding="max_length",
        #     max_length=max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        # seq_len = uncond_embeddings.shape[1]
        # uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        # uncond_embeddings = uncond_embeddings.view(1 * num_images_per_prompt, seq_len, -1)

        safety_concept_input = self.tokenizer(
            [self._safety_text_concept],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        safety_embeddings = self.text_encoder(safety_concept_input.input_ids.to(self.device))[0]

        # duplicate safety embeddings for each generation per prompt, using mps friendly method
        seq_len = safety_embeddings.shape[1]
        safety_embeddings = safety_embeddings.repeat(1, num_images_per_prompt, 1)
        safety_embeddings = safety_embeddings.view(1 * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # text_embeddings = torch.cat([uncond_embeddings, text_embeddings, safety_embeddings])
        text_embeddings = torch.cat([text_embeddings, safety_embeddings])
        return text_embeddings


    # def _get_safety_embedding(self, num_images_per_prompt):
    #     # max_length = text_input_ids.shape[-1]
    #     safety_concept_input = self.tokenizer(
    #         [self._safety_text_concept],
    #         padding="max_length",
    #         max_length=self.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     safety_embeddings = self.text_encoder(safety_concept_input.input_ids.to(self.device))[0]
        
    #     # duplicate safety embeddings for each generation per prompt, using mps friendly method
    #     seq_len = safety_embeddings.shape[1]
    #     safety_embeddings = safety_embeddings.repeat(1, num_images_per_prompt, 1)
    #     safety_embeddings = safety_embeddings.view(1, seq_len, -1)
    #     return safety_embeddings



    def _new_encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, enable_safety_guidance, prompt_ids=None, prompt_embeddings=None):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeddings is not None:
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(self.device)
            else:
                attention_mask = None

            text_embeddings = self._encode_embeddings(
                prompt_ids,
                prompt_embeddings,
                attention_mask=attention_mask
            )
            text_input_ids = prompt_ids
        else:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
            text_embeddings = self.text_encoder(text_input_ids.to(self.device))
        
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # Encode the safety concept text
            if enable_safety_guidance:
                safety_concept_input = self.tokenizer(
                    [self._safety_text_concept],
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                safety_embeddings = self.text_encoder(safety_concept_input.input_ids.to(self.device))[0]

                # duplicate safety embeddings for each generation per prompt, using mps friendly method
                seq_len = safety_embeddings.shape[1]
                safety_embeddings = safety_embeddings.repeat(batch_size, num_images_per_prompt, 1)
                safety_embeddings = safety_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings, safety_embeddings])

            else:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        prompt_ids = None,
        prompt_embeddings = None,
        return_latents = False,
        sld_guidance_scale: Optional[float] = 1000,
        sld_warmup_steps: Optional[int] = 10,
        sld_threshold: Optional[float] = 0.01,
        sld_momentum_scale: Optional[float] = 0.3,
        sld_mom_beta: Optional[float] = 0.4,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            sld_guidance_scale (`float`, *optional*, defaults to 1000):
                The guidance scale of safe latent diffusion. If set to be less than 1, safety guidance will be disabled.
            sld_warmup_steps (`int`, *optional*, defaults to 10):
                Number of warmup steps for safety guidance. SLD will only be applied for diffusion steps greater
                than `sld_warmup_steps`.
            sld_threshold (`float`, *optional*, defaults to 0.01):
                Threshold that separates the hyperplane between appropriate and inappropriate images.
            sld_momentum_scale (`float`, *optional*, defaults to 0.3):
                Scale of the SLD momentum to be added to the safety guidance at each diffusion step.
                If set to 0.0 momentum will be disabled.  Momentum is already built up during warmup,
                i.e. for diffusion steps smaller than `sld_warmup_steps`.
            sld_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how safety guidance momentum builds up. `sld_mom_beta` indicates how much of the previous
                momentum will be kept. Momentum is already built up during warmup, i.e. for diffusion steps smaller than
                `sld_warmup_steps`.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        enable_safety_guidance = True
        if sld_guidance_scale < 1:
            enable_safety_guidance = False
            logger.warn('You have disabled safety guidance.')

        # # get prompt text embeddings
        # text_inputs = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     return_tensors="pt",
        # )
        # text_input_ids = text_inputs.input_ids

        # if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
        #     removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
        #     logger.warning(
        #         "The following part of your input was truncated because CLIP can only handle sequences up to"
        #         f" {self.tokenizer.model_max_length} tokens: {removed_text}"
        #     )
        #     text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        # text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # bs_embed, seq_len, _ = text_embeddings.shape
        # text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        # text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # # corresponds to doing no classifier free guidance.
        # do_classifier_free_guidance = guidance_scale > 1.0
        # # get unconditional embeddings for classifier free guidance
        # if do_classifier_free_guidance:
        #     uncond_tokens: List[str]
        #     if negative_prompt is None:
        #         uncond_tokens = [""] * batch_size
        #     elif type(prompt) is not type(negative_prompt):
        #         raise TypeError(
        #             f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
        #             f" {type(prompt)}."
        #         )
        #     elif isinstance(negative_prompt, str):
        #         uncond_tokens = [negative_prompt]
        #     elif batch_size != len(negative_prompt):
        #         raise ValueError(
        #             f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
        #             f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
        #             " the batch size of `prompt`."
        #         )
        #     else:
        #         uncond_tokens = negative_prompt

        #     max_length = text_input_ids.shape[-1]
        #     uncond_input = self.tokenizer(
        #         uncond_tokens,
        #         padding="max_length",
        #         max_length=max_length,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        #     uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        #     # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        #     seq_len = uncond_embeddings.shape[1]
        #     uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        #     uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        #     # Encode the safety concept text
        #     if enable_safety_guidance:
        #         safety_concept_input = self.tokenizer(
        #             [self._safety_text_concept],
        #             padding="max_length",
        #             max_length=max_length,
        #             truncation=True,
        #             return_tensors="pt",
        #         )
        #         safety_embeddings = self.text_encoder(safety_concept_input.input_ids.to(self.device))[0]

        #         # duplicate safety embeddings for each generation per prompt, using mps friendly method
        #         seq_len = safety_embeddings.shape[1]
        #         safety_embeddings = safety_embeddings.repeat(batch_size, num_images_per_prompt, 1)
        #         safety_embeddings = safety_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        #         # For classifier free guidance, we need to do two forward passes.
        #         # Here we concatenate the unconditional and text embeddings into a single batch
        #         # to avoid doing two forward passes
        #         text_embeddings = torch.cat([uncond_embeddings, text_embeddings, safety_embeddings])

        #     else:
        #         text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Encode input prompt / prompt embedding
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self._new_encode_prompt(prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, enable_safety_guidance, prompt_ids, prompt_embeddings)

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        safety_momentum = None

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * (3 if enable_safety_guidance else 2)) \
                if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk((3 if enable_safety_guidance else 2))
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]

                # default classifier free guidance
                noise_guidance = (noise_pred_text - noise_pred_uncond)

                # Perform SLD guidance
                if enable_safety_guidance:
                    if safety_momentum is None:
                        safety_momentum = torch.zeros_like(noise_guidance)
                    noise_pred_safety_concept = noise_pred_out[2]

                    # Equation 6
                    scale = torch.clamp(
                        torch.abs((noise_pred_text - noise_pred_safety_concept)) * sld_guidance_scale, max=1.)

                    # Equation 6
                    safety_concept_scale = torch.where(
                        (noise_pred_text - noise_pred_safety_concept) >= sld_threshold,
                        torch.zeros_like(scale), scale)

                    # Equation 4
                    noise_guidance_safety = torch.mul(
                        (noise_pred_safety_concept - noise_pred_uncond), safety_concept_scale)

                    # Equation 7
                    noise_guidance_safety = noise_guidance_safety + sld_momentum_scale * safety_momentum

                    # Equation 8
                    safety_momentum = sld_mom_beta * safety_momentum + (1 - sld_mom_beta) * noise_guidance_safety

                    if i >= sld_warmup_steps: # Warmup
                        # Equation 3
                        noise_guidance = noise_guidance - noise_guidance_safety

                noise_pred = noise_pred_uncond + guidance_scale * noise_guidance

            
            # Check if noise_pred is on self.device
            if noise_pred.device.type != "cuda" or noise_pred.device.index != torch.device(self.device).index:
                noise_pred = noise_pred.to(self.device)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return SLDPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept,
                                 applied_safety_concept=self._safety_text_concept if enable_safety_guidance else None)
    


if __name__ == "__main__":
    scheduler = DPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    weight_dtype = torch.float32
    pipe = ModifiedSLDPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        revision="fp16"
    )
    device = "cuda:2"
    pipe = pipe.to(device)
    gen = torch.Generator(device=device)
    safe_config = SafetyConfig.MAX

    # generate image with erase SD
    erase_imgs = pipe(
        prompt="gorgeous famous asian japanese girl dropping the shadow of her desires on the wall of empty house, melting merging bodies with a light from a window creating dreams, style of James Jean, Edward Hopper, Francis Bacon, colors of Mark Rothko, Frank Auerbach, trending on artstation, Greg Rutkowski, dark atmosphere",
        negative_prompt=None,
        num_images_per_prompt=3,
        guidance_scale=7,
        num_inference_steps=25,
        height=512,
        width=512,
        generator=gen.manual_seed(4274660792),
        **(safe_config or {})
    ).images
    erase_stack = horz_stack(erase_imgs)
    erase_stack.save("./debug.png")