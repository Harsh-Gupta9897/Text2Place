from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *
import numpy as np
import cv2

class CustomTanh(torch.nn.Module):
    def forward(self, input):
        return (torch.tanh(input) + 1) / 2

class NormalizedSoftplus(torch.nn.Module):
    def forward(self, input):
        softplus = torch.nn.functional.softplus(input)
        return softplus / softplus.max()

class NormalizedGELU(torch.nn.Module):
    def forward(self, input):
        gelu = torch.nn.functional.gelu(input)
        return gelu / gelu.max()

@threestudio.register("stable-diffusion-guidance")
class StableDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        # print("pretrained_model_name_or_path: ", pretrained_model_name_or_path)
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = False
        controller = None
    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        print("pretrained_model_name_or_path: ", self.cfg.pretrained_model_name_or_path)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
            # local_files_only= True
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps(self.cfg.min_step_percent, self.cfg.max_step_percent)  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need m
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None



        ###--------- Text2Place Start here --------------###

        #### Parameter Initialization for single blob ########

        # self.center = nn.Parameter(torch.sigmoid(torch.tensor([0.4,0.6])))  # x ∈ [0, 1]^2
        # # self.center = torch.sigmoid(torch.tensor([0.6,0.4]))  # x ∈ [0, 1]^2
        # # self.scale = nn.Parameter(torch.sigmoid(torch.tensor([0.4]))*2) # s ∈ R
        # self.scale = torch.tensor([1.0]) # s ∈ R
        # self.aspect_ratio = nn.Parameter(torch.tensor([2.0])) # a ∈ R
        # self.angle = nn.Parameter(torch.tensor([0 * torch.pi])) # nn.Parameter( torch.tensor(0 * torch.pi)) # θ ∈ [−π, π]
            

        #### Parameter Initialization for multiple blobs ########

        # for multiple persons(n_persons), n= total number of blobs required , default=5 for each person
        self.n_persons = 1
        self.n= 5 * self.n_persons      
        

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Centers for each blob, shape: n x 2 , in case for static fixed centers
        # self.center = nn.Parameter(torch.tensor([[-1.0,-3.0],[-1.0,-1.0],[-1.0,-1.0],[-1.0,-3.0],[-1.0,2.0]]))

        # Centers for each blob, shape: 1 x 2 , in case for dynamic centers
        self.center = nn.Parameter(torch.tensor([[-1.0,-3.0]]))

        # parameters for each blob
        self.scale = nn.Parameter(torch.ones(self.n)*0.6)
        self.aspect_ratio = nn.Parameter(torch.ones(self.n) * 2.0) # a ∈ R for each blob
        self.angle = nn.Parameter(torch.ones(self.n))  # θ ∈ [−π, π] for each blob

        ## relative angles 
        self.relative_angles = nn.Parameter(torch.ones(self.n))
        self.radius = 0.1 #0.07 #0.216  #0.13 



        threestudio.info(f"Loaded Stable Diffusion!")
        

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        text_embed_perturb_variance: False,
    ):
        batch_size = elevation.shape[0]  
        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            if(text_embed_perturb_variance):
                std_dev = text_embed_perturb_variance
                text_embeddings += torch.randn_like(text_embeddings)*std_dev
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils_unpaired: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        training_dcgan=False,
        text_embed_perturb_variance=False,
        guidance_eval=False,
        prompt_utils_paired = None,
        **kwargs,
    ):  
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, t, prompt_utils_unpaired, elevation, azimuth, camera_distances
            )
        else:
            if(prompt_utils_paired == None):
                grad, guidance_eval_utils = self.compute_grad_sds(
                    latents, t, prompt_utils_unpaired, elevation, azimuth, camera_distances,text_embed_perturb_variance
                )
            else:
                grad_unpaired, guidance_eval_utils_unpaired = self.compute_grad_sds(
                    latents, t, prompt_utils_unpaired, elevation, azimuth, camera_distances,text_embed_perturb_variance
                )
                grad = grad_unpaired
                with torch.no_grad():
                    grad_paired, guidance_eval_utils_paired = self.compute_grad_sds(
                        latents, t, prompt_utils_paired, elevation, azimuth, camera_distances,text_embed_perturb_variance
                    )
                    grad_paired = grad_paired.detach()



        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
       
        # loss = SpecifyGradient.apply(latents, grad) | We wish to modify the gradients of the latents by substituting the values from grad, which is not straightforward
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead |
        # Make the target gradients zero and gradient of loss_sds will become d(loss)/d(latents) = grad | Neat trick to avoid initializing the gradients manually 
        target = (latents - grad).detach()
        
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
            "grad_loss": 0,
            "target": target,
            "grad_paired": grad_paired if prompt_utils_paired != None else None,
            "grad_unpaired": grad_unpaired if prompt_utils_paired != None else None,
            "grad":grad
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = latents_noisy.shape[0]  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand(
            [bs, -1]
        ) > t_orig.unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(len(t)):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
            "grad_loss": 0,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

    def compute_gaussian_blobs(self,shape):
        # Constants
        c = torch.tensor([0.02])  # Controls blob edge sharpness   
        # Create a grid of (x, y) coordinates
        y = torch.linspace(0, 1, steps=shape[0], device=self.center.device)
        x = torch.linspace(0, 1, steps=shape[1], device=self.center.device)
        X, Y = torch.meshgrid(x, y)
        grid = torch.stack((X.flatten(), Y.flatten()), dim=1)

        #### ---------------- for one blob ---------------------- ####
        # s_center = 0.6* self.sigmoid(self.center) + 0.2   ## for static centers
        s_center = 0.6* self.sigmoid(self.center) + 0.2

        s_angle = self.tanh(self.angle)*torch.pi
        s_relative_angles = self.tanh(self.relative_angles)*torch.pi/4

        #### ---------------- for n blobs  ---------------------- ####
        # List to store the masks for each blob
        masks = []

        center_idx = self.n//self.n_persons
        cnt=0
        # Compute mask for central blob
        for i in range(self.n):
            if i%center_idx == 0:
                central_mask = self._compute_blob_mask(s_center[i//center_idx], self.scale[i], self.aspect_ratio[i], s_angle[i], grid, shape, c)
                masks.append(central_mask)
                temp_center = s_center[i//center_idx]
                cnt+=1
            else:
                r_angle,scale,aspect_ratio,angle = s_relative_angles[i-cnt], self.scale[i], self.aspect_ratio[i],s_angle[i]
                ref_center = temp_center + torch.stack([self.radius * torch.cos(r_angle), self.radius * torch.sin(r_angle)])
                ref_mask = self._compute_blob_mask(ref_center, scale, aspect_ratio, angle, grid, shape, c)
                # ref_mask = self._compute_blob_mask(s_center[i], scale, aspect_ratio, angle, grid, shape, c) ## for static centers
                temp_center = ref_center
                masks.append(ref_mask)

        return masks


    def _compute_blob_mask(self, center, scale, aspect_ratio, angle, grid, shape, c):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        R = torch.stack([torch.stack([cos_angle, -sin_angle]),
                torch.stack([sin_angle, cos_angle])], dim=0)

        # Compute the covariance matrix Σ
        a = scale * torch.sqrt(aspect_ratio)
        b = scale / torch.sqrt(aspect_ratio)
        Σ = torch.zeros((2, 2), device=center.device)
        Σ[0, 0] = a**2
        Σ[1, 1] = b**2

        Σ  = Σ*c
        # Compute the Mahalanobis distance
        grid_centered = grid - center
        Σ_inv = torch.inverse(torch.matmul(torch.matmul(R, Σ), R.T))
        mahalanobis_distances = torch.matmul(grid_centered, Σ_inv) * grid_centered
        mahalanobis_distances = mahalanobis_distances.sum(dim=1)

        gaussian_blob = torch.exp(-0.5 * mahalanobis_distances)

        # Reshape to the grid size and add to masks list
        mask = gaussian_blob.reshape(shape)
        return mask

    


    