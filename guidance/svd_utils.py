from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator

from PIL import Image
import os
import pickle
import random

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims, _resize_with_antialiasing
from diffusers.utils.torch_utils import randn_tensor

class StableVideoDiffusion(nn.Module):
    def __init__(self, device, t_range=[0.02, 0.98]):
        super().__init__()
        
        self.device = device
        print(f'[INFO] loading stable video diffusion...')
        self.precision_t = torch.float32
        self.decode_chunk_size = 4
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained("/home/nas2_userG/junhahyung/kkn/checkpoint/stable-video-diffusion", torch_dtype=torch.float32)
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32)
        
        svd_pipe.enable_sequential_cpu_offload()
        sd_pipe.to("cuda")
        
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=torch.float32)
        self.tokenizer = sd_pipe.tokenizer
        self.text_encoder = sd_pipe.text_encoder
        
        self.sd_unet = sd_pipe.unet
        self.svd_unet = svd_pipe.unet
        
        self.image_processor = svd_pipe.image_processor
        self.feature_extractor = svd_pipe.feature_extractor
        self.image_encoder = svd_pipe.image_encoder
        
        self.sd_vae = sd_pipe.vae
        self.svd_vae = svd_pipe.vae
        
        del svd_pipe
        del sd_pipe

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable video diffusion!')
        
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    def save_local(self, latents, image_latent, image_embedding, scheduler, t):
        data_to_save = {
            'latents': latents,
            'image_latent': image_latent,
            'image_embedding': image_embedding,
            'scheduler': scheduler,
            't': t
        }
        with open('workspace/data.pkl', 'wb') as f:
            pickle.dump(data_to_save, f)
        assert 0, 'saved'
    
    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.svd_unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.svd_unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids
    
    def train_step(self, pred_rgb, clip_embedding, vae_latent, guidance_scale=100, as_latent=False, grad_scale=1, save_guidance_path:Path=None):
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_with_vae(pred_rgb_512, do_classifier_free_guidance=False) # [4, 4, 64, 64]
            
        print(f"[DEBUG] latents shape: {latents.shape}")
        num_frames = latents.shape[0]
        assert num_frames == 4, f'num_frames: {num_frames}'

        self.scheduler.set_timesteps(self.num_train_timesteps)

        t = torch.randint(self.min_step, self.max_step, (1,), device=self.device)
        print(f"[DEBUG] t: {t}")
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t) # [4, 4, 64, 64]
            latents_noisy = latents_noisy.unsqueeze(0) # [1, 4, 4, 64, 64]
            # latents_noisy = latents_noisy.repeat(1, 1, 1, 1, 1) # [1, 4, 4, 64, 64]    
            image_latents = vae_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1) # [2, 4, 4, 64, 64]

            added_time_ids = self._get_add_time_ids(
                fps=30,
                motion_bucket_id=127,
                noise_aug_strength=0.02,
                dtype=torch.float32,
                batch_size=1,
                num_videos_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            added_time_ids = added_time_ids.to(self.device)
            
            latent_model_input = torch.cat([latents_noisy] * 2) # [2, 4, 4, 64, 64]
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            print(f"[DEBUG] latent_model_input shape: {latent_model_input.shape} image_latents shape: {image_latents.shape}")
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2) # [2, 4, 8, 64, 64]
            
            min_guidance_scale = guidance_scale
            max_guidance_scale = guidance_scale
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
            guidance_scale = guidance_scale.to(self.device, latents.dtype)
            guidance_scale = guidance_scale.repeat(1, 1)
            guidance_scale = _append_dims(guidance_scale, latents.ndim)
            
            noise_pred = self.svd_unet(
                latent_model_input,
                t,
                encoder_hidden_states=clip_embedding,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]
            
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        noise_pred = noise_pred[0]
        latents_noisy = latents_noisy[0]
        
        # w(t), sigma_t^2
        t = t.repeat(4)
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        print('ㅡ'*20)
        print(f"noise_pred min: {noise_pred.min()}, noise_pred max: {noise_pred.max()}")
        print(f"noise min: {noise.min()}, noise max: {noise.max()}")
        print(f"grad min: {grad.min()}, grad max: {grad.max()}")
        print("ㅡ"*20)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))
                print(f"[INFO] pred_x0: {pred_x0.shape}, result_hopefully_less_noisy_image: {result_hopefully_less_noisy_image.shape}")

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]
        
        print(f"[DEBUG] grad: {grad}")
        print(f"[DEBUG] loss: {loss}")
        print(f"[DEBUG] loss shape: {loss.shape}")
        print(f"[DEBUG] my_loss: {0.5 * F.mse_loss(grad, 0*grad) / latents.shape[0]}")

        return loss
    
    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.sd_unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.sd_unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / self.sd_vae.config.scaling_factor * latents

        imgs = self.sd_vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs
        
    @torch.no_grad()
    def encode_with_vae(self, image, height=512, width=512, noise_aug_strength=0.02, num_videos_per_prompt=1, do_classifier_free_guidance=True):
        image = self.image_processor.preprocess(image, height=height, width=width).to(self.device)
        
        with torch.random.fork_rng():
            generator = torch.manual_seed(42)
            noise = randn_tensor(image.shape, generator=generator, device=self.device, dtype=self.precision_t)
            image = image + noise_aug_strength * noise
        
        image = image.to(device=self.device)
        image_latents = self.svd_vae.encode(image).latent_dist.mode()
        
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])
        
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1) # [2, 4, 64, 64]
        
        return image_latents
    
    @torch.no_grad()
    def encode_with_clip(self, image, num_videos_per_prompt=1, do_classifier_free_guidance=True):
        dtype = next(self.image_encoder.parameters()).dtype
        device = self.device

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings
    
    def get_image_latent(self, view, pos_embeds, neg_embeds, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]
        
        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        assert imgs.shape[0] == 1
        img = imgs[0]
        img = Image.fromarray(img)
        
        folder_path = 'nerf/data/images'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, f'{view}.png')
        img.save(file_path)
        print(f"Saved image {file_path}!")
        
        clip_embedding = self.encode_with_clip(img)
        vae_latent = self.encode_with_vae(img)
        
        return clip_embedding, vae_latent