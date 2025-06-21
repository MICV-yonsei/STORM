import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention


logger = logging.get_logger(__name__)

class StormPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
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

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds
    

    def _compute_cost_function_adj(self, sub_attn: torch.Tensor, sub_attn_adj: torch.Tensor, alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:        
        N = sub_attn.size(0)
        M = sub_attn_adj.size(0) 
        device = sub_attn.device

        positions_n = torch.arange(N, device=device).unsqueeze(1).float() # (N, 1)
        positions_m = torch.arange(M, device=device).unsqueeze(0).float() # (1, M)
        
        position_cost = (positions_n - positions_m).abs()
        value_cost = (sub_attn.unsqueeze(1) - sub_attn_adj.unsqueeze(0)).abs()
        cost_matrix = alpha * position_cost + beta * value_cost

        max_cost = cost_matrix.max()
        if max_cost > 1e-6: 
            cost_matrix = cost_matrix / max_cost
        else:
            cost_matrix = torch.zeros_like(cost_matrix, device=device)
        return cost_matrix
    
    
    def _compute_loss_from_ot(self, time_step: int,
                            attention_store: AttentionStore,
                            indices_to_alter_total: List[int],
                            attention_res: int = 16,
                            smooth_attentions: bool = False,
                            sigma: float = 0.5,
                            kernel_size: int = 3,
                            normalize_eot: bool = False,
                            centroid: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        
        losses, coordinate = self._storm_loss(
            time_step = time_step,
            attention_maps=attention_maps,
            indices_to_alter_total=indices_to_alter_total,
            normalize_eot=normalize_eot,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            centroid = centroid)
        
        return losses, coordinate

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float, max_norm: float = 1.0) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        grad_norm = torch.norm(grad_cond)
        if grad_norm > max_norm:
            grad_cond = grad_cond * (max_norm / grad_norm)
            
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step_spatial(self,
                                        time_step: int,
                                        latents: torch.Tensor,
                                        indices_to_alter_total: List[int],
                                        loss: torch.Tensor,
                                        threshold: float,
                                        text_embeddings: torch.Tensor,
                                        text_input,
                                        attention_store: AttentionStore,
                                        step_size: float,
                                        t: int,
                                        attention_res: int = 16,
                                        smooth_attentions: bool = True,
                                        sigma: float = 0.5,
                                        kernel_size: int = 3,
                                        max_refinement_steps: int = 30,
                                        normalize_eot: bool = False,
                                        centroid: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        
        target_loss = max(0, 1. - threshold)
        
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            loss, _ = self._compute_loss_from_ot(
                                time_step = time_step,
                                attention_store = attention_store,
                                indices_to_alter_total = indices_to_alter_total,    
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                attention_res = attention_res,
                                normalize_eot = False,
                                centroid = centroid)
            
            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            
            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                    f'Finished with a max attention') 
                break

        self.unet.zero_grad()
        return loss, latents
    
    def _compute_centroid_2d(self, attn_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.arange(attn_map.shape[0], device=attn_map.device).float()
        x_centroid = (attn_map.sum(dim=0) * indices).sum() / (attn_map.sum() + 1e-8) 
        y_centroid = (attn_map.sum(dim=1) * indices).sum() / (attn_map.sum() + 1e-8) 
        return x_centroid, y_centroid
    
    def _exp_cost_weight(self, t_values: Union[int, float, torch.Tensor], k: float = 0.4, W_max: int = 100) -> torch.Tensor:
        if not isinstance(t_values, torch.Tensor):
            t_values = torch.tensor(t_values, device=self.device)
        return 1 + (W_max - 1) * (1 - torch.exp(-k * t_values))
    
    def _compute_cost_function(self, attn: torch.Tensor, direction: str, w: float = 100) -> torch.Tensor:
        cost_matrix = torch.zeros_like(attn, device=attn.device)
        centroid_sub_x, centroid_sub_y = self._compute_centroid_2d(attn)
        
        height, width = attn.shape
        x_coords = torch.arange(width, device=attn.device).float().unsqueeze(0) # (1, W)
        y_coords = torch.arange(height, device=attn.device).float().unsqueeze(1) # (H, 1)

        if direction == 'left':
            cost_factor = torch.where(x_coords < centroid_sub_x, 1 / (w * (centroid_sub_x - x_coords + 1e-8)), w * (x_coords - centroid_sub_x + 1e-8))
        elif direction == 'right':
            cost_factor = torch.where(x_coords > centroid_sub_x, 1 / (w * (x_coords - centroid_sub_x + 1e-8)), w * (centroid_sub_x - x_coords + 1e-8))
        elif direction == 'above':
            cost_factor = torch.where(y_coords < centroid_sub_y, 1 / (w * (centroid_sub_y - y_coords + 1e-8)), w * (y_coords - centroid_sub_y + 1e-8))
            cost_factor = cost_factor.transpose(0, 1) # Transpose to match broadcasting with x_coords
        elif direction == 'below':
            cost_factor = torch.where(y_coords > centroid_sub_y, 1 / (w * (y_coords - centroid_sub_y + 1e-8)), w * (centroid_sub_y - y_coords + 1e-8))
            cost_factor = cost_factor.transpose(0, 1) # Transpose to match broadcasting with x_coords
        else: # For 'obj' or other directions
            cost_factor = torch.ones_like(attn, device=attn.device)

        cost_matrix = attn * cost_factor 
        cost_matrix_flat = cost_matrix.flatten().unsqueeze(1).repeat(1, attn.numel())               
        
        return cost_matrix_flat
    
    
    def sinkhorn(self, a, b, cost_matrix, reg=0.1, num_iters=100):
        """
        Compute the Sinkhorn algorithm for optimal transport with regularization.
        Args:
        - a: Source distribution (tensor)
        - b: Target distribution (tensor)
        - cost_matrix: Precomputed cost matrix between source and target
        - reg: Regularization parameter for Sinkhorn algorithm
        - num_iters: Number of iterations for the algorithm
        
        Returns:
        - transport_plan: Optimal transport plan matrix
        """
        # Initialize the scaling vectors
        u = torch.ones_like(a)  # [n]
        v = torch.ones_like(b)  # [m]
        
        # Exponentiated cost matrix
        K = torch.exp(-cost_matrix / (reg+ 1e-8))  # Sinkhorn kernel

        # Sinkhorn iterations
        for _ in range(num_iters):
            u = a / (K @ v + 1e-8)
            v = b / (K.t() @ u + 1e-8)
        
        # Optimal transport plan
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        return transport_plan
    
    def _create_circular_distribution_image(self, image_size, center, radius, sigma=3):
        height, width = image_size
        cx, cy = center
        # Coordinate grid
        x = torch.arange(0, width, dtype=torch.float32).cuda()
        y = torch.arange(0, height, dtype=torch.float32).cuda()
        xv, yv = torch.meshgrid(x, y, indexing="ij") #.cuda()  # torch meshgrid 사용
        # distance between each pixel and the center
        distance = torch.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)
        # calculate the distribution based on the distance
        distribution = torch.exp(-distance ** 2 / (2 * sigma ** 2))
        # Create a circular mask (zero outside the radius)
        mask = distance <= radius
        distribution = distribution * mask.float() 
        # normalize the distribution (0~1)
        distribution_normalized = distribution / distribution.max()
        return distribution_normalized
    
    def _create_circular_distribution(self, sub_attn: torch.Tensor, obj_attn: torch.Tensor, spatial_condition_flag: bool, non_spatial_condition_flag: bool) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        centroid_sub_x, centroid_sub_y = self._compute_centroid_2d(sub_attn)
        centroid_obj_x, centroid_obj_y = self._compute_centroid_2d(obj_attn)
        
        # Coordinate calculation logic
        x_coordinate = (centroid_obj_x - centroid_sub_x).clamp(min=0) / 10
        y_coordinate = (centroid_obj_y - centroid_sub_y).clamp(min=0) / 10
        coordinate = [x_coordinate.item(), y_coordinate.item()]
    
        image_size = (16, 16) # Good candidate for a constant or parameter
        radius = 4
        sigma = 3
        
        if spatial_condition_flag: # 'above', 'below' related
            if non_spatial_condition_flag: # Completely random position
                center_coordinate = torch.rand(4, device=sub_attn.device) * 16
                sub_circular = self._create_circular_distribution_image(image_size, (center_coordinate[0], center_coordinate[1]), radius, sigma)
                obj_circular = self._create_circular_distribution_image(image_size, (center_coordinate[2], center_coordinate[3]), radius, sigma)
            else: # Specific direction ('above', 'below')
                sub_circular = self._create_circular_distribution_image(image_size, (8, (0 + centroid_obj_y) / 2), radius, sigma) # more below
                obj_circular = self._create_circular_distribution_image(image_size, (8, (16 + centroid_sub_y) / 2), radius, sigma) # more above
        else: # 'left', 'right' related
            sub_circular = self._create_circular_distribution_image(image_size, ((0 + centroid_obj_x) / 2, 8), radius, sigma) # more left
            obj_circular = self._create_circular_distribution_image(image_size, ((16 + centroid_sub_x) / 2, 8), radius, sigma) # more right
            
        return sub_circular, obj_circular, coordinate
    
    def adjust_attention(self, sub_attn_adj: torch.Tensor, obj_attn_adj: torch.Tensor, target_shape: Tuple[int, int] = (16, 16)) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if sub_attn_adj.shape != target_shape:
            sub_attn_adj = None
        if obj_attn_adj.shape != target_shape:
            obj_attn_adj = None
        return sub_attn_adj, obj_attn_adj
    
    def _get_attention_maps_for_tokens(self, attention_for_text: torch.Tensor, indices: List[Optional[int]]) -> Optional[torch.Tensor]:
        """Extracts attention maps for given token indices, handling None values.
        
        Args:
            attention_for_text (torch.Tensor): Attention map tensor, expected shape [Batch, Spatial_Dim, Sequence_Length].
            indices (List[Optional[int]]): List of token indices to extract.
            
        Returns:
            Optional[torch.Tensor]: Extracted attention map, averaged across selected tokens, or None if no valid indices.
        """
        valid_indices = [idx for idx in indices if idx is not None]
        if not valid_indices:
            return None # No valid indices, return None

        # Corrected indexing: Access the last dimension (Sequence_Length)
        # attention_for_text shape: [Batch, Spatial_Dim (H*W), Sequence_Length]
        # We want to select specific `Sequence_Length` indices.
        # This results in: [Batch, Spatial_Dim (H*W), len(valid_indices)]
        selected_attentions = attention_for_text[:, :, valid_indices]
        
        # If valid_indices has multiple elements, mean across the last dimension
        # If valid_indices has only one element, this will effectively remove that dimension
        # resulting in [Batch, Spatial_Dim (H*W)]
        if len(valid_indices) > 1:
            return selected_attentions.mean(dim=-1)
        elif len(valid_indices) == 1:
            # If only one index, just squeeze the last dimension
            return selected_attentions.squeeze(-1)
        else: # Should not happen due to initial `if not valid_indices` check, but for robustness
            return None
    
    def _normalize_attention_map(self, attn_map: torch.Tensor) -> torch.Tensor:
        """Normalizes an attention map to sum to 1."""
        return attn_map / (attn_map.sum() + 1e-8)
    
    def _apply_smoothing(self, attn_map: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Applies Gaussian smoothing to an attention map."""
        device = attn_map.device
        # Assuming single channel (1) for these attention maps
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(device)
        padded_attn = F.pad(attn_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
        smoothed_attn = smoothing(padded_attn).squeeze(0).squeeze(0)
        return smoothed_attn
    
    def _storm_loss(self, 
                    time_step: int,
                    attention_maps: torch.Tensor,
                    indices_to_alter_total: List[List[Optional[int]]],
                    smooth_attentions: bool = False,
                    sigma: float = 0.5,
                    kernel_size: int = 3,
                    normalize_eot: bool = False,
                    centroid: Optional[List[int]] = None 
                    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Computes spatial and attribute losses based on attention maps and token indices.
        
        Args:
            time_step (int): Current denoising timestep.
            attention_maps (torch.Tensor): Aggregated attention maps from the UNet.
            indices_to_alter_total (List[List[int | None]]): List of indices for subject, object, and their adjectives.
            smooth_attentions (bool): Whether to apply Gaussian smoothing to attention maps.
            sigma (float): Sigma for Gaussian smoothing kernel.
            kernel_size (int): Kernel size for Gaussian smoothing.
            normalize_eot (bool): Whether to normalize attention by end-of-text token.
            centroid (List[int], optional): Pre-computed centroid coordinates. Defaults to None.
            
        Returns:
            Tuple[torch.Tensor, List[float]]: A tuple containing the total loss and coordinate values.
        """
        device = attention_maps.device

        # 1. Prepare Attention Maps
        prompt_str = self.prompt[0] if isinstance(self.prompt, list) else self.prompt
        last_idx = len(self.tokenizer(prompt_str)['input_ids']) - 1 if normalize_eot else -1
        
        attention_for_text = attention_maps[:, :, 1:last_idx] # [Batch, Height, Width, Token_Length]
        attention_for_text = F.softmax(attention_for_text * 100, dim=-1)

        # Shift indices since the first token was removed
        shifted_indices_to_alter_total = [[idx - 1 if idx is not None else None for idx in sublist] 
                                        for sublist in indices_to_alter_total]
        subject_indices, adjective_indices = shifted_indices_to_alter_total

        # 2. Parse Prompt Conditions
        is_directional_spatial = any(word in prompt_str for word in ["top", "above", "below"])
        is_directional_horizontal = any(word in prompt_str for word in ["left", "right"])
        is_non_spatial = not (is_directional_spatial or is_directional_horizontal)
        is_object_relation = any(word in prompt_str for word in ["side", "next", "near"])
        
        # Determine which index from the token lists corresponds to subject/object based on prompt direction
        # This logic is complex and might benefit from more structured configuration, e.g., mapping prompt keywords to index roles.
        sub_idx_in_list = 0 # Default to first item in the list of indices
        obj_idx_in_list = 1 # Default to second item

        if is_directional_horizontal:
            if "left" in prompt_str: # e.g., "A on the left of B" -> A is subject[0], B is subject[1]
                sub_idx_in_list = 0
                obj_idx_in_list = 1
            elif "right" in prompt_str: # e.g., "A on the right of B" -> A is subject[0], B is subject[1]
                sub_idx_in_list = 0
                obj_idx_in_list = 1
        elif is_directional_spatial:
            if "top" in prompt_str or "above" in prompt_str: # e.g., "A above B"
                sub_idx_in_list = 0
                obj_idx_in_list = 1
            elif "below" in prompt_str: # e.g., "A below B"
                sub_idx_in_list = 0
                obj_idx_in_list = 1
        # If non_spatial, default indices (0, 1) are used for sub/obj.

        # 3. Extract Subject and Object Attention Maps
        # Ensure indices are valid (not None) before extracting
        # It's assumed that indices_to_alter contains at least two lists for subject and object
        sub_attn = self._get_attention_maps_for_tokens(attention_for_text, [subject_indices[sub_idx_in_list]])
        obj_attn = self._get_attention_maps_for_tokens(attention_for_text, [subject_indices[obj_idx_in_list]])
        
        sub_attn_adj = self._get_attention_maps_for_tokens(attention_for_text, [adjective_indices[sub_idx_in_list]])
        obj_attn_adj = self._get_attention_maps_for_tokens(attention_for_text, [adjective_indices[obj_idx_in_list]])

        # Handle cases where attention maps might be None (e.g., if token index was None)
        if sub_attn is None or obj_attn is None:
            logger.warning("Subject or object attention maps are None. Returning zero loss.")
            return torch.tensor(0.0, device=device), [0.0, 0.0]

        # 4. Smooth and Normalize Attention Maps
        if smooth_attentions:
            sub_attn = self._apply_smoothing(sub_attn, kernel_size, sigma)
            obj_attn = self._apply_smoothing(obj_attn, kernel_size, sigma)
        
        # Create circular distributions for target
        sub_circular, obj_circular, coordinate = self._create_circular_distribution(
            sub_attn, obj_attn, is_directional_spatial, is_non_spatial
        )

        # Normalize and flatten all attention maps
        sub_attn_flat = self._normalize_attention_map(sub_attn).flatten()
        obj_attn_flat = self._normalize_attention_map(obj_attn).flatten()
        sub_circular_flat = self._normalize_attention_map(sub_circular).flatten()
        obj_circular_flat = self._normalize_attention_map(obj_circular).flatten()

        sub_attn_adj_flat = None
        obj_attn_adj_flat = None
        if sub_attn_adj is not None:
            sub_attn_adj_flat = self._normalize_attention_map(sub_attn_adj).flatten()
        if obj_attn_adj is not None:
            obj_attn_adj_flat = self._normalize_attention_map(obj_attn_adj).flatten()
            
        # 5. Compute Spatial Cost Matrices
        w = self._exp_cost_weight(time_step)
        
        # Reshape to 2D for _compute_cost_function
        attn_map_res = attention_maps.shape[1] # Assuming square attention maps (e.g., 16)
        sub_attn_2d = sub_attn_flat.reshape(attn_map_res, attn_map_res)
        obj_attn_2d = obj_attn_flat.reshape(attn_map_res, attn_map_res)


        if is_directional_spatial: # 'above' or 'below'
            cost_matrix_sub = self._compute_cost_function(obj_attn_2d, 'above', w=w)
            cost_matrix_obj = self._compute_cost_function(sub_attn_2d, 'below', w=w)
        elif is_directional_horizontal: # 'left' or 'right'
            cost_matrix_sub = self._compute_cost_function(obj_attn_2d, 'left', w=w)
            cost_matrix_obj = self._compute_cost_function(sub_attn_2d, 'right', w=w)
        else: # Non-spatial or other general relationship
            # Default direction 'obj' (or general) for non-spatial scenarios
            cost_matrix_sub = self._compute_cost_function(obj_attn_2d, 'obj', w=w)
            cost_matrix_obj = self._compute_cost_function(sub_attn_2d, 'obj', w=w)

        # 6. Compute Attribute Cost Matrices (for adjectives)
        cost_matrix_sub_adj = None
        cost_matrix_obj_adj = None
        if sub_attn_adj_flat is not None: 
            cost_matrix_sub_adj = self._compute_cost_function_adj(sub_attn_flat, sub_attn_adj_flat)
        if obj_attn_adj_flat is not None:
            cost_matrix_obj_adj = self._compute_cost_function_adj(obj_attn_flat, obj_attn_adj_flat)
                
        # 7. Compute Sinkhorn Transport Plans and Losses
        if is_object_relation: # Relation between two objects (e.g., "cake to the left of suitcase")
            transport_plan_sub = self.sinkhorn(sub_attn_flat, obj_attn_flat, cost_matrix_sub, reg=0.1)
            transport_plan_obj = self.sinkhorn(obj_attn_flat, sub_attn_flat, cost_matrix_obj, reg=0.1)
            loss_sub = torch.sum(transport_plan_sub * cost_matrix_sub)
            loss_obj = torch.sum(transport_plan_obj * cost_matrix_obj)
        else: # Subject/object matching a circular distribution
            transport_plan_sub = self.sinkhorn(sub_attn_flat, sub_circular_flat, cost_matrix_sub, reg=0.1)   
            transport_plan_obj = self.sinkhorn(obj_attn_flat, obj_circular_flat, cost_matrix_obj, reg=0.1)             
            loss_sub = torch.sum(transport_plan_sub * cost_matrix_sub)   
            loss_obj = torch.sum(transport_plan_obj * cost_matrix_obj)
        
        # Attribute Losses calculation
        loss_sub_adj, loss_obj_adj = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        if cost_matrix_sub_adj is not None:
            transport_plan_adj_sub = self.sinkhorn(sub_attn_flat, sub_attn_adj_flat, cost_matrix_sub_adj, reg=0.1)
            loss_sub_adj = torch.sum(transport_plan_adj_sub * cost_matrix_sub_adj)
        if cost_matrix_obj_adj is not None:
            transport_plan_adj_obj = self.sinkhorn(obj_attn_flat, obj_attn_adj_flat, cost_matrix_obj_adj, reg=0.1)
            loss_obj_adj = torch.sum(transport_plan_adj_obj * cost_matrix_obj_adj)
        
        # Combine attribute losses
        loss_adj = (loss_sub_adj + loss_obj_adj) / (2 if cost_matrix_sub_adj is not None and cost_matrix_obj_adj is not None else 1)
            
        # 8. Determine Final Total Loss
        TIME_STEP_THRESHOLD = 7 # This could be an instance variable or passed as a config
        if time_step < TIME_STEP_THRESHOLD: 
            total_loss = loss_sub + loss_obj
        else:
            total_loss = loss_sub + loss_obj + loss_adj

        logger.info(f'Spatial Loss: {loss_sub.item() + loss_obj.item():.4f}, Attribute Loss: {loss_adj.item():.4f}')
        return total_loss, coordinate
        
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
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
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
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
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1
            
        spatial_condition = 0 if "left" in prompt or "right" in prompt else 1 
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():

                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()
                    
                    loss, coordinate = self._compute_loss_from_ot(
                                time_step = i,
                                attention_store = attention_store,
                                indices_to_alter_total = indices_to_alter,    
                                attention_res = attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot = sd_2_1)
                    
                    if not run_standard_sd and len(indices_to_alter[0]) == 2:
                        if i in thresholds.keys() and loss > 1. - thresholds[i] and 0 < i < 25 and coordinate[spatial_condition] > 0:
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            
                            loss, latents = self._perform_iterative_refinement_step_spatial(
                                time_step = i,
                                latents=latents,
                                indices_to_alter_total=indices_to_alter,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=prompt_embeds,
                                text_input=text_inputs,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1)

                        if 0 < i < 25:
                            torch.cuda.empty_cache()
                            
                            loss, _ = self._compute_loss_from_ot(
                                time_step = i,
                                attention_store = attention_store,
                                indices_to_alter_total = indices_to_alter,    
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                attention_res = attention_res,
                                normalize_eot = sd_2_1)
                            
                            if loss != 0:
                                latents = self._update_latent(latents=latents, loss=loss,
                                                              step_size=scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss:0.4f}')
                torch.cuda.empty_cache()
                image = self.decode_latents(latents)
                if output_type == "pil":
                    image = self.numpy_to_pil(image)
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
