import os
import folder_paths
import logging
import torch
import gc
import psutil
import math
from PIL import ImageOps
import numpy as np

from nodes import MAX_RESOLUTION
import comfy.sd
from comfy_api.input_impl import VideoFromFile
from comfy.comfy_types import IO, ComfyNodeABC
from comfy.utils import ProgressBar, common_upscale
import comfy.model_management as mm
import node_helpers


class FirstLastFrameXZ(ComfyNodeABC):
    """
    Extracts the first or last frame from a selected video file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.filter_files_content_types(
            [
                f for f in os.listdir(folder_paths.get_input_directory())
                if os.path.isfile(os.path.join(folder_paths.get_input_directory(), f))
            ],
            ["video"],
        )
        return {
            "required": {
                "file": (sorted(files), {"video_upload": True}),
                "first_or_last": (["first", "last"],),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_frame"
    CATEGORY = "xzuynodes"

    def extract_frame(self, file, first_or_last):
        path = folder_paths.get_annotated_filepath(file)
        video = VideoFromFile(path)
        frames = video.get_components().images
        try:
            length = len(frames)
        except Exception:
            frames = [frames]
            length = 1
        if length == 0:
            raise ValueError(f"Video '{file}' has no frames.")

        raw = frames[0] if first_or_last == "first" else frames[-1]

        if raw.dim() == 3:
            return (raw.unsqueeze(0),)
        else:
            return (raw,)


class ImageResizeKJ:
    """
    KJNodes original `ImageResizeKJ`. I prefer the way this one handles `keep_proportion` compared to the v2.

    https://github.com/kijai/ComfyUI-KJNodes/blob/0addfc6101f7a834c7fb6e0a1b26529360ab5350/nodes/image_nodes.py#L2137
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,}),
                "upscale_method": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"],),
                "keep_proportion": ("BOOLEAN", { "default": True}),
                "divisible_by": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1,}),
            },
            "optional": {
                "width_input": ("INT", {"forceInput": True}),
                "height_input": ("INT", {"forceInput": True}),
                "get_image_size": ("IMAGE",),
                "crop": (["center", "disabled"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "xzuynodes"
    DESCRIPTION = """
Resizes the image to the specified width and height.  
Size can be retrieved from the inputs, and the final scale  
is  determined in this order of importance:  
- get_image_size  
- width_input and height_input  
- width and height widgets  
  
Keep proportions keeps the aspect ratio of the image, by  
highest dimension.
"""

    def resize(
        self,
        image,
        width,
        height,
        keep_proportion,
        upscale_method,
        divisible_by,
        width_input=None,
        height_input=None,
        get_image_size=None,
        crop="center",
    ):
        B, H, W, C = image.shape

        if width_input:
            width = width_input
        if height_input:
            height = height_input
        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape

        if keep_proportion and get_image_size is None:
            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            if width == 0 and height != 0:
                ratio = height / H
                width = round(W * ratio)
            elif height == 0 and width != 0:
                ratio = width / W
                height = round(H * ratio)
            elif width != 0 and height != 0:
                # Scale based on which dimension is smaller in proportion to the desired dimensions
                ratio = min(width / W, height / H)
                width = round(W * ratio)
                height = round(H * ratio)
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        if divisible_by > 1 and get_image_size is None:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        image = image.movedim(-1, 1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1, -1)

        return (image, image.shape[2], image.shape[1],)


class ImageResizeXZ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,}),
                "upscale_method": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"],),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1,}),
                "crop": (["center", "disabled"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "xzuynodes"
    DESCRIPTION = "Resize image to maintain aspect ratio."

    def resize(
        self,
        image,
        width,
        height,
        upscale_method,
        divisible_by,
        crop="center",
    ):
        B, H, W, C = image.shape
        original_pixels = W * H
        aspect_ratio = W / H
        target_pixels = width * height

        if original_pixels == target_pixels:
            new_width = divisible_by * round(W / divisible_by)
            new_height = divisible_by * round(H / divisible_by)
        else:
            # Estimate width from aspect and target pixel count
            ideal_width = math.sqrt(target_pixels * aspect_ratio)
            new_width = divisible_by * round(ideal_width / divisible_by)

            # Match pixel count
            ideal_height = target_pixels / new_width
            new_height = divisible_by * round(ideal_height / divisible_by)

        image = image.movedim(-1, 1)
        image = common_upscale(image, new_width, new_height, upscale_method, crop)
        image = image.movedim(1, -1)

        return (image, new_width, new_height,)


class CLIPTextEncodeXZ(ComfyNodeABC):
    """
    Uses code from `CLIPTextEncode`, `ConditioningCombined`, and `ConditioningAverage` nodes to split up a prompt and then average or combines the conditioning of all splits.

    - `average` will pad all the conditioning to the same length, stack them, and then get the mean of it all.
    - Enabling `use_mask` will ignore any zeroes during the averaging.
    - `combine` just adds them all together (`sum(conds, [])`).

    https://github.com/comfyanonymous/ComfyUI/blob/f7fb1937127a8ed011b99424598c9ab1e8565112/nodes.py#L49
    https://github.com/comfyanonymous/ComfyUI/blob/f7fb1937127a8ed011b99424598c9ab1e8565112/nodes.py#L72
    https://github.com/comfyanonymous/ComfyUI/blob/f7fb1937127a8ed011b99424598c9ab1e8565112/nodes.py#L84
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "split_string": (
                    ["\\n", ",", "."], {"tooltip": "Delimiter on which to split the prompt before encoding."}
                ),
                "method": (["average", "combine"],),
                "use_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("An averaged or combined conditioning over all text segments.",)
    FUNCTION = "encode"
    CATEGORY = "xzuynodes"
    DESCRIPTION = (
        "Splits the prompt on your delimiter, encodes each piece, "
        "then pads & averages or combines the resulting conditioning layers."
    )

    def encode(self, clip, text, split_string, method, use_mask):
        if split_string == "\\n":
            fixed_split_string = "\n"
        else:
            fixed_split_string = split_string
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint "
                "does not contain a valid clip or text encoder model."
            )

        segs = [s.strip() for s in text.split(fixed_split_string) if s.strip()]
        if not segs:
            # fallback: full prompt
            return (clip.encode_from_tokens_scheduled(clip.tokenize(text)),)

        conds = []
        pbar = ProgressBar(len(segs))
        for seg in segs:
            try:
                tok = clip.tokenize(seg)
                conds.append(clip.encode_from_tokens_scheduled(tok))
            except Exception as e:
                logging.warning(f"CLIP encode failed on segment {seg!r}: {e}")
            pbar.update(1)

        if not conds:
            # if *all* encodings failed, fallback
            return (clip.encode_from_tokens_scheduled(clip.tokenize(text)),)
        elif method == "combine":
            return (sum(conds, []),)

        num_layers = len(conds[0])
        averaged = []
        for layer_idx in range(num_layers):
            # collect tensors and mets
            tensors = []
            pooled = []
            for cond in conds:
                t, meta = cond[layer_idx]
                tensors.append(t)
                pooled.append(meta.get("pooled_output", None))

            # pad tensors to max seq‐length
            seq_lengths = [t.shape[1] for t in tensors]
            max_S = max(seq_lengths)
            padded = []
            for t in tensors:
                B, S, D = t.shape
                if S < max_S:
                    pad = torch.zeros((B, max_S - S, D), device=t.device, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)

            # stack and compute masked mean
            if use_mask:
                # create masks
                masks = [(t.abs().sum(dim=-1) != 0) for t in padded]

                stacked = torch.stack(padded, dim=0)
                mask_stack = torch.stack(masks, dim=0)

                masked = stacked * mask_stack.unsqueeze(-1)  # zero out padded positions
                sum_t = masked.sum(dim=0)                    # sum over segments
                counts = mask_stack.sum(dim=0).clamp(min=1)  # counts of valid tokens
                mean_t = sum_t / counts.unsqueeze(-1)        # divide by counts
            else:
                stacked = torch.stack(padded, dim=0)
                mean_t = torch.mean(stacked, dim=0)

            # average pooled_output if present
            valid_p = [p for p in pooled if p is not None]
            meta_out = {}
            if valid_p:
                p_stack = torch.stack(valid_p, dim=0)
                meta_out["pooled_output"] = torch.mean(p_stack, dim=0)

            averaged.append([mean_t, meta_out])

        return (averaged,)


class CLIPLoaderXZ:
    """
    Same as `CLIPLoader`, but I added a cuda option since default wasn't loading to GPU with lowvram mode.

    https://github.com/comfyanonymous/ComfyUI/blob/f7fb1937127a8ed011b99424598c9ab1e8565112/nodes.py#L919
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (
                    [
                        "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
                        "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace"
                    ],
                ),
            },
            "optional": {
                "device": (["default", "cpu", "cuda:0"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "xzuynodes"
    DESCRIPTION = (
        "[Recipes]\n\n"
        "stable_diffusion: clip-l\n"
        "stable_cascade: clip-g\n"
        "sd3: t5 xxl/ clip-g / clip-l\n"
        "stable_audio: t5 base\n"
        "mochi: t5 xxl\n"
        "cosmos: old t5 xxl\n"
        "lumina2: gemma 2 2B\n"
        "wan: umt5 xxl\n"
        "hidream: llama-3.1 (Recommend) or t5"
    )

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        elif device == "cuda:0":
            model_options["load_device"] = torch.device(device)

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        return (clip,)


class DualCLIPLoaderXZ:
    """
    Same as `DualCLIPLoader`, but I added a cuda option since default wasn't loading to GPU with lowvram mode.

    https://github.com/comfyanonymous/ComfyUI/blob/bd951a714f8c736680fe13e735eee71acf73dd4c/nodes.py#L946
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"],),
            },
            "optional": {
                "device": (["default", "cpu", "cuda:0"], {"advanced": True}),}
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "xzuynodes"

    DESCRIPTION = (
        "[Recipes]\n\n"
        "sdxl: clip-l, clip-g\n"
        "sd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\n"
        "flux: clip-l, t5\n"
        "hidream: at least one of t5 or llama, recommended t5 and llama"
    )

    def load_clip(self, clip_name1, clip_name2, type, device="default"):
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        elif device == "cuda:0":
            model_options["load_device"] = torch.device(device)

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)


class TripleCLIPLoaderXZ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name3": (folder_paths.get_filename_list("text_encoders"), )
            },
            "optional": {
                "device": (["default", "cpu", "cuda:0"], {"advanced": True}),
            },
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "xzuynodes"
    DESCRIPTION = "[Recipes]\n\nsd3: clip-l, clip-g, t5"

    def load_clip(self, clip_name1, clip_name2, clip_name3, device="default"):
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        elif device == "cuda:0":
            model_options["load_device"] = torch.device(device)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
        )
        return (clip,)


class WanImageToVideoXZ:
    """
    Same as `WanImageToVideo` but it runs the memory clearing code of `FreeMemoryBase` before doing *tiled* decoding.

    https://github.com/comfyanonymous/ComfyUI/blob/f7fb1937127a8ed011b99424598c9ab1e8565112/comfy_extras/nodes_wan.py#L10
    https://github.com/ShmuelRonen/ComfyUI-FreeMemory/blob/44fc13f97feec9fdb50ccf342ad64eeb52a95512/free_memory_node.py#L8
    https://github.com/comfyanonymous/ComfyUI/blob/7f492522b6dcb142ff2c4d3438310773d9a80551/nodes.py#L341
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "start_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "xzuynodes"

    def encode(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        tile_size,
        overlap,
        temporal_size,
        temporal_overlap,
        start_image=None,
        clip_vision_output=None,
    ):
        print("Attempting to free GPU VRAM and system RAM aggressively...")
        # GPU VRAM
        if torch.cuda.is_available():
            gpu_before = torch.cuda.memory_allocated()
            mm.unload_all_models()
            mm.soft_empty_cache()
            torch.cuda.empty_cache()
            gpu_after = torch.cuda.memory_allocated()
            freed = (gpu_before - gpu_after) / 1e9
            print(f"GPU VRAM: before={gpu_before/1e9:.2f} GB, after={gpu_after/1e9:.2f} GB, freed={freed:.2f} GB")
        else:
            print("CUDA not available—skipping GPU cleanup.")

        # System RAM
        ram_before = psutil.virtual_memory().percent
        collected = gc.collect()
        print(f"Garbage collector collected {collected} objects.")
        ram_after = psutil.virtual_memory().percent
        print(f"System RAM: before={ram_before:.1f}%, after={ram_after:.1f}%, freed={ram_before - ram_after:.1f}%")

        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image = torch.ones(
                (length, height, width, start_image.shape[-1]),
                device=start_image.device,
                dtype=start_image.dtype,
            ) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode_tiled(
                image[:, :, :, :3],
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=overlap,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )
            mask = torch.ones(
                (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                device=start_image.device,
                dtype=start_image.dtype,
            )
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(
                positive,
                {"concat_latent_image": concat_latent_image, "concat_mask": mask},
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"concat_latent_image": concat_latent_image, "concat_mask": mask},
            )

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )

        return (positive, negative, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "FirstLastFrameXZ": FirstLastFrameXZ,
    "ImageResizeKJ": ImageResizeKJ,
    "ImageResizeXZ": ImageResizeXZ,
    "CLIPTextEncodeXZ": CLIPTextEncodeXZ,
    "CLIPLoaderXZ": CLIPLoaderXZ,
    "DualCLIPLoaderXZ": DualCLIPLoaderXZ,
    "TripleCLIPLoaderXZ": TripleCLIPLoaderXZ,
    "WanImageToVideoXZ": WanImageToVideoXZ,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FirstLastFrameXZ": "First/Last Frame (XZ)",
    "ImageResizeKJ": "Resize Image (Original KJ)",
    "ImageResizeXZ": "Resize Image (XZ)",
    "CLIPTextEncodeXZ": "CLIP Text Encode (XZ)",
    "CLIPLoaderXZ": "Load CLIP (XZ)",
    "DualCLIPLoaderXZ": "DualCLIPLoader (XZ)",
    "TripleCLIPLoaderXZ": "TripleCLIPLoader (XZ)",
    "WanImageToVideoXZ": "WanImageToVideo (XZ)",
}
