import os
import folder_paths
import logging
import torch

from nodes import MAX_RESOLUTION
from comfy_api.input_impl import VideoFromFile
from comfy.comfy_types import IO, ComfyNodeABC
from comfy.utils import ProgressBar, common_upscale


class LastFrameNode(ComfyNodeABC):
    """Extracts the last frame from a selected video file."""

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.filter_files_content_types(
            [
                f for f in os.listdir(folder_paths.get_input_directory())
                if os.path.isfile(os.path.join(folder_paths.get_input_directory(), f))
            ],
            ["video"],
        )
        return {"required": {"file": (sorted(files), {"video_upload": True})}}

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_last_frame"
    CATEGORY = "xzuynodes"

    def extract_last_frame(self, file: str):
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
        return ([frames[-1]],)


class ImageResizeKJ:
    """
    KJNodes original resizing node. I prefer the way this one handles keep_proportion compared to the v2.
    
    https://github.com/kijai/ComfyUI-KJNodes/blob/0addfc6101f7a834c7fb6e0a1b26529360ab5350/nodes/image_nodes.py#L2137
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"],),
                "keep_proportion": ("BOOLEAN", { "default": True }),
                "divisible_by": ("INT", { "default": 8, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "width_input": ("INT", { "forceInput": True}),
                "height_input": ("INT", { "forceInput": True}),
                "get_image_size": ("IMAGE",),
                "crop": (["center", "disabled"],),
            }
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

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, 
               width_input=None, height_input=None, get_image_size=None, crop="center"):
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

        image = image.movedim(-1,1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1,-1)

        return(image, image.shape[2], image.shape[1],)


class CLIPTextEncodeAveraged(ComfyNodeABC):
    """Uses code from CLIPTextEncode and ConditioningAverage nodes to split up a prompt and then average the conditioning of all splits."""

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "text": (IO.STRING, {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "The text to be encoded."
                }),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "split_string": (["\\n", ",", "."], {
                    "tooltip": "Delimiter on which to split the prompt before encoding."
                }),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("An averaged conditioning over all text segments.",)
    FUNCTION = "encode"
    CATEGORY = "xzuynodes"
    DESCRIPTION = (
        "Splits the prompt on your delimiter, encodes each piece, "
        "then pads & averages the resulting conditioning layers."
    )

    def encode(self, clip, text, split_string):
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

            # stack & mean
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


class CLIPTextEncodeCombined(ComfyNodeABC):
    """Uses code from CLIPTextEncode and ConditioningCombine nodes to split up a prompt and then combine the conditioning of all splits."""

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "text": (IO.STRING, {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "The text to be encoded."
                }),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "split_string": (["\\n", ",", "."], {
                    "tooltip": "Delimiter on which to split the prompt before encoding."
                }),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A combined conditioning of all text segments.",)
    FUNCTION = "encode"
    CATEGORY = "xzuynodes"
    DESCRIPTION = (
        "Splits the prompt on your delimiter, encodes each piece, "
        "then combines all resulting conditioning lists."
    )

    def encode(self, clip, text, split_string):
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
        else:
            return (sum(conds, []),)


NODE_CLASS_MAPPINGS = {
    "LastFrameNode": LastFrameNode,
    "ImageResizeKJ": ImageResizeKJ,
    "CLIPTextEncodeAveraged": CLIPTextEncodeAveraged,
    "CLIPTextEncodeCombined": CLIPTextEncodeCombined,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LastFrameNode": "Last Frame Extractor",
    "ImageResizeKJ": "Resize Image (Original KJ)",
    "CLIPTextEncodeAveraged": "CLIP Text Encode (Averaged)",
    "CLIPTextEncodeCombined": "CLIP Text Encode (Combined)",
}
