import os
import folder_paths

from nodes import MAX_RESOLUTION
from comfy_api.input_impl import VideoFromFile
from comfy.comfy_types import IO, ComfyNodeABC
from comfy.utils import common_upscale


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
               width_input=None, height_input=None, get_image_size=None, crop="disabled"):
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


NODE_CLASS_MAPPINGS = {
    "LastFrameNode": LastFrameNode,
    "ImageResizeKJ": ImageResizeKJ,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LastFrameNode": "Last Frame Extractor",
    "ImageResizeKJ": "Resize Image (Original KJ)",
}
