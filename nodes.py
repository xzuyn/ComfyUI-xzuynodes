import os
from comfy.comfy_types import IO, ComfyNodeABC
import folder_paths
from comfy_api.input_impl import VideoFromFile


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


NODE_CLASS_MAPPINGS = {
    "LastFrameNode": LastFrameNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LastFrameNode": "Last Frame Extractor"
}
