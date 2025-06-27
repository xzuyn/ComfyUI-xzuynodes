# xzuynodes-ComfyUI

A collection of nodes I created to solve various issues or ideas I had.

## List of Nodes

### First/Last Frame (XZ)

- Lets you input a video and it will output the first or last frame. Useful for video models if you want to extend.

### Resize Image (Original KJ)

- [KJNodes' original `ImageResizeKJ` node](https://github.com/kijai/ComfyUI-KJNodes/blob/0addfc6101f7a834c7fb6e0a1b26529360ab5350/nodes/image_nodes.py#L2137), as I prefer its `keep_proportion` method.

### Resize Image (XZ)

### CLIP Text Encode (XZ)

### CLIP Loader (XZ)

- Same as `CLIPLoader`, but I added a `"cuda:0"` option since `"default"` wasn't loading to GPU with `--lowvram` mode.

### DualCLIPLoader (XZ)

- Same changes as `CLIP Loader (XZ)`, but using `DualCLIPLoader`.

### TripleCLIPLoader (XZ)

- Same changes as `CLIP Loader (XZ)`, but using `TripleCLIPLoader`.

### WanImageToVideo (XZ)
