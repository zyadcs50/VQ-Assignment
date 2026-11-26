import numpy as np
from PIL import Image
def prepare_image(image_path, block_h, block_w):
    if block_h <= 0 or block_w <= 0:
        raise ValueError("Block dimensions must be positive.")
    # 2. Read the input image
    img = Image.open(image_path)

    # 3. Convert image to grayscale
    img = img.convert("L")
    image = np.array(img, dtype=np.uint8)

    # 4. Store original image dimensions
    original_height, original_width = image.shape

    # 5. Calculate required padding
    pad_bottom = (block_h - original_height % block_h) % block_h
    pad_right = (block_w - original_width % block_w) % block_w

    # 6. Apply padding
    padded_image = np.pad(image,((0, pad_bottom), (0, pad_right)),mode="edge")

    # 7. Store padded dimensions
    padded_height, padded_width = padded_image.shape

    # 8. Divide image into blocks
    blocks_per_col = padded_height // block_h
    blocks_per_row = padded_width // block_w

    blocks = (padded_image.reshape(blocks_per_col,block_h,
            blocks_per_row,block_w ).swapaxes(1, 2).reshape(-1, block_h, block_w))

    # 9. Convert each block into a vector
    vectors = blocks.reshape(blocks.shape[0], -1).astype(np.float32)
    # 10. Store information needed by other modules
    image_info = {
        "original_height": original_height,
        "original_width": original_width,
        "padded_height": padded_height,
        "padded_width": padded_width,
        "block_height": block_h,
        "block_width": block_w,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right
    }

    return vectors, image_info