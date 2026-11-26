
import numpy as np
import math
import json
from PIL import Image


def get_compressed_file(filename):
    with open(filename, "rb") as file:
        compressed_data = file.read()
    return compressed_data


def get_codebook(filename):
    codebook = np.load(filename)
    return codebook



def get_metadata(filename):
    with open(filename, "r") as file:
        metadata = json.load(file)
    return metadata


def bytes_to_binary_stream(data):
    binary_stream = ""
    for byte in data:
        binary_stream += format(byte, "08b")
    return binary_stream



# codebook = get_codebook("codebook.npy")

# print(codebook.shape)

def binary_to_indices(binary_stream, bits_per_index, padding_bits):
    indices = []
    if padding_bits > 0:
        binary_stream = binary_stream[:-padding_bits]
        
    for i in range(0, len(binary_stream), bits_per_index):
        index_bits = binary_stream[i:i + bits_per_index]
        index = int(index_bits, 2)
        indices.append(index)
    return indices
    
    
    
    

def recreate_vectors_from_indices(indices, codebook):
    vectors = []
    for index in indices:
        vector = codebook[index]
        vectors.append(vector)
    return np.array(vectors)




def reconstruct_image(vectors, metadata):
    block_width = metadata["block_width"]
    block_height = metadata["block_height"]
    
    blocks = vectors.reshape(len(vectors) , block_height, block_width)
    
    padded_width = metadata["padded_width"]
    padded_height = metadata["padded_height"]
    blocks_per_col = padded_height // block_height
    blocks_per_row = padded_width // block_width


    image = blocks.reshape(blocks_per_col, blocks_per_row, block_height, block_width)
    
    image = image.swapaxes(1,2)
    
    image = image.reshape(padded_height, padded_width)
    
    original_height = metadata["original_height"]
    original_width = metadata["original_width"]
    img = image[:original_height, :original_width]
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    final_img = Image.fromarray(img)
    final_img.save("decompressed.png")
    return final_img


def decompress(compressed_filename, codebook_filename, metadata_filename):

    compressed_data = get_compressed_file(compressed_filename)

    codebook = get_codebook(codebook_filename)

    metadata = get_metadata(metadata_filename)

    binary_stream = bytes_to_binary_stream(compressed_data)

    indices = binary_to_indices(
        binary_stream,
        metadata["bits_per_index"],
        metadata["binary_padding_bits"]
    )

    vectors = recreate_vectors_from_indices(indices, codebook)

    final_img = reconstruct_image(vectors, metadata)

    return final_img



