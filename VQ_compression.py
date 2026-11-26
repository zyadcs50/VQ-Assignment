import numpy as np
import math
import json
import os
def build_codebook(vectors, codebook_size):
    if codebook_size <= 0:
        raise ValueError("Codebook size must be positive.")
    if codebook_size > len(vectors):
        raise ValueError("Codebook size cannot be greater than the number of vectors.")
    indices = np.linspace(0, len(vectors) - 1,codebook_size,dtype=int)
    codebook = vectors[indices].copy()
    return codebook
def calculate_distance(vectors, codebook):
    distances = np.zeros( (len(vectors), len(codebook)))
    for i, vector in enumerate(vectors):
        for j, codeword in enumerate(codebook):
            distances[i, j] = np.linalg.norm(vector - codeword)
    return distances

def assign_vectors(vectors, codebook):
    distances = calculate_distance(vectors,codebook)
    assignments = np.argmin(distances, axis=1)
    return assignments

def update_codebook(vectors, assignments, codebook_size):
    new_codebook = np.zeros((codebook_size, vectors.shape[1]), dtype=np.float32)
    for i in range(codebook_size):
      assigned_vectors = vectors[assignments == i ]
      if len(assigned_vectors) > 0:
            new_codebook[i] = np.mean( assigned_vectors,axis=0)
    return new_codebook

def vector_quantization(vectors,codebook_size,max_iterations=20):
    codebook = build_codebook( vectors,codebook_size)
    for i in range(max_iterations):
        assignments = assign_vectors( vectors,codebook )
        new_codebook = update_codebook(vectors,assignments, codebook_size)
        if np.allclose( codebook, new_codebook):
            codebook = new_codebook
            break
        codebook = new_codebook
    assignments = assign_vectors(vectors, codebook)
    return codebook, assignments


def indices_to_binary(assignments,codebook_size):
    bits_per_index = math.ceil(math.log2(codebook_size))
    binary_stream = ""
    for index in assignments:
        binary_stream += format(int(index),f"0{bits_per_index}b")
    return binary_stream, bits_per_index

def binary_to_bytes(binary_stream):
    padding_bits = ( 8 - len(binary_stream) % 8) % 8
    binary_stream += "0" * padding_bits
    data = bytearray()
    for i in range(0,len(binary_stream),8 ):
        byte = binary_stream[ i:i + 8 ]
        data.append( int(byte, 2))
    return bytes(data), padding_bits


def save_binary_file(filename, data):
    with open(filename, "wb") as file:
        file.write(data)


def save_codebook(filename, codebook):
   np.save(filename,codebook )

def create_metadata(image_info,codebook_size,num_vectors,
                   vector_dimension, bits_per_index,
                     binary_padding_bits,max_iterations):
    metadata = {
        "original_height":
            image_info["original_height"],

        "original_width":
            image_info["original_width"],

        "padded_height":
            image_info["padded_height"],

        "padded_width":
            image_info["padded_width"],

        "block_height":
            image_info["block_height"],

        "block_width":
            image_info["block_width"],

        "pad_bottom":
            image_info["pad_bottom"],

        "pad_right":
            image_info["pad_right"],

        "codebook_size":
            codebook_size,

        "vector_dimension":
            vector_dimension,

        "num_vectors":
            num_vectors,

        "bits_per_index":
            bits_per_index,

        "binary_padding_bits":
            binary_padding_bits,

        "image_mode":
            "L",
        "max_iterations":
            max_iterations
    }

    return metadata


def save_metadata(filename, metadata):
    with open(filename, "w") as file:
        json.dump( metadata,file,indent=4)


def compress(vectors,image_info,codebook_size,
    output_codebook="codebook.npy",output_binary="compressed.bin",
    output_metadata="metadata.json", max_iterations=20):
    codebook, assignments = vector_quantization( vectors,codebook_size, max_iterations)
    # 2. Convert codeword indices to binary
    binary_stream, bits_per_index = indices_to_binary(assignments,codebook_size)
    # 3. Convert binary stream to bytes
    compressed_data, binary_padding_bits = binary_to_bytes( binary_stream)
    # 4. Save codebook
    save_codebook(output_codebook,codebook )
    # 5. Save compressed binary
    save_binary_file(output_binary, compressed_data)

    # 6. Create metadata
    metadata = create_metadata(
        image_info=image_info,
        codebook_size=codebook_size,
        num_vectors=len(vectors),
        vector_dimension=vectors.shape[1],
        bits_per_index=bits_per_index,
        binary_padding_bits=binary_padding_bits,
        max_iterations=max_iterations
    )
    save_metadata(output_metadata, metadata)

    return codebook, assignments, metadata


def calculate_compression_ratio(original_height,original_width,
    codebook_file,compressed_file, metadata_file):
    original_size = (original_height * original_width )
    codebook_size = os.path.getsize(codebook_file)
    compressed_size = os.path.getsize(compressed_file)
    metadata_size = os.path.getsize( metadata_file )
    total_compressed_size = ( codebook_size + compressed_size + metadata_size )
    data_compression_ratio = ( original_size / compressed_size)
    total_compression_ratio = ( original_size / total_compressed_size)
    return (data_compression_ratio, total_compression_ratio)