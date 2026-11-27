
from image_processing import prepare_image
from VQ_compression import compress
from VQ_decompression import decompress
import os
def main():
    print("=== Vector Quantization Image Compression ===")
    print("1. Compress Image")
    print("2. Decompress Image")

    choice = input("Choose an operation (1/2): ")

    if choice == "1":

        print("\n--- Compression ---")

        image_path = input("Enter image path: ")
        block_h = int(input("Enter block height: "))
        block_w = int(input("Enter block width: "))
        codebook_size = int(input("Enter codebook size: "))

        vectors, image_info = prepare_image(
            image_path, block_h, block_w
        )

        print("\n--- Image Processing ---")

        print("Original dimensions:",
              (image_info["original_height"],
               image_info["original_width"]))

        print("Padded dimensions:",
              (image_info["padded_height"],
               image_info["padded_width"]))

        print("Padding:",
              (image_info["pad_bottom"],
               image_info["pad_right"]))

        print("Block size:",
              (image_info["block_height"],
               image_info["block_width"]))

        print("Number of vectors:", len(vectors))
        print("Vector dimension:", vectors.shape[1])

        codebook, assignments, metadata = compress(
            vectors, image_info, codebook_size
        )

        print("\n--- Compression Results ---")

        print("Codebook shape:", codebook.shape)
        print("Number of assignments:", len(assignments))
        print("Codebook saved as: codebook.npy")
        print("Compressed data saved as: compressed.bin")
        print("Metadata saved as: metadata.json")
        
        original_size = os.path.getsize(image_path)

        compressed_size = (
            os.path.getsize("compressed.bin")
            + os.path.getsize("codebook.npy")
            + os.path.getsize("metadata.json")
        )

        compression_ratio = original_size / compressed_size
        print("\nCompression completed successfully!")
        print(f"Compression Ratio: {compression_ratio:.2f}")

    elif choice == "2":

        print("\n--- Decompression ---")

        img = decompress(
            "compressed.bin",
            "codebook.npy",
            "metadata.json"
        )

        img.save("decompressed.png")

        print("Decompression completed successfully!")
        print("Decompressed image saved as: decompressed.png")
        print("Decompressed image dimensions:", img.size)

    else:
        print("\nInvalid choice! Please choose 1 or 2.")


if __name__ == "__main__":
    main()
