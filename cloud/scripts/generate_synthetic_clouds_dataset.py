import os
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse
from utils import load_image, patchify, save_patches, get_filenames
import pylab
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing the images to process."
    )
    parser.add_argument(
        "--input_mask", type=str, required=True,
        help="Directory containing the mask."
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=2, required=True,
        help="Size of the patches (height, width)."
    )
    parser.add_argument(
        "--crop", type=int, nargs=4, required=True,
        help="Coordinates for cropping (left, upper, right, lower)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the patches."
    )
    parser.add_argument(
        "--max_files", type=int, default=-1,
        help="Maximum number of files to process. Default is -1 (all files)."
    )
    return parser


# linear interpolation
def lerp(a, b, x):
    "linear interpolation i.e dot product"
    return a + x * (b - a)


# smoothing function, 1st & 2nd derivative = 0 for this function
def fade(f):
    return (-2 * f ** 3) + (3 * f ** 2) 


# calculate gradient vectors & dot product
def gradient(c, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    gradient_co = vectors[c % 4]
    return gradient_co[:, :, 0] * x + gradient_co[:, :, 1] * y


#perlin noise generator
def perlin(x, y, seed):
    # create a permutation table based on number of pixels, with randomised seed
    np.random.seed(seed)
    ptable = np.arange(1024 * 512, dtype=int)

    # shuffle our numbers in the table
    np.random.shuffle(ptable)

    # create a 2d array and then turn it one dimensional
    # so that we can apply our dot product interpolations easily
    ptable = np.stack([ptable, ptable]).flatten()
    
    # grid coordinates
    xi, yi = x.astype(int), y.astype(int)
   
    # distance vector coordinates
    xg, yg = x - xi, y - yi
    
    # apply fade function to distance coordinates
    xf, yf = fade(xg), fade(yg)
    
    # the gradient vector coordinates in the top left, top right, bottom left bottom right
   
    n00 = gradient(ptable[ptable[xi] + yi], xg, yg)
    n01 = gradient(ptable[ptable[xi] + yi + 1], xg, yg - 1)
    n11 = gradient(ptable[ptable[xi + 1] + yi + 1], xg - 1, yg - 1)
    n10 = gradient(ptable[ptable[xi + 1] + yi], xg - 1, yg)
    
    # apply linear interpolation i.e dot product to calculate average
    x1 = lerp(n00, n10, xf)
    x2 = lerp(n01, n11, xf)  
    return lerp(x1, x2, yf)  


#generate randomised perlin noise sample (clouds) as plot -> convert to array
def generate_synthetic_clouds(
        shape: Tuple[int, int],
        res: Tuple[int, int],
        seed: Tuple[int, int]
        #octaves: int (excluded)
        ) -> np.ndarray:
    """
    Generate a 2D numpy array of noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        octaves: The number of octaves in the noise.
    Returns:
        A numpy array of shape shape with the generated noise.
    """
    # create evenly spaced out numbers in a specified interval
    lin_array_width = np.linspace(1, res[0], shape[0], endpoint=False)
    lin_array_height = np.linspace(1, res[1], shape[1], endpoint=False)

    # create grid using linear 1d arrays
    x, y = np.meshgrid(lin_array_width, lin_array_height)

    #generate perlin noise plot
    rng = np.random.default_rng()
    rand_seed = rng.integers(low = seed[0], high = seed[1], size = 1)
    pylab.ioff()
    plot = plt.figure(figsize = ((shape[0]/100, shape[1]/100)))
    plt.axis('off')
    plt.imshow(perlin(x, y, rand_seed))
    plot.tight_layout(pad = 0)
    plot.canvas.draw()
    
    #output 2d noise array
    noise_arr = np.frombuffer(plot.canvas.tostring_rgb(), dtype=np.uint8)
    noise_arr = noise_arr.reshape(plot.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return noise_arr


#apply clouds array to mask generated from 'isolate_borders' script
def apply_synthetic_clouds_to_mask(
        noise: np.ndarray,
        mask: np.ndarray) -> np.ndarray:
    """Apply the generated noise to the mask. First, normalize the
    noise to be between 0 and 255, then add the noise to the mask.
    The mask is assumed to be in the range [0, 255]. Clip the output
    to be in the range [0, 255] and convert it to uint8.
    Args:
        noise: The generated noise.
        mask: The mask to apply the noise to.
    Returns:
        A numpy array of the same shape as the mask with the noise
        applied.
    """
    # convert RGB to grayscale
    noise_gray = np.dot(noise[...,:3], [0.2989, 0.5870, 0.1140])
    
    output_arr = noise_gray + mask
    
    return output_arr


def patchify(
        img: np.ndarray,
        patch_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """
    Patchify the image into patches of size patch_sizes.
    Args:
        img (np.ndarray): The image to patchify.
        patch_sizes (Tuple[int, int]): The size of the patches.
    Returns:
        List[np.ndarray]: A list of patches.
    """
    patches = []
    
    for i in range(0, img.shape[0], patch_size[0]):
        for j in range(0, img.shape[1], patch_size[1]):
            patch = img[i: i + patch_size[0], j: j + patch_size[1]]
            patches.append(patch)
    
    return patches


#redefine to save imgs as grayscale (otherwise can't save)
def save_patches(
        patches: List[np.ndarray], 
        output_dir: str,
        starting_index: int) -> int:
    """Save the synthetic clouds to the output directory.
    Args:
        patches: The list of patches.
        output_dir: The output directory.
        starting_index: The starting index for naming the files.
    Returns:
        None
    """
    
    for patch in patches:
        img_patch = Image.fromarray(patch).convert('L')
        img_patch.save(f'{output_dir}/{starting_index}.jpg')
        starting_index += 1
    
    return starting_index


def process(
        N: int, 
        input_mask: str,
        patch_size: Tuple[int, int],
        #crop: Tuple[int, int, int, int],
        output_dir: str) -> None:
    """
    Process the real clouds by loading the images, patchifying them,
    and saving the patches to the output directory.
    Args:
        N (int): Number of images to process.
        input_mask (str): Path to the input mask.
        patch_size (Tuple[int, int]): Size of the patches (height, width).
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        output_dir (str): Directory to save the patches.
    Returns:
        None
    """
    shape = (1024, 512)
    res = (10, 6)
    seed = (20, 50)
    
    #load mask 
    mask_img = Image.open(input_mask).convert('L')
    mask_arr = np.array(mask_img)
    
    i = 0

    while (i < N): #use N = no. real patches generated earlier 
        #generate perlin noise array of size 1024x512px
        noise_arr = generate_synthetic_clouds(shape, res, seed)
        
        #add noise array to mask
        output_arr = apply_synthetic_clouds_to_mask(noise_arr, mask_arr)
        
        #convert output array into patches
        patches = patchify(output_arr, patch_size)
        i = save_patches(patches, output_dir, i)
    
    return NotImplementedError


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()
    filenames = get_filenames(args.input_dir)
    if args.max_files > 0:
        filenames = filenames[:args.max_files]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    process(len(filenames), args.input_mask, args.patch_size, args.crop, args.output_dir)


if __name__ == "__main__":
    main()
