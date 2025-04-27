import os
from typing import List, Tuple, Union
from PIL import Image
import numpy as np


def get_filenames(input_dir: str) -> List[str]:
    """
    Get the list of image filenames from the input directory.
    Args:
        input_dir (str): Directory containing the images.
    Returns:
        List[str]: List of image filenames.
    """
    files = os.listdir(input_dir)
    filenames = [f for f in files if os.path.isfile(f'{input_dir}/{f}')]
        
    return filenames


def load_image(
        path: str,
        crop: Tuple[int, int, int, int] = None,
        convert: str = "RGB") -> Union[np.ndarray, None]:
    """
    Load an image from the given path and convert it to a numpy array.
    Args:
        path (str): Path to the image.
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        convert (str): Color mode to convert the image to (default is "RGB").
    Returns:
        np.ndarray: Numpy array of the image.
    """
    with Image.open(path) as img:
        cropped_img = img.crop(crop)
        img_arr = np.array(cropped_img)
    
    return img_arr


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
        img_patch = Image.fromarray(patch)
        img_patch.save(f'{output_dir}/{starting_index}.jpg')
        starting_index += 1
    
    return starting_index
