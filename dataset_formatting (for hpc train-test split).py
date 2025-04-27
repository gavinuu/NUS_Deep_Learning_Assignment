import pathlib
from pathlib import Path
import typing
from typing import Tuple, List
import logging
import os

def browse_folder(
        path: Path,
        A: str,
        B: str) -> Tuple[List[Path], List[Path]]:
    """
    Browse a directory and return a list of all the jpg files in it and its subfolder.
    """
    logging.info(f"Browsing folder {path}")
    filenames_synth, filenames_real = [], []
    for p in path.iterdir():
        if p.is_dir():
            filenames = browse_folder(p, A, B)
            filenames_real.extend(filenames[0])
            filenames_synth.extend(filenames[1])
        elif p.suffix == '.jpg':
            if A in str(p):
                path_synth = p
                path_real = Path(str(p).replace(A, B))
                filenames_synth.append(path_synth)
                filenames_real.append(path_real)
                
    return filenames_synth, filenames_real #to split into train and test datasets

def check_existence(
        filenames: List[Path]) -> None:
    """
    Check if the paths exist.
    Args:
        filenames (List[Path]): List of filenames.
    Returns:
        None
    """
    for file_path in filenames:
        assert file_path.exists(), f'Msissing corresponding real patch of {file_path}'
        
def create_folders(
        input_dir: Path, 
        folder_list: List[str]) -> None:
    """
    Create folders for training and testing data.
    Args:
        input_dir (Path): Input directory.
        folder_list (List[str]): List of folder names to create.
    Returns:
        None
    """
    os.makedirs(input_dir/folder_list[0], exist_ok=True)
    os.makedirs(input_dir/folder_list[1], exist_ok=True)
    os.makedirs(input_dir/folder_list[2], exist_ok=True)
    os.makedirs(input_dir/folder_list[3], exist_ok=True)

def split_train_test(
        filenames: List[Path], 
        alpha: float = 0.9) -> Tuple[List[Path], List[Path]]:
    """
    Split the filenames into training and testing sets.
    Args:
        filenames (List[Path]): List of filenames.
        alpha (float): Ratio of training data to total data.
    Returns:
        Tuple[List[Path], List[Path]]: Training and testing filenames.
    """
    assert 0 < alpha < 1 
    n = len(filenames)
    n_train = int(n * alpha)
    filenames_train = filenames[:n_train]
    filenames_test = filenames[n_train:]
    return filenames_train, filenames_test

def create_symlinks(
        filenames: List[Path],
        #input_dir: Path, (excluded)
        split_dir: Path) -> None:
    """
    Create symbolic links for the training and testing images.
    Args:
        filenames (List[Path]): List of filenames.
        input_dir (Path): Input directory.
        split_dir (Path): Split directory.
    Returns:
        None
    """
    for i in range(len(filenames)):
        os.symlink(filenames[i], f'{split_dir}/{i}.jpg')

def main():
    dir_name = 'C:/Users/gavin/Desktop/UniStuff/Y4S2/EE4115/Assignment4_retry/NUS_Deep_Learning_Assignment/cloud/dataset'
    datapath = Path(dir_name)
    paths_synth, paths_real = browse_folder(datapath, 'synth', 'real')
    
    check_existence(paths_synth)
    check_existence(paths_real)
    
    create_folders(datapath, ['trainA', 'testA', 'trainB', 'testB'])
    
    trainA, testA = split_train_test(paths_synth) 
    trainB, testB = split_train_test(paths_real)
    
    create_symlinks(trainA, f'{dir_name}/trainA')
    create_symlinks(testA, f'{dir_name}/testA')
    create_symlinks(trainB, f'{dir_name}/trainB')
    create_symlinks(testB, f'{dir_name}/testB')

if __name__ == '__main__':
    main()