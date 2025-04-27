import pathlib
from pathlib import Path
import typing
from typing import Tuple, List
import logging
import os


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
        "--A", type=str, required=True,
        help="Directory containing the images of modality A to process."
    )
    parser.add_argument(
        "--B", type=str, required=True,
        help="Directory containing the images of modality B to process."
    )
    parser.add_argument(
        "--folders", type=str, nargs=4, required=True,
        help="List of folder names to create for training and testing data."
    )
    parser.add_argument(
        "--alpha", type=float, required=True,
        help="Ratio of training data to total data."
    )
    return parser


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
                
    return filenames_synth, filenames_real


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


def process(input_dir: str, A: str, B: str, folders: List[str], alpha: float) -> None:
    """
    Format the dataset for training and testing.
    Args:
        data_dir (str): Directory containing the images.
        A (str): Name of the first dataset.
        B (str): Name of the second dataset.
        folders (List[str]): List of folder names to create.
        alpha (float): Ratio of training data to total data.
    Returns:
        None
    """
    datapath = Path(input_dir)
    paths_synth, paths_real = browse_folder(datapath, A, B)
    
    check_existence(paths_synth)
    check_existence(paths_real)
    
    create_folders(datapath, folders)
    
    trainA, testA = split_train_test(paths_synth) 
    trainB, testB = split_train_test(paths_real)
    
    create_symlinks(trainA, f'{input_dir}/trainA')
    create_symlinks(testA, f'{input_dir}/testA')
    create_symlinks(trainB, f'{input_dir}/trainB')
    create_symlinks(testB, f'{input_dir}/testB')
    return NotImplementedError


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()

    process(args.input_dir, args.A, args.B, args.folders, args.alpha)


if __name__ == "__main__":
    main()
