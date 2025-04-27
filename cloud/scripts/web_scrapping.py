import os
import requests
from datetime import datetime, timedelta
import argparse


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--start_date", type=str, required=True,
        help="Start date for downloading images (format: YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end_date", type=str, required=True,
        help="End date for downloading images (format: YYYY-MM-DD)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the images."
    )
    return parser


def download_images(
        start_date  : str, 
        end_date    : str,
        output_dir  : str) -> None:
    """
    Downloads satellite images from NEA from start_date to end_date (inclusive).
    
    Args:
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'
        output_dir (str): Directory to save images
    Returns:
        None
    """
    #NEA website main url 
    url_main = 'https://www.nea.gov.sg/docs/default-source/satelliteimage/BlueMarbleASEAN_'
    
    #Set start time
    start_time = '0000'
    start_str = start_date + '_' + start_time
    start = datetime.strptime(start_str, '%Y%m%d_%H%M')
    
    #Set end time
    end_time = '2340'
    end_str = end_date + '_' + end_time
    end = datetime.strptime(end_str, '%Y%m%d_%H%M')
    
    #increment period (20min)
    twenty_min = timedelta(minutes = 20)
    
    while end >= start:
        #get full url for img
        url_date_time = datetime.strftime(start, '%Y%m%d_%H%M')
        url_full = url_main + url_date_time + '.jpg'
        data = requests.get(url_full).content
        
        #create img file        
        file = open(f'{output_dir}/img_{url_date_time}.jpg', 'wb')
        file.write(data)
        file.close()
        print(f'img for {url_date_time} downloaded in {output_dir}')
        
        #increment start time by 20min for next img url
        start += twenty_min


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()
    download_images(args.start_date, args.end_date, args.output_dir)


if __name__ == "__main__":
   main()
    
