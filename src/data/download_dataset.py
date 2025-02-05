import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Progress bar

def download_json(json_url, json_file):
    """Download the JSON file if it does not exist."""
    if os.path.exists(json_file):
        print(f"{os.path.basename(json_file)} already exists, skipping JSON download.")
        return
    print(f"Downloading JSON dataset from {json_url} ...")
    try:
        response = requests.get(json_url, timeout=10)
        response.raise_for_status()
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Downloaded JSON dataset.")
    except Exception as e:
        print(f"Error downloading JSON dataset: {e}")

def download_image(filename, base_url, images_dir):
    """Download a single image if not already present."""
    image_path = os.path.join(images_dir, filename)
    if os.path.exists(image_path):
        # Image already exists; skip downloading.
        return
    image_url = base_url + filename
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        with open(image_path, "wb") as f_out:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f_out.write(chunk)
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def download_images(json_file, images_dir, base_url, num_threads=16):
    """Download all unique images in parallel with a progress bar."""
    # Load the JSON file.
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

    print(f"Found {len(data)} records in JSON file.")
    # Extract unique image filenames.
    image_filenames = set()
    for record in data:
        if "image" in record:
            image_filenames.add(record["image"])
    total_images = len(image_filenames)
    print(f"Found {total_images} unique image filenames.")
    
    # Create the images directory if it doesn't exist.
    os.makedirs(images_dir, exist_ok=True)
    
    # Download images in parallel using a ThreadPoolExecutor.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_image, filename, base_url, images_dir): filename 
                   for filename in image_filenames}
        # Wrap the futures in a tqdm progress bar.
        with tqdm(total=len(futures), desc="Downloading images", unit="img") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                pbar.update(1)
    print("Image download complete.")

def download_dataset(json_url, json_file, images_dir, base_url, num_threads=16):
    """
    Download the dataset JSON (if not present) and then download all images in parallel.
    """
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    download_json(json_url, json_file)
    download_images(json_file, images_dir, base_url, num_threads)

if __name__ == "__main__":
    # Define base directory and paths.
    base_dir = os.path.join("src", "data", "LLaVA_Instruct_Files")
    json_file = os.path.join(base_dir, "llava_instruct_150k.json")
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(base_dir, exist_ok=True)
    
    # Specify the URL to download the JSON dataset.
    json_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
    
    # Base URL for images (using COCO 2017 train images as in your code).
    base_url = "http://images.cocodataset.org/train2017/"
    
    download_dataset(json_url, json_file, images_dir, base_url, num_threads=16)
