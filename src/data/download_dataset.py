import os
import json
import requests

# Set paths: adjust these paths as needed.
base_dir = os.path.join("src", "data", "LLaVA_Instruct_Files")
json_file = os.path.join(base_dir, "llava_instruct_150k.json")
images_dir = os.path.join(base_dir, "images")

# Create the images directory if it doesn't exist.
os.makedirs(images_dir, exist_ok=True)

# Use the COCO 2017 train images base URL.
base_url = "http://images.cocodataset.org/train2017/"

# Load the JSON file.
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Found {len(data)} records in JSON file.")

# Extract unique image filenames.
image_filenames = set()
for record in data:
    if "image" in record:
        image_filenames.add(record["image"])

print(f"Found {len(image_filenames)} unique image filenames.")

# Download each image if not already present.
for filename in image_filenames:
    image_path = os.path.join(images_dir, filename)
    if os.path.exists(image_path):
        print(f"{filename} already exists, skipping.")
        continue

    image_url = base_url + filename
    print(f"Downloading {filename} from {image_url} ...")
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code.
        with open(image_path, "wb") as f_out:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f_out.write(chunk)
        print(f"Downloaded {filename}")
    except requests.HTTPError as e:
        # For a 404 error or other HTTP errors, log and skip.
        print(f"Error downloading {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error downloading {filename}: {e}")

print("Image download complete.")
