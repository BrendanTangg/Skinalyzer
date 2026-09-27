import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PERFECT_CORP_API_KEY")

if not API_KEY:
    raise ValueError("PERFECT_CORP_API_KEY was not found in .env")

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

IMAGE_PATH = "test_face.jpg"


# --------------------------------------------------
# STEP 1: Ask Perfect Corp for an upload location
# --------------------------------------------------

file_url = "https://yce-api-01.makeupar.com/s2s/v2.0/file"

file_data = {
    "files": [
        {
            "content_type": "image/jpeg",
            "file_name": "test_face.jpg",
            "file_size": os.path.getsize(IMAGE_PATH)
        }
    ]
}

print("Requesting upload URL...")

response = requests.post(
    file_url,
    headers={
        **headers,
        "Content-Type": "application/json"
    },
    json=file_data
)

print("Status code:", response.status_code)

# Convert Perfect Corp's response into a Python dictionary
response_data = response.json()

# Get the information for our uploaded file
file_info = response_data["data"]["files"][0]

file_id = file_info["file_id"]

# Perfect Corp tells us how and where to upload the image
upload_request = file_info["requests"][0]
upload_url = upload_request["url"]
upload_headers = upload_request["headers"]

print("Received file ID.")
print("Uploading test_face.jpg...")


# --------------------------------------------------
# STEP 2: Upload the actual image
# --------------------------------------------------

with open(IMAGE_PATH, "rb") as image_file:
    image_data = image_file.read()

upload_response = requests.put(
    upload_url,
    headers=upload_headers,
    data=image_data
)

print("Upload status code:", upload_response.status_code)

if upload_response.status_code == 200:
    print("Image uploaded successfully!")
    print("File ID:", file_id)
else:
    print("Image upload failed.")
    print(upload_response.text)

# --------------------------------------------------
# STEP 3: Request skin texture analysis
# --------------------------------------------------

skin_analysis_url = (
    "https://yce-api-01.makeupar.com/"
    "s2s/v2.1/task/skin-analysis"
)

analysis_data = {
    "src_file_id": file_id,
    "dst_actions": [
        "hd_texture"
    ],
    "format": "json"
}

print("\nRequesting skin texture analysis...")

analysis_response = requests.post(
    skin_analysis_url,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json=analysis_data
)

print("Analysis status code:", analysis_response.status_code)

analysis_result = analysis_response.json()

print("Analysis response:")
print(analysis_result)