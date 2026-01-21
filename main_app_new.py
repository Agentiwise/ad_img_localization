import streamlit as st
import requests
import json
import base64
import io
import zipfile
import time
import os
import re
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException, Timeout

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

ANALYSIS_MODEL = "google/gemini-3-flash-preview"
GENERATION_MODEL = "google/gemini-3-pro-image-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3
MAX_WORKERS = 6

SERVICE_ACCOUNT_FILE = "drive_upload_service_account_key.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

ASPECT_RATIO_OPTIONS = {
    "Original": None,
    "1:1 - Square | Instagram": "1:1",
    "2:3 - Portrait": "2:3",
    "3:2 - DSLR": "3:2",
    "4:3 - Standard": "4:3",
    "4:5 - Instagram Feed": "4:5",
    "9:16 - Reels": "9:16",
    "16:9 - YouTube": "16:9",
}

# -------------------------------------------------
# GOOGLE DRIVE HELPERS
# -------------------------------------------------

def drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    #st.write("🔐 Drive authenticated as:", creds.service_account_email)
    return build("drive", "v3", credentials=creds)


def extract_folder_id(url: str) -> str | None:
    patterns = [
        r"folders/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def verify_drive_folder_access(folder_id: str, svc) -> bool:
    try:
        folder = svc.files().get(
            fileId=folder_id,
            fields="id, name"
        ).execute()
        st.write(f"📁 Drive folder verified: {folder['name']}")
        return True
    except HttpError as e:
        st.error("❌ Drive folder not accessible. Share it with the service account.")
        st.error(str(e))
        return False


def drive_upload(local_path: str, folder_id: str, svc):
    file_name = os.path.basename(local_path)
    mime = mimetypes.guess_type(local_path)[0] or "application/octet-stream"

    st.write(f"⬆️ Uploading: {file_name}")

    media = MediaFileUpload(local_path, mimetype=mime, resumable=True)
    meta = {"name": file_name, "parents": [folder_id]}

    svc.files().create(
        body=meta,
        media_body=media,
        fields="id"
    ).execute()

# -------------------------------------------------
# OPENROUTER HELPERS
# -------------------------------------------------

def encode_bytes_to_base64(raw_bytes: bytes, mime_type: str) -> str:
    """Encode raw bytes to base64 data URL."""
    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode()}"


def encode_image_to_base64(uploaded_file):
    raw = uploaded_file.getvalue()
    return f"data:{uploaded_file.type};base64,{base64.b64encode(raw).decode()}"


def robust_openrouter_call(api_key, payload, label):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit Image Localization App",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except (Timeout, RequestException) as e:
            last_error = str(e)
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"{label} failed after retries: {last_error}")

# -------------------------------------------------
# AI STEPS
# -------------------------------------------------

def analyze_localization(api_key, image_b64, language, extra):
    payload = {
        "model": ANALYSIS_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""You are a meticulous product/visual QA analyst. Your job is to scan a single product image and identify only the text that belongs to infographic/overlay/UI elements that should be localized into {language} for a {language} product page.

Critical scope rules:
Translate:
- Overlays, badges, banners, callouts, captions, labels, pointers, stickers, ribbons, corner tags, headers, footers.

Do NOT translate:
- Any text printed on the physical product or packaging.
- Logos, brand names, model numbers, trademarks, certifications.
- URLs, QR codes, barcodes, watermarks, icons.

If unreadable but clearly overlay text, mark as UNREADABLE.

Return STRICT JSON array with:
- what_text
- position_hint
- changed_to

Tone: concise, professional ecommerce copy.
Do not remove or rewrite brand names.
Preserve numerals and units.

Additional user instructions:
{extra}"""},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }],
    }
    r = robust_openrouter_call(api_key, payload, "LOCALIZATION")
    return r["choices"][0]["message"]["content"]


def analyze_aspect(api_key, image_b64, ratio):
    payload = {
        "model": ANALYSIS_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
You are an expert visual layout and image composition specialist.

Analyze the provided image and describe how to adapt it to a {ratio} aspect ratio, while still keeping all of the original elements of the image.

Guidelines:
- Preserve the main subject and focal point
- Avoid aggressive cropping of oriiginal elements, rather reposition or resize them
- Prefer intelligent expansion, repositioning, or background continuation
- Maintain natural proportions
- Ensure the image feels native to the target aspect ratio
- Do not add new objects unless required for balance

Return concise, actionable steps (3-4 bullet points max).
Plain text only.
Always add a prefix to your answerstating: Change the given imaget to {ratio} (mention the required aspect ratio) following the idea: followed by your description

just the prefix and the description no preamble or commnetary or explanation
"""},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }],
    }
    r = robust_openrouter_call(api_key, payload, "ASPECT")
    return r["choices"][0]["message"]["content"]


def generate_image(api_key, prompt, image_b64, ratio):
    payload = {
        "model": GENERATION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }],
        "modalities": ["image", "text"],
    }

    if ratio:
        payload["image_config"] = {"aspect_ratio": ratio, "image_size": "2K"}

    r = robust_openrouter_call(api_key, payload, "GENERATION")
    return r["choices"][0]["message"]["images"][0]["image_url"]["url"]

# -------------------------------------------------
# PARALLEL WORKER (THREAD-SAFE VERSION)
# -------------------------------------------------

def process_image_from_bytes(api_key, file_bytes, file_name, file_type, language, extra, ratio):
    """
    Process a single image from raw bytes (thread-safe).
    
    Args:
        api_key: OpenRouter API key
        file_bytes: Raw image bytes (read in main thread)
        file_name: Original filename
        file_type: MIME type (e.g., 'image/jpeg')
        language: Target language for localization
        extra: Additional instructions
        ratio: Target aspect ratio or None
    """
    img_b64 = encode_bytes_to_base64(file_bytes, file_type)

    localization = analyze_localization(api_key, img_b64, language, extra)
    prompt = localization

    if ratio:
        aspect = analyze_aspect(api_key, img_b64, ratio)
        prompt = f"{aspect}\n\n{localization}"

    image_url = generate_image(api_key, prompt, img_b64, ratio)

    return {
        "original_name": file_name,
        "generated_image": image_url,
    }


def process_image(api_key, file, language, extra, ratio):
    """Legacy function kept for compatibility."""
    img_b64 = encode_image_to_base64(file)

    localization = analyze_localization(api_key, img_b64, language, extra)
    prompt = localization

    if ratio:
        aspect = analyze_aspect(api_key, img_b64, ratio)
        prompt = f"{aspect}\n\n{localization}"

    image_url = generate_image(api_key, prompt, img_b64, ratio)

    return {
        "original_name": file.name,
        "generated_image": image_url,
    }

# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------

def main():
    st.set_page_config(page_title="AI Image Localization", layout="wide")
    st.title("AI Marketing Image Localization")

    if "results" not in st.session_state:
        st.session_state.results = []

    with st.sidebar:
        api_key = st.text_input("OpenRouter API Key", type="password")
        language = st.text_input("Output Language", "English")
        extra = st.text_input("Additional Instructions")
        ratio_label = st.selectbox("Aspect Ratio", ASPECT_RATIO_OPTIONS.keys())
        drive_url = st.text_input("Google Drive Folder URL (optional)")
        files = st.file_uploader(
            "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        run = st.button("Process Images", type="primary")

    if not run or not api_key or not files:
        return

    ratio = ASPECT_RATIO_OPTIONS[ratio_label]

    drive_enabled = bool(drive_url.strip())
    drive_svc = None
    folder_id = None

    if drive_enabled:
        folder_id = extract_folder_id(drive_url)
        drive_svc = drive_service()
        if not verify_drive_folder_access(folder_id, drive_svc):
            st.stop()

    # -------------------------------------------------
    # PRE-READ ALL FILES IN MAIN THREAD (CRITICAL FIX)
    # -------------------------------------------------
    # Streamlit's UploadedFile objects are NOT thread-safe.
    # We must read all bytes in the main thread before passing to workers.
    
    file_data = []
    for f in files:
        file_data.append({
            "bytes": f.getvalue(),      # Read bytes in main thread
            "name": f.name,
            "type": f.type,
        })
    
    st.info(f"🚀 Processing {len(file_data)} images in parallel (up to {min(MAX_WORKERS, len(file_data))} concurrent)...")

    progress = st.progress(0)
    status_text = st.empty()
    output_container = st.container()

    # Clear previous results
    st.session_state.results = []

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(file_data))) as executor:
        # Submit all tasks with pre-read bytes
        futures = {
            executor.submit(
                process_image_from_bytes,
                api_key,
                fd["bytes"],
                fd["name"],
                fd["type"],
                language,
                extra,
                ratio
            ): fd["name"]
            for fd in file_data
        }

        completed = 0
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                result = future.result()
                st.session_state.results.append(result)

                with output_container:
                    st.image(result["generated_image"], caption=result["original_name"])

            except Exception as e:
                st.error(f"❌ Failed to process {file_name}: {str(e)}")

            completed += 1
            progress.progress(completed / len(file_data))
            status_text.text(f"Completed {completed}/{len(file_data)} images")

    progress.empty()
    status_text.empty()
    st.success(f"✅ Processed {len(st.session_state.results)} images successfully")

    # -------------------------------------------------
    # DRIVE UPLOAD (SERIAL, AFTER ALL DONE)
    # -------------------------------------------------

    if drive_enabled:
        st.subheader("Uploading to Google Drive")
        for item in st.session_state.results:
            header, encoded = item["generated_image"].split(",", 1)
            img_bytes = base64.b64decode(encoded)

            local_path = os.path.join(
                TEMP_DIR, f"generated_{item['original_name']}"
            )
            with open(local_path, "wb") as f:
                f.write(img_bytes)

            drive_upload(local_path, folder_id, drive_svc)
            os.remove(local_path)

        st.success("✅ All images uploaded to Google Drive")

    # -------------------------------------------------
    # ZIP DOWNLOAD (CORRECT & VERIFIED)
    # -------------------------------------------------

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in st.session_state.results:
            header, encoded = item["generated_image"].split(",", 1)
            zipf.writestr(
                f"generated_{item['original_name']}",
                base64.b64decode(encoded),
            )

    st.download_button(
        "Download All Images (ZIP)",
        zip_buffer.getvalue(),
        "generated_images.zip",
        "application/zip",
    )

if __name__ == "__main__":
    main()