import streamlit as st
import aiohttp
import asyncio
import json
import base64
import io
import zipfile
import time
import os
import re
import mimetypes
from requests.exceptions import RequestException

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
        st.info(f"📁 Drive folder verified: {folder['name']}")
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
# ASYNC OPENROUTER HELPERS
# -------------------------------------------------

def encode_bytes_to_base64(raw_bytes: bytes, mime_type: str) -> str:
    """Encode raw bytes to base64 data URL."""
    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode()}"


async def async_openrouter_call(session: aiohttp.ClientSession, api_key: str, payload: dict, label: str):
    """
    Async HTTP call to OpenRouter with retry logic.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit Image Localization App",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)
            await asyncio.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"{label} failed after retries: {last_error}")

# -------------------------------------------------
# ASYNC AI STEPS
# -------------------------------------------------

async def async_analyze_localization(session, api_key, image_b64, language, extra):
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
- Logos, brand names, model numbers, trademarks, certifications, addresses.
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
    r = await async_openrouter_call(session, api_key, payload, "LOCALIZATION")
    return r["choices"][0]["message"]["content"]


async def async_analyze_aspect(session, api_key, image_b64, ratio):
    payload = {
        "model": ANALYSIS_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
You are an expert visual layout and image composition specialist.

Analyze the provided image and describe how to adapt it to a {ratio} aspect ratio, while still keeping all of the original elements of the image.

Guidelines:
- Preserve the main subject, background and focal point
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
    r = await async_openrouter_call(session, api_key, payload, "ASPECT")
    return r["choices"][0]["message"]["content"]


async def async_generate_image(session, api_key, prompt, image_b64, ratio):
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

    r = await async_openrouter_call(session, api_key, payload, "GENERATION")
    return r["choices"][0]["message"]["images"][0]["image_url"]["url"]

# -------------------------------------------------
# ASYNC PARALLEL WORKER
# -------------------------------------------------

async def async_process_single_image(
    session: aiohttp.ClientSession,
    api_key: str,
    file_data: dict,
    language: str,
    extra: str,
    ratio: str | None,
    index: int,
):
    """
    Process a single image asynchronously.
    Returns result dict with timing info.
    """
    start_time = time.time()
    file_name = file_data["name"]
    
    try:
        img_b64 = encode_bytes_to_base64(file_data["bytes"], file_data["type"])

        # Run localization analysis
        localization = await async_analyze_localization(session, api_key, img_b64, language, extra)
        prompt = localization

        # Run aspect analysis if needed
        if ratio:
            aspect = await async_analyze_aspect(session, api_key, img_b64, ratio)
            prompt = f"{aspect}\n\n{localization}"

        # Generate the final image
        image_url = await async_generate_image(session, api_key, prompt, img_b64, ratio)

        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "index": index,
            "original_name": file_name,
            "original_b64": img_b64,  # Store original for side-by-side display
            "generated_image": image_url,
            "elapsed_seconds": round(elapsed, 2),
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "index": index,
            "original_name": file_name,
            "original_b64": encode_bytes_to_base64(file_data["bytes"], file_data["type"]),
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


async def process_all_images_async(
    api_key: str,
    file_data_list: list,
    language: str,
    extra: str,
    ratio: str | None,
):
    """
    Process all images in TRUE parallel using asyncio.
    """
    # Create a single session for connection pooling
    connector = aiohttp.TCPConnector(limit=MAX_WORKERS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all tasks
        tasks = [
            async_process_single_image(
                session, api_key, fd, language, extra, ratio, i
            )
            for i, fd in enumerate(file_data_list)
        ]
        
        # Run ALL tasks concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

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
    # PRE-READ ALL FILES IN MAIN THREAD
    # -------------------------------------------------
    
    file_data_list = []
    for f in files:
        file_data_list.append({
            "bytes": f.getvalue(),
            "name": f.name,
            "type": f.type,
        })
    
    num_images = len(file_data_list)
    #st.info(f"🚀 Processing {num_images} images...")

    # Clear previous results
    st.session_state.results = []

    # -------------------------------------------------
    # RUN ASYNC PROCESSING
    # -------------------------------------------------
    
    overall_start = time.time()
    
    with st.spinner(f"Processing {num_images} images..."):
        # Run the async function
        results = asyncio.run(
            process_all_images_async(
                api_key, file_data_list, language, extra, ratio
            )
        )
    
    overall_elapsed = time.time() - overall_start

    # -------------------------------------------------
    # DISPLAY RESULTS SIDE-BY-SIDE
    # -------------------------------------------------
    
    st.subheader("📊 Result")
    
    # Show timing summary
    successful = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed = [r for r in results if isinstance(r, dict) and not r.get("success")]



    st.info(f"Successful: {len(successful)}/{num_images}")

    

    st.divider()

    # Sort results by original index to maintain order
    sorted_results = sorted(
        [r for r in results if isinstance(r, dict)],
        key=lambda x: x.get("index", 0)
    )

    # Display each result side-by-side
    for result in sorted_results:
        if result.get("success"):
            st.session_state.results.append(result)
            
            st.markdown(f"### 📷 {result['original_name']} *(processed in {result['elapsed_seconds']}s)*")
            
            # Side-by-side display
            col_orig, col_gen = st.columns(2)
            
            with col_orig:
                st.markdown("**Original**")
                st.image(result["original_b64"], use_container_width=True)
            
            with col_gen:
                st.markdown("**Generated**")
                st.image(result["generated_image"], use_container_width=True)
            
            st.divider()
        else:
            st.error(f"❌ Failed: {result['original_name']} - {result.get('error', 'Unknown error')}")

    # -------------------------------------------------
    # DRIVE UPLOAD (SERIAL, AFTER ALL DONE)
    # -------------------------------------------------

    if drive_enabled and st.session_state.results:
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
    # ZIP DOWNLOAD
    # -------------------------------------------------

    if st.session_state.results:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for item in st.session_state.results:
                header, encoded = item["generated_image"].split(",", 1)
                zipf.writestr(
                    f"generated_{item['original_name']}",
                    base64.b64decode(encoded),
                )

        st.download_button(
            "📥 Download All Images (ZIP)",
            zip_buffer.getvalue(),
            "generated_images.zip",
            "application/zip",
        )

if __name__ == "__main__":
    main()