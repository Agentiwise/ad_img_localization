import streamlit as st
import requests
import json
import base64
import io
import zipfile
import time

# --- Configuration ---
# Models as specified in your request. 
# Note: If 'openai/gpt-5' is not available, try 'openai/gpt-4o'.
ANALYSIS_MODEL = "openai/gpt-5" 
GENERATION_MODEL = "google/gemini-3-pro-image-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Helper Functions ---

def encode_image_to_base64(uploaded_file):
    """Converts a Streamlit UploadedFile to a base64 string."""
    bytes_data = uploaded_file.getvalue()
    base64_str = base64.b64encode(bytes_data).decode('utf-8')
    mime_type = uploaded_file.type
    return f"data:{mime_type};base64,{base64_str}"

def analyze_image(api_key, base64_image_url, language):
    """
    Step 1: Analyzes the image using the specified analysis model.
    """
    analysis_prompt=f"""You are a meticulous product/visual QA analyst. Your job is to scan a single product image and identify only the text that belongs to infographic/overlay/UI elements that should be localized into {language} for a {language}  product page.  Critical scope rules (read carefully)  Translate (include in the list):  Text in overlays, badges, banners, callouts, captions, labels, pointers/arrows, corner tags, stickers, footer/header bars, or any post-production infographic panel placed on top of the image.  Do NOT translate (exclude from the list):  Any text printed on the physical product or its packaging/label (e.g., bottle/jar/box/cap/handle).  Logos, brand names, model names/numbers, trademarks, certification marks (CE, UL, etc.), QR codes, barcodes, URLs, social handles, app icons.  Watermarks or photographer/brand credits.  Text that is unreadable due to resolution; mark as UNREADABLE only if it’s clearly an overlay but cannot be read (still provide position_hint, and set changed_to to "").  When unsure whether text is product-bound vs overlay: If it follows the product’s surface perspective/curvature, lighting, or material (gloss/matte/emboss), treat it as product/packaging → EXCLUDE. If it sits flat in screen space with uniform sharpness, drop shadows, or graphic shapes, treat it as overlay → INCLUDE.  Output format (strict)  Return a JSON array. Each item must have exactly three fields:  "what_text": the exact source text string as it appears (trim whitespace; preserve casing/punctuation). If the block is bullets/lines, keep line breaks using \n.  "position_hint": a concise, human-friendly locator using plain language (see allowed vocabulary below).  "changed_to": the natural, fluent {language} translation suitable for ecommerce. If the item is UNREADABLE, set to "".  No extra keys, no commentary outside the JSON.  Positioning vocabulary (use these patterns)  Compose "position_hint" using one or more of the following, separated by “ — ” if needed:  Regions: top-left, top-center, top-right, mid-left, center, mid-right, bottom-left, bottom-center, bottom-right  Containers/graphics: header banner, footer strip, round badge, ribbon, sidebar panel, callout box, sticker, corner tag  Line/sequence cues (when text spans lines/bullets): first line, second line, third line, bullet 1, bullet 2, …  Relative anchors (optional): above product, below product, left of product, right of product, over product background, next to logo (only as a landmark; do not translate the logo)  Examples:  top-left — header banner  mid-right — round badge  bottom-center — footer strip — second line  left of product — callout box — bullet 3  Be brief and unambiguous. If the same phrase appears in multiple overlay locations, list each occurrence separately with its own "position_hint". {language}  localization guidance  Tone: concise, neutral/professional ecommerce copy; avoid over-formality.  Keep brand names, product names, model numbers, trademarks, URLs in English (do not translate).  Prefer natural and easy {language} translation or word selection over literal translation; localize idioms.  Keep Arabic numerals as in source (e.g., “12”, “3.5mm”). Do not change units unless clearly part of marketing copy.  Edge cases  If an overlay contains both text and a logo, include only the text portion in "what_text"; exclude the logo.  If text is partially occluded but readable, include it; if not readable, mark as UNREADABLE with changed_to: "".  Ignore decorative letters/numbers that have no marketing meaning (unless they read as “SALE”, “NEW”, “X2”, etc.).  Quality checks before you output  Verify that no product/packaging text or logos appear in the list.  Ensure every item has a clear "position_hint" using the vocabulary above and a {language} translation (or "" if unreadable overlay)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501", # Optional: Good practice for OpenRouter
        "X-Title": "Streamlit Image App", 
    }

    # Payload based on your second example (Image Analysis)
    payload = {
        "model": ANALYSIS_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": analysis_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image_url
                        }
                    }
                ]
            }
        ],
        # Optional: Enable reasoning if the model supports it (like in your first snippet)
        # "extra_body": {"reasoning": {"enabled": True}} 
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        
        # Extract content
        if 'choices' in result and len(result['choices']) > 0:
            analysis_text = result['choices'][0]['message']['content']
            return analysis_text, None
        else:
            return None, "No choices returned from Analysis API"
            
    except Exception as e:
        return None, str(e)

def generate_image_from_analysis(api_key, analysis_text, base64_original_image):
    """
    Step 2: Generates a new image based on the analysis text and the original image.
    Uses the Gemini image preview model logic provided.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit Image App",
    }

    # Payload based on your third example (Image Generation)
    payload = {
        "model": GENERATION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f" {analysis_text}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_original_image
                        }
                    }
                ]
            }
        ],
        "modalities": ["image", "text"]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        # Logic to extract image from the special Gemini/OpenRouter response format
        if result.get("choices"):
            message = result["choices"][0]["message"]
            
            # Check for images in the message (specific to Gemini Image Preview models on OpenRouter)
            if message.get("images"):
                # Usually returns a list, we take the first one
                image_data = message["images"][0]
                image_url = image_data.get("image_url", {}).get("url")
                
                # If it's a base64 data URL, we can return it directly
                return image_url, None
            
            # Fallback: Sometimes image is in 'content' if it's a URL
            elif message.get("content"):
                 # This path is less likely for this specific model but good for safety
                return None, "Model returned text content but no 'images' field."
                
        return None, "No image generated in response"

    except Exception as e:
        return None, str(e)

# --- Main App ---

def main():
    st.set_page_config(page_title="AI Image Transformation Pipeline", layout="wide")
    st.title("AI Image Analysis & Generation")
    st.markdown(f"Using **{ANALYSIS_MODEL}** for analysis and **{GENERATION_MODEL}** for generation.")

    # Sidebar for Inputs
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenRouter API Key", type="password")
        language = st.text_input("Output language", type="default")
        uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        process_btn = st.button("Process Images", type="primary")

    # Initialize Session State to hold results
    if 'results' not in st.session_state:
        st.session_state.results = []

    # Processing Logic
    if process_btn and api_key and uploaded_files:
        st.session_state.results = [] # Clear previous results
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing image {idx + 1} of {len(uploaded_files)}: {uploaded_file.name}...")
            
            # 1. Prepare Image
            base64_img = encode_image_to_base64(uploaded_file)
            
            # 2. Analyze Image
            analysis_result, err = analyze_image(api_key, base64_img, language)
            
            if err:
                st.error(f"Error analyzing {uploaded_file.name}: {err}")
                continue
                
            # 3. Generate New Image
            status_text.text(f"Generating variation for image {idx + 1}...")
            generated_img_url, gen_err = generate_image_from_analysis(api_key, analysis_result, base64_img)
            
            if gen_err:
                st.error(f"Error generating for {uploaded_file.name}: {gen_err}")
                continue

            # 4. Store Result
            st.session_state.results.append({
                "original_name": uploaded_file.name,
                "analysis": analysis_result,
                "generated_image": generated_img_url
            })
            
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_text.text("Processing complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    # Display Results Grid
    if st.session_state.results:
        st.divider()
        st.header("Results")
        
        # Download All Button Logic
        # We need to process the base64 strings back to bytes for the ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for item in st.session_state.results:
                try:
                    # generated_image is a data URI: "data:image/png;base64,....."
                    header, encoded = item['generated_image'].split(",", 1)
                    img_data = base64.b64decode(encoded)
                    file_name = f"generated_{item['original_name']}"
                    # Ensure extension is correct based on header if needed, defaulting to input ext
                    zip_file.writestr(file_name, img_data)
                except Exception as e:
                    st.warning(f"Could not prepare {item['original_name']} for download: {e}")
        
        st.download_button(
            label="Download All Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="generated_images.zip",
            mime="application/zip"
        )

        # Grid Display
        for item in st.session_state.results:
            with st.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.caption("Original")
                    # We don't have the original bytes object anymore easily unless we stored it
                    # But we can display the name. 
                    st.text(item['original_name'])

                with col2:
                    st.caption("Generated (Gemini 3 Pro)")
                    st.image(item['generated_image'], use_container_width=True)
                
                st.divider()

    elif process_btn and not api_key:
        st.warning("Please enter your OpenRouter API Key.")

if __name__ == "__main__":
    main()