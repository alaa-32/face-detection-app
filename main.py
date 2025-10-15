import io
import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# ----------------------------- Helpers -----------------------------
@st.cache_resource
def load_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade from " + cascade_path)
    return cascade


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 255, 0)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def pil_to_cv2(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(bgr_img: np.ndarray):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="alaa Face Detection App", page_icon="🤖", layout="centered")
st.title("🤖 Face Detection App – Powered by alaa")

with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
1. **Choose input source**: upload an image **or** take a photo with your **camera**.  
2. Use the **sidebar** to tune detection: `scaleFactor`, `minNeighbors`, rectangle **color** and **thickness**.  
3. Click **Detect faces**.  
4. **Save to disk** (in your project folder) or **Download** the processed image.
        """
    )

# ----------------------------- Sidebar Controls -----------------------------
st.sidebar.header("Detection Settings")
scale_factor = st.sidebar.slider("scaleFactor", 1.05, 1.50, 1.20, 0.01,
                                 help="Lower = more accurate & slower. Higher = faster, may miss faces.")
min_neighbors = st.sidebar.slider("minNeighbors", 3, 12, 5, 1,
                                  help="Higher = fewer false positives, but may miss faint faces.")
rect_color_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")
rect_thickness = st.sidebar.slider("Rectangle thickness (px)", 1, 10, 2)

st.sidebar.caption("Tip: If faces are missed, try lowering scaleFactor or minNeighbors.")

# ----------------------------- Source Picker -----------------------------
st.subheader("📸 Choose Input Source")
source = st.radio("Select source:", ["Upload Image", "Use Camera"], horizontal=True)

pil_input = None
if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_input = Image.open(uploaded_file)
        st.image(pil_input, caption="Original image", use_container_width=True)
else:
    camera_input = st.camera_input("Take a photo")
    if camera_input is not None:
        pil_input = Image.open(camera_input)
        st.image(pil_input, caption="Captured image", use_container_width=True)

detect_btn = st.button("🔍 Detect faces")

# ----------------------------- Detection -----------------------------
if pil_input is None:
    st.info("Upload an image or take a photo to begin.")
else:
    if detect_btn:
        face_cascade = load_cascade()

        bgr = pil_to_cv2(pil_input)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        color_bgr = hex_to_bgr(rect_color_hex)
        for (x, y, w, h) in faces:
            cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, rect_thickness)

        out_pil = cv2_to_pil(bgr)
        st.image(out_pil, caption=f"Detected faces: {len(faces)}", use_container_width=True)

        # ---------------------- Save / Download ----------------------
        st.subheader("Save / Download")
        default_name = f"faces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filename = st.text_input("Filename to save on disk (project folder)", value=default_name)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save to disk (cv2.imwrite)"):
                try:
                    save_path = os.path.join(os.getcwd(), filename)
                    ok = cv2.imwrite(save_path, cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR))
                    if ok:
                        st.success(f"Saved to: {save_path}")
                    else:
                        st.error("cv2.imwrite returned False. Check filename and permissions.")
                except Exception as e:
                    st.error(f"Save failed: {e}")

        with c2:
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download processed image",
                data=buf.getvalue(),
                file_name=default_name,
                mime="image/png"
            )


