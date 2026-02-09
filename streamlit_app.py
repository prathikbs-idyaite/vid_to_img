import streamlit as st
import tempfile
import cv2
import numpy as np
from pathlib import Path

# ===== IMPORT YOUR EXISTING LOGIC =====
from vid_to_img_frame_extractor import extract_best_frames

st.set_page_config(
    page_title="Video Best Frame Extractor",
    layout="wide"
)

st.title("🎞️ Video Best Frame Extraction")
st.markdown(
    "Upload a video and automatically extract the **best quality frames** using AI."
)

# ===============================
# Sidebar Controls
# ===============================
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of frames to extract",
    min_value=1,
    max_value=10,
    value=5
)

frame_skip = st.sidebar.slider(
    "Frame skip (higher = faster)",
    min_value=1,
    max_value=10,
    value=5
)

# ===============================
# Upload Video
# ===============================
uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video:
    st.video(uploaded_video)

    if st.button("🚀 Extract Best Frames"):
        with st.spinner("Processing video… this may take a moment ⏳"):
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = Path(tmpdir) / uploaded_video.name

                with open(video_path, "wb") as f:
                    f.write(uploaded_video.read())

                frames = extract_best_frames(
                    str(video_path),
                    top_k=top_k,
                    frame_skip=frame_skip
                )

        st.success(f"✅ Extracted {len(frames)} frames")

        # ===============================
        # Display Frames
        # ===============================
        cols = st.columns(3)

        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cols[i % 3].image(
                rgb,
                caption=f"Frame {i + 1}",
                use_container_width=True
            )

        # ===============================
        # Download Frames
        # ===============================
        st.subheader("⬇️ Download Frames")

        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", frame)

            st.download_button(
                label=f"Download Frame {i + 1}",
                data=buffer.tobytes(),
                file_name=f"best_frame_{i + 1}.jpg",
                mime="image/jpeg"
            )
