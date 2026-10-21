import cv2
import pandas as pd
import streamlit as st
from processing import Params, analyze, decode_image

st.set_page_config(page_title="Oral Lesion Screening (Cancer-Detector-tool)", layout="wide")
st.title("Oral Lesion Screening and Cancer Detection ")
st.warning("Research a diagnostic tool. Any suspicious lesion needs a dentist or doctor.")

rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Upload image", "Camera snapshot", "Live camera"])
    st.header("Settings")
    p = Params()
    p.denoise = st.checkbox("Denoise (bilateral, slower)", source != "Live camera")
    p.clahe_clip = st.slider("CLAHE contrast", 1.0, 6.0, 2.0, 0.5)
    st.subheader("Red patches")
    p.red_delta = st.slider("Redness above typical tissue", 5, 60, 18)
    p.red_val_min = st.slider("Min brightness", 0, 255, 50)
    st.subheader("White patches")
    p.white_sat_max = st.slider("Max saturation", 5, 120, 45)
    p.white_val_min = st.slider("Min brightness ", 100, 255, 170)
    p.white_val_max = st.slider("Max brightness (cuts glare)", 180, 255, 245)
    st.subheader("Cleanup")
    p.kernel = st.slider("Morphology kernel", 3, 15, 5, 2)
    p.min_area_pct = st.slider("Min region size (% of image)", 0.05, 5.0, 0.3, 0.05)


def show_results(data: bytes):
    try:
        res = analyze(decode_image(data), p)
    except ValueError as e:
        st.error(str(e))
        return
    s = res.stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attention level", s["level"], f"score {s['score']}", delta_color="off")
    c2.metric("Red area", f"{s['red_pct']}%")
    c3.metric("White area", f"{s['white_pct']}%")
    c4.metric("Regions flagged", s["n_regions"])
    t1, t2, t3, t4 = st.tabs(["Result", "Enhanced", "Masks", "Regions"])
    with t1:
        a, b = st.columns(2)
        a.image(rgb(res.original), caption="Original", use_container_width=True)
        b.image(rgb(res.annotated), caption="Flagged regions (R=red, W=white)", use_container_width=True)
        _, buf = cv2.imencode(".png", res.annotated)
        st.download_button("Download annotated image", buf.tobytes(), "annotated.png", "image/png")
    with t2:
        st.image(rgb(res.enhanced), caption="After denoise + CLAHE", use_container_width=True)
    with t3:
        a, b = st.columns(2)
        a.image(res.red_mask, caption="Red mask", use_container_width=True)
        b.image(res.white_mask, caption="White mask", use_container_width=True)
    with t4:
        if res.regions:
            st.dataframe(pd.DataFrame(res.regions).drop(columns="bbox"), use_container_width=True)
            st.caption("Irregularity = 1 - mean(circularity, solidity). Higher means a more ragged border.")
        else:
            st.write("No regions above the size threshold.")


if source == "Upload image":
    up = st.file_uploader("Upload an oral photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if up:
        show_results(up.getvalue())
    else:
        st.info("Upload a well-lit, in-focus close-up to begin.")

elif source == "Camera snapshot":
    shot = st.camera_input("Take a photo (good lighting, hold steady)")
    if shot:
        show_results(shot.getvalue())

else:
    try:
        from streamlit_webrtc import webrtc_streamer, RTCConfiguration
        from live import LiveProcessor
    except ImportError:
        st.error("Live mode needs extra packages: `pip install streamlit-webrtc av`")
        st.stop()

    l1, l2, l3 = st.columns(3)
    view = l1.selectbox("Live view", ["Annotated", "Enhanced", "Red mask", "White mask"])
    mirror = l2.checkbox("Mirror (selfie view)", True)
    max_side = l3.select_slider("Processing size (px, lower = faster)", [320, 480, 640], 480)

    ctx = webrtc_streamer(
        key="oral-live",
        video_processor_factory=LiveProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"width": {"ideal": 640}}, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        # push the current sidebar settings into the running video thread
        ctx.video_processor.params = p
        ctx.video_processor.view = view
        ctx.video_processor.mirror = mirror
        ctx.video_processor.max_side = max_side
        if st.button("Capture current frame"):
            with ctx.video_processor._lock:
                frame, stats = ctx.video_processor.last_frame, ctx.video_processor.last_stats
            if frame is not None:
                st.image(rgb(frame), caption=f"Captured: {stats['level']} (score {stats['score']})")
                _, buf = cv2.imencode(".png", frame)
                st.download_button("Download capture", buf.tobytes(), "capture.png", "image/png")
    else:
        st.info("Click START and allow camera access. Settings in the sidebar apply live.")

st.caption("Teeth, glare, and lighting can trigger false white flags; shadows can trigger false red flags. "
           "Tune the sliders and treat results as a prompt for professional review.")