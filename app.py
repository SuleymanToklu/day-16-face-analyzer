import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
os.environ["OMP_NUM_THREADS"] = "1"        

import streamlit as st

try:
    import cv2
    st.write("OpenCV:", cv2.__version__)
except Exception as e:
    import traceback
    st.error("cv2 import edilemedi")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

import numpy as np
from deepface import DeepFace
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import threading
import time

st.set_page_config(
    page_title="Modern Yüz Analizi",
    page_icon="😎",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size: 3.5rem; font-weight: bold; color: #FF4B4B; text-align: center; margin-bottom: 20px; }
.sub-header { font-size: 1.5rem; color: #C0C0C0; text-align: center; margin-bottom: 40px; }
.results-title { font-size: 2rem; font-weight: bold; color: #FAFAFA; margin-bottom: 20px; }
.stMetric { background-color: #262626; border-radius: 10px; padding: 15px; margin-bottom: 10px; border: 1px solid #FF4B4B; }
</style>
""", unsafe_allow_html=True)

if 'snapshot' not in st.session_state:
    st.session_state.snapshot = None

lock = threading.Lock()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with lock:
            self.latest_frame = img
        return img
        
def perform_analysis(frame):
    try:
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,  # ✅ Buraya değişiklik
            silent=True
        )[0]

        results = {
            "age": analysis['age'],
            "gender": analysis['dominant_gender'].capitalize(),
            "emotion": analysis['dominant_emotion'].capitalize()
        }
        region = analysis['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return results, frame
    except Exception as e:
        return None, frame

st.markdown('<p class="main-title">🤖 Modern Yüz Analizi</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Canlı kamera veya fotoğraf yükleyerek anlık analiz yapın.</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷 Webcam ile Analiz", "🖼️ Fotoğraf Yükle"])

with tab1:
    col1_cam, col2_cam = st.columns([2, 1])

    with col1_cam:
        st.header("Canlı Kamera")
        
        webrtc_ctx = webrtc_streamer(
            key="face-analysis-streamer",
            video_transformer_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            if st.button("📸 Anlık Görüntü Çek ve Analiz Et"):
                frame_to_analyze = None
                for _ in range(10): 
                    if webrtc_ctx.video_transformer:
                        with lock:
                            frame_to_analyze = webrtc_ctx.video_transformer.latest_frame
                        if frame_to_analyze is not None:
                            break
                    time.sleep(0.1)
                
                if frame_to_analyze is not None:
                    st.session_state.snapshot = frame_to_analyze
                else:
                    st.warning("Kamera akışından kare alınamadı. Lütfen tekrar deneyin.")
        
    with col2_cam:
        st.markdown('<p class="results-title">Analiz Sonuçları</p>', unsafe_allow_html=True)
        results_placeholder_cam = st.container(border=True)

        if st.session_state.snapshot is not None:
            with st.spinner("Analiz ediliyor..."):
                results, image_with_box = perform_analysis(st.session_state.snapshot)
            
            with col1_cam:
                 st.subheader("Analiz Edilen Görüntü")
                 st.image(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))

            with results_placeholder_cam:
                if results:
                    st.metric(label="Tahmini Yaş", value=results.get("age", "..."))
                    st.metric(label="Tespit Edilen Cinsiyet", value=results.get("gender", "..."))
                    st.metric(label="Baskın Duygu", value=results.get("emotion", "..."))
                else:
                    st.error("Analiz edilecek yüz bulunamadı.")
            
            if st.button("Yeni Analiz Yap"):
                st.session_state.snapshot = None
                st.rerun()
        else:
             with results_placeholder_cam:
                st.metric(label="Tahmini Yaş", value="...")
                st.metric(label="Tespit Edilen Cinsiyet", value="...")
                st.metric(label="Baskın Duygu", value="...")

with tab2:
    uploaded_file = st.file_uploader("Analiz için bir fotoğraf seçin...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_analyze = cv2.imdecode(file_bytes, 1)
        
        col1_upload, col2_upload = st.columns([2, 1])
        
        with col1_upload:
            st.image(image_to_analyze, channels="BGR", caption="Yüklenen Fotoğraf")
            
        with col2_upload:
            st.markdown('<p class="results-title">Analiz Sonuçları</p>', unsafe_allow_html=True)
            results_placeholder_upload = st.container(border=True)
            
            if st.button("Fotoğrafı Analiz Et", key="analyze_upload"):
                with st.spinner("Fotoğraf analiz ediliyor..."):
                    results, image_with_box = perform_analysis(image_to_analyze.copy())

                with results_placeholder_upload:
                    if results:
                        st.metric(label="Tahmini Yaş", value=results.get("age", "..."))
                        st.metric(label="Tespit Edilen Cinsiyet", value=results.get("gender", "..."))
                        st.metric(label="Baskın Duygu", value=results.get("emotion", "..."))
                        with col1_upload:
                            st.image(image_with_box, channels="BGR", caption="Analiz Edilmiş Fotoğraf")
                    else:
                        st.error("Bu fotoğrafta bir yüz tespit edilemedi.")