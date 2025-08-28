import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np

st.set_page_config(
    page_title="AI Yüz Analiz Aracı",
    page_icon="🧠",
    layout="wide",
)

st.header("Yüz Analiz Aracı 🧠", divider="rainbow")
st.write("Lütfen analiz etmek istediğiniz bir fotoğraf dosyasını (.png, .jpg, .jpeg) yükleyin.")

uploaded_file = st.file_uploader(
    "Dosya Yükle",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    with st.spinner("Analiz ediliyor... Lütfen bekleyin."):
        try:
            analysis = DeepFace.analyze(
                img_path=image,
                actions=["age", "gender", "race", "emotion"],
                detector_backend="opencv",
                enforce_detection=False 
            )
            
            st.subheader("Analiz Edilen Fotoğraf")
            st.image(image, channels="BGR", use_column_width=True)

            if analysis:
                st.subheader("Analiz Sonuçları 📊")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Yaş", value=analysis[0]["age"])
                    st.metric(label="Cinsiyet", value=analysis[0]["gender"])
                
                with col2:
                    st.metric(label="Duygu Durumu", value=analysis[0]["dominant_emotion"])
                    st.metric(label="Irk", value=analysis[0]["dominant_race"])

                st.success("Analiz tamamlandı!")
            
        except Exception as e:
            st.error(f"Hata oluştu: {e}. Lütfen başka bir fotoğraf deneyin.")
            st.info("Eğer hata devam ediyorsa, fotoğrafın içinde net bir yüz olduğundan emin olun.")