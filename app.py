import gradio as gr
from deepface import DeepFace
import numpy as np

def analyze_face_gradio(img_path):
    """
    Given an image path, analyzes the face using DeepFace and returns the results.
    """
    
    if img_path is None:
        return "Lütfen bir fotoğraf yükleyin."
    
    try:
        analysis = DeepFace.analyze(
            img_path=img_path,
            actions=["age", "gender", "race", "emotion"],
            detector_backend="opencv",
            enforce_detection=False # Handles images with no faces
        )
        
        if not analysis:
            return "Fotoğrafta bir yüz tespit edilemedi. Lütfen başka bir fotoğraf deneyin."

        dominant_emotion = analysis[0]["dominant_emotion"]
        dominant_race = analysis[0]["dominant_race"]
        age = analysis[0]["age"]
        gender = analysis[0]["gender"]
        
        result_str = (
            f"**Yaş:** {age}\n\n"
            f"**Cinsiyet:** {gender}\n\n"
            f"**Duygu Durumu:** {dominant_emotion}\n\n"
            f"**Irk:** {dominant_race}"
        )
        
        return result_str

    except Exception as e:
        return f"Bir hata oluştu: {e}. Lütfen başka bir fotoğraf deneyin."


gr.Interface(
    fn=analyze_face_gradio, 
    inputs=gr.Image(type="filepath"), 
    outputs="text",
    title="AI Yüz Analiz Aracı",
    description="Yüz analizi yapmak için bir fotoğraf yükleyin (yaş, cinsiyet, duygu ve ırk tahmini).",
    theme=gr.themes.Soft()
).launch()