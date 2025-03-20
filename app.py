import streamlit as st
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import sentencepiece  # Add this to ensure tokenizer loads properly
import torch  # Add this to avoid backend errors
import os


# Load Whisper and MarianMT models
whisper_model = whisper.load_model("small")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

# Translation function
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Real-time recording + processing
def process_audio():
    st.info("Upload your audio file (MP3/WAV)")
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if audio_file is not None:
        # Save the uploaded file
        with open("input_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())

        # Transcribe using Whisper
        result = whisper_model.transcribe("input_audio.wav")
        transcribed_text = result["text"]
        st.write(f"**Transcribed:** {transcribed_text}")

        # Translate
        translated_text = translate_text(transcribed_text)
        st.write(f"**Translated:** {translated_text}")

        # Convert to audio
        tts = gTTS(translated_text, lang="hi")
        tts.save("translated_output.mp3")
        st.audio("translated_output.mp3", format="audio/mp3")

# Streamlit UI
st.title("Real-Time Speech Translation (English to Hindi)")

if st.button("Start Processing"):
    process_audio()
