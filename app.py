import streamlit as st
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

# Load Whisper and MarianMT models
whisper_model = whisper.load_model("small")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def record_and_translate():
    samplerate = 16000
    duration = 5  # Record for 5 seconds
    st.info("Listening...")

    # Record audio using sounddevice
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wav.write("realtime_audio.wav", samplerate, audio)

    # Transcribe
    result = whisper_model.transcribe("realtime_audio.wav")
    transcribed_text = result["text"]
    st.write(f"**Transcribed:** {transcribed_text}")

    # Translate
    translated_text = translate_text(transcribed_text)
    st.write(f"**Translated:** {translated_text}")

    # Convert to audio and play
    tts = gTTS(translated_text, lang="hi")
    tts.save("translated_output.mp3")
    st.audio("translated_output.mp3", format="audio/mp3")

# Streamlit UI
st.title("Real-Time Speech Translation (English to Hindi)")

if st.button("Start Recording"):
    record_and_translate()
