
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import time

st.set_page_config(page_title="Sound Detector", layout="centered")
st.title("Sound Detector")

@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    classesaval = model.class_map_path().numpy().decode("utf-8")
    classnames = []

    with open(classesaval, "r", encoding="utf-8") as file:
        next(file)
        for line in file:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                classnames.append(parts[2])
    return model, classnames

with st.spinner("Loading model"):
    model, classnames = load_model()



duration = 5
samplerate = 16000
totalsamples = int(samplerate * duration)

if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Listening", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("Stop Listening", use_container_width=True):
        st.session_state.running = False

status = st.empty()
results = st.empty()
if st.session_state.running:
    status.info("Listening now")

    while st.session_state.running:
        try:
            audio = sd.rec(
                totalsamples,
                samplerate=samplerate,
                channels=1,
                dtype="float32")
            sd.wait()

            waveform = audio.flatten()
            waveformtensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

            scores, embeddings, spectrogram = model(waveformtensor)

            meanscores = tf.reduce_mean(scores, axis=0).numpy()
            topindices = np.argsort(meanscores)[-5:][::-1]

            with results.container():
                st.subheader("Top 5 Predictions")

                for index in topindices:
                    soundname = classnames[index]
                    score = meanscores[index] * 100

                    st.write(f"{soundname} : {score:.2f}%")
                    st.progress(float(np.clip(score / 100, 0, 1)))

        except Exception as e:
            status.error(f"Error: {e}")
            st.session_state.running = False
            break

    status.success("Detector stopped")

else:
    status.warning("Detector is  stopped")
