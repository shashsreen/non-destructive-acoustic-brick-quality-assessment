import matplotlib.pyplot as plt
from scipy.io import wavfile
import streamlit as st
import numpy as np
import librosa
import joblib
import time

from scipy.signal import butter, lfilter

st.set_page_config(
    page_title="Brick Quality Analyzer",
    layout="wide"
)

st.title("🧱 Brick Quality Analyzer")
st.markdown("AI Based Acoustic Brick Testing System")



def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_noise_filter(data, lowcut=20.0, highcut=3500.0, fs=8000, order=3):
    """Improved bandpass filter for brick acoustic analysis."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def extract_acoustic_features(file_path):

    y, sr = librosa.load(file_path, sr=None)

    # Remove DC offset
    y = y - np.mean(y)

    # Apply filtering
    y_clean = apply_noise_filter(y, fs=sr)

    # Normalize safely
    if np.max(np.abs(y_clean)) != 0:
        y_clean = y_clean / np.max(np.abs(y_clean))

    # FFT
    fft = np.abs(np.fft.rfft(y_clean))
    freqs = np.fft.rfftfreq(len(y_clean), 1/sr)

    # Peak frequency
    peak_freq = freqs[np.argmax(fft)]

    # Peak magnitude
    peak_magnitude = np.max(fft)

    # RMS energy
    rms_energy = np.sqrt(np.mean(y_clean**2))

    # Zero crossing rate
    zero_crossings = np.mean(librosa.zero_crossings(y_clean))

    # Spectral centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=y_clean, sr=sr)
    )

    # Feature vector
    features = np.array([
        peak_freq,
        peak_magnitude,
        rms_energy,
        zero_crossings,
        spectral_centroid
    ])

    return features
import joblib
st.write("Loading model...")
model = joblib.load("brick_model.pkl")
st.write("Model loaded successfully")


def predict_brick(file_path, model):

    features = extract_acoustic_features(file_path)

    prediction = model.predict([features])[0]

    confidence = max(model.predict_proba([features])[0]) * 100

    if prediction == 1:
        print(f"GOOD BRICK ({confidence:.2f}%)")
    else:
        print(f"BAD BRICK ({confidence:.2f}%)")

    return prediction, confidence

def record_audio():
    import sounddevice as sd
    from scipy.io.wavfile import write

    fs = 48000
    duration = 3

    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='int16',
        
    )

    sd.wait()

    write("live_test.wav", fs, audio)

    return "live_test.wav"

brick_model = joblib.load("brick_model.pkl")

if st.button("START TEST"):

    countdown = st.empty()

    for i in range(3,0,-1):
        countdown.markdown(f"# {i}")
        time.sleep(1)

    countdown.markdown("### Recording...")

    audio_file = record_audio()


    prediction, confidence = predict_brick(
        audio_file,
        brick_model
    )
    st.progress(confidence/100)

    if prediction == 1:

        st.success(
            f"GOOD BRICK ({confidence:.2f}%)"
        )

    else:

        st.error(
            f"BAD BRICK ({confidence:.2f}%)"
        )

    
  # WAVEFORM
    sr, audio = wavfile.read(audio_file)

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(audio)
    ax.set_title("Recorded Impact Signal")

    st.pyplot(fig)

  # FFT
    y, sr = librosa.load(audio_file, sr=None)

    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(freqs, fft)

    ax.set_title("FFT Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    st.pyplot(fig)

  # FEATURES
    features = extract_acoustic_features(audio_file)

    st.metric("Peak Frequency", f"{features[0]:.0f} Hz")
    st.metric("RMS Energy", f"{features[2]:.4f}")
    st.metric("Spectral Centroid", f"{features[4]:.0f} Hz")