# Non-Destructive Acoustic Brick Quality Assessment

An automated, low-cost, non-destructive quality inspection system that evaluates building brick integrity by analyzing impact sound signatures captured via hardware micro-controllers and processed through a robust digital signal processing (DSP) and machine learning backend.

---

## 🏗️ System Architecture

The project follows a clean **Frontend-Backend separation**:
- **Backend (`brick_analysis2.ipynb` / Core Pipeline):** Handles ESP32 serial audio data acquisition, digital signal filtering, multi-domain feature extraction, and Random Forest classifier training.
- **Frontend (`brick_dashboard.py`):** An interactive **Streamlit** dashboard that triggers real-time test recordings, renders live waveforms and FFT spectra, and displays classification confidence scoring.

---

## 🛠️ Key Technical Components

### 1. Hardware & Audio Acquisition (`ESP32`)
- Connects to an ESP32 microcontroller via USB serial communication (`COM_PORT`, 115200 baud rate).
- Captures raw impact acoustic data streams, normalizes ADC readings, and saves high-fidelity `.wav` recordings for offline or real-time analysis.

### 2. Digital Signal Processing (DSP) Pipeline
- **DC Offset Removal:** Subtracts the mean waveform value to center the signal.
- **Butterworth Band-Pass Filter:** Applies a 3rd-order Butterworth band-pass filter ($20\text{ Hz} - 3500\text{ Hz}$) to eliminate environmental background noise while preserving resonant brick frequencies.
- **Amplitude Normalization:** Safely normalizes the filtered waveform to prevent volume-dependent classification skew.

### 3. Hybrid Feature Extraction
Extracts a 5-dimensional feature vector combining time-domain and frequency-domain parameters:
1. **Peak Frequency:** Dominant resonant frequency extracted using Fast Fourier Transform (FFT).
2. **Peak Magnitude:** Maximum amplitude observed in the frequency spectrum.
3. **RMS Energy:** Overall signal energy measure:
   $$x_{RMS} = \sqrt{\frac{1}{N}\sum_{n=1}^{N}x(n)^2}$$
4. **Zero-Crossing Rate (ZCR):** Rate of sign changes in the time-domain signal:
   $$ZCR = \frac{1}{2N}\sum_{n=1}^{N}|\text{sgn}[x(n)] - \text{sgn}[x(n-1)]|$$
5. **Spectral Centroid:** Center of mass of the frequency spectrum representing acoustic brightness:
   $$C_t = \frac{\sum_{k=1}^{M}f(k)M_t(k)}{\sum_{k=1}^{M}M_t(k)}$$

### 4. Machine Learning Classification
- Trained on extracted feature vectors using an ensemble **Random Forest Classifier** ($n\_estimators=200$, $max\_depth=10$).
- Evaluated using stratified train-test splits and visualized via confusion matrices, achieving strong validation accuracy in distinguishing structural brick quality.

---

## 🚀 Quick Start & Installation

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the interactive Streamlit dashboard:
   ```bash
   streamlit run frontend/brick_dashboard.py
   ```
