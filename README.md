# 🇧🇩 Bangladeshi Banknote Recognition System (BDT)

<p align="center">
  <img src="https://img.shields.io/badge/Vercel-Ready-black?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel">
  <img src="https://img.shields.io/badge/ONNX_Web-1.18%2B-blue?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime Web">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>An ultra-fast, real-time Bangladeshi currency (Taka / ৳) detection and classification system powered by Deep Learning, MobileNetV3-Large, and OpenCV.</strong>
</p>

---

## 📌 Overview

The **Bangladeshi Banknote Recognition System** is an intelligent computer vision solution engineered to identify and classify Bangladeshi currency notes in real time using a standard webcam or video feed. Built with lightweight yet highly accurate neural network architecture (**MobileNetV3-Large**), the system performs low-latency inference on both CPU and GPU edge devices.

This project is tailored for:
- 👁️ **Assistive Technologies**: Providing rapid, automated currency recognition for visually impaired individuals.
- 🏪 **Point of Sale (POS) & Smart Kiosks**: Real-time cash acceptance, sorting, and verification assistance.
- ⚡ **Edge AI Prototyping**: Resource-efficient deployment on embedded devices (laptops, mini PCs, single-board computers).

---
<img width="1276" height="592" alt="image" src="https://github.com/user-attachments/assets/ffa65e72-d01f-4076-852b-d48edf1866a0" />

## 💵 Supported Denominations

The system is trained to recognize all **9 active Bangladeshi Taka (BDT)** banknote denominations currently in circulation:

| Denomination | Common Color Tone | Bengali Label | Status |
| :---: | :---: | :---: | :---: |
| **৳ 2** | Green / Peach | দুই টাকা | Supported ✅ |
| **৳ 5** | Grayish Purple / Bronze | পাঁচ টাকা | Supported ✅ |
| **৳ 10** | Violet / Pinkish Purple | দশ টাকা | Supported ✅ |
| **৳ 20** | Olive Green / Orange | বিশ টাকা | Supported ✅ |
| **৳ 50** | Light Orange / Brown | পঞ্চাশ টাকা | Supported ✅ |
| **৳ 100** | Sky Blue / Blue | একশত টাকা | Supported ✅ |
| **৳ 200** | Green / Golden Yellow | দুইশত টাকা | Supported ✅ |
| **৳ 500** | Emerald Green / Cyan | পাঁচশত টাকা | Supported ✅ |
| **৳ 1000** | Ruby Red / Magenta | এক হাজার টাকা | Supported ✅ |

---

## 📊 Dataset

The model is trained on the Bangladeshi banknote dataset available on Kaggle:
- **Dataset Link**: [CSE475 Group-J Banknote Dataset (Kaggle)](https://www.kaggle.com/datasets/nadimsrabon6/cse475-groupj-dataset)

---

## ✨ Key Features

- **⚡ High-Throughput Real-Time Inference**:
  Direct video buffer optimization (`CAP_PROP_BUFFERSIZE = 1`) and PyTorch `@torch.inference_mode()` ensure ultra-low latency and maximum frame rates.

- **🎯 Guided Region of Interest (ROI)**:
  Includes a targeted bounding guide to frame notes comfortably and reject background clutter. Toggle seamlessly between target box and full-frame mode with a single keypress.

- **🌊 Temporal Moving-Average Smoothing**:
  Maintains a sliding temporal probability window (`collections.deque`) across consecutive frames to prevent jitter, false-positive flickers, and fluctuating confidence scores.

- **📊 Live Heads-Up Display (HUD)**:
  - Real-time FPS telemetry.
  - Active detection confidence percentage.
  - Live **Top-3 class likelihood distribution** breakdown.
  - Visual status color indicators (Green for confident detection, Orange for searching/low confidence).

- **🎮 Dynamic Interactive Controls**:
  Adjust confidence thresholds, toggle mirror mode, swap ROI modes, and snapshot frames directly during the live camera session.

---

## 🛠️ Architecture & Pipeline

```text
  [ Webcam / Camera Feed ]
             │
             ▼
  [ Frame Acquisition (OpenCV) ]
             │
       (Mirror Flip / ROI Crop)
             │
             ▼
  [ Preprocessing & Normalization ]
    • BGR -> RGB conversion
    • Resize to (224 x 224)
    • Standard Mean / Std Normalization
             │
             ▼
  [ MobileNetV3-Large Model (PyTorch / timm) ]
             │
             ▼
  [ Softmax Probabilities ]
             │
             ▼
  [ Rolling Moving-Average Queue ] (Temporal Smoothing)
             │
             ▼
  [ Threshold Verification & UI Rendering ]
             │
             ▼
  [ Real-Time Annotated Video Display + HUD ]
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.9, 3.10, or 3.11 recommended.
- A functional webcam or external USB camera.
- *Optional*: NVIDIA GPU with CUDA drivers installed (for hardware-accelerated inference).

### 1. Clone or Navigate to the Repository

```bash
cd "BD Note detection"
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages via `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note on PyTorch (CUDA GPU Acceleration):**  
> If you have an NVIDIA GPU and wish to leverage CUDA acceleration, install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 💻 Usage

To launch the real-time detection application with default settings:

```bash
python realtime_detect.py
```

### Command-Line Arguments

Customize the camera index, input resolution, model path, or smoothing filter using CLI flags:

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--model` | `str` | `best_banknote_model.pth` | Path to the trained model checkpoint (`.pth`). |
| `--cam` | `int` | `0` | Camera device index (`0` for primary/integrated, `1` for external). |
| `--width` | `int` | `1280` | Frame capture width in pixels. |
| `--height` | `int` | `720` | Frame capture height in pixels. |
| `--threshold`| `float` | `0.75` | Minimum confidence score required to confirm a banknote class. |
| `--smooth` | `int` | `6` | Number of sequential frames to average for temporal stabilization. |

#### Example Commands

Run on an external webcam (index 1) with custom resolution and higher confidence:
```bash
python realtime_detect.py --cam 1 --width 1920 --height 1080 --threshold 0.85
```

Run with aggressive smoothing for exceptionally stable output:
```bash
python realtime_detect.py --smooth 10 --threshold 0.80
```

---

## ⌨️ Interactive Keyboard Shortcuts

While the detection window is focused, you can control the application in real time using the following hotkeys:

| Key | Action | Description |
| :---: | :--- | :--- |
| <kbd>q</kbd> / <kbd>Esc</kbd> | **Quit** | Safely exits the video feed and closes all OpenCV windows. |
| <kbd>r</kbd> | **Toggle ROI** | Switches between central **Region of Interest box** and **Full Frame** mode. |
| <kbd>m</kbd> | **Mirror Mode** | Toggles horizontal flip on/off (convenient for selfie webcams). |
| <kbd>+</kbd> / <kbd>=</kbd> | **Increase Threshold** | Raises the confidence threshold by `+0.05` (up to `0.99`). |
| <kbd>-</kbd> / <kbd>_</kbd> | **Decrease Threshold** | Lowers the confidence threshold by `-0.05` (down to `0.30`). |
| <kbd>s</kbd> | **Snapshot** | Captures and saves the current annotated frame as `capture_XXX.jpg`. |

---

## 🧠 Model Specifications

The packaged model (`best_banknote_model.pth`) contains the complete configuration and calibrated weights:

- **Backbone**: `mobilenetv3_large_100` ([timm](https://github.com/huggingface/pytorch-image-models))
- **Dataset**: [CSE475 Group-J Dataset (Kaggle)](https://www.kaggle.com/datasets/nadimsrabon6/cse475-groupj-dataset)
- **Parameters / Size**: ~17 MB (compact footprint ideal for mobile and edge distribution)
- **Input Resolution**: $224 \times 224$ pixels, 3 channels (RGB)
- **Normalization**: Pre-computed dataset channel means and standard deviations embedded in the checkpoint
- **Inference Mode**: Evaluated in strict `@torch.inference_mode()` with automated CPU/CUDA allocation

---

## 📁 Repository Structure

```text
BD Note detection/
├── .gitignore                 # Git ignore rules for Python, cache, and OS files
├── LICENSE                    # MIT open-source license
├── README.md                  # Comprehensive project documentation
├── app.js                     # Mobile web app logic & ONNX Runtime Web inference
├── best_banknote_model.pth    # Pre-trained MobileNetV3 PyTorch checkpoint (~17 MB)
├── export_onnx.py             # Script to export PyTorch weights to ONNX
├── index.html                 # Mobile-first responsive web UI for Vercel
├── manifest.json              # PWA manifest for mobile home-screen install
├── model.onnx                 # Browser-optimized ONNX model (~16 MB)
├── realtime_detect.py         # Main desktop real-time OpenCV detection script
├── requirements.txt           # Python dependencies & library versions
├── style.css                  # Modern dark-mode UI stylesheet
└── vercel.json                # Vercel deployment & cache optimization config
```

---

## 🌐 Deploy to Vercel (Mobile Version)

You can host this project on **Vercel** with **zero server cost** and run real-time banknote detection directly on your mobile device (Android / iOS):

### How It Works on Mobile:
- **Client-Side AI**: Powered by [ONNX Runtime Web](https://onnxruntime.ai/), the neural network runs directly inside your phone's browser using WebAssembly / WebGL.
- **Zero Latency**: No video frames are transmitted to a server. All inference occurs privately and instantly on your device.
- **Audio Voice Readout**: Automatically speaks the detected denomination aloud (e.g. *"500 Taka"*) using the Web Speech API.
- **Installable PWA**: Tap **"Add to Home Screen"** on your mobile browser to use it like a native mobile app.

### Deploy Steps:

1. **Push your repository to GitHub** (already configured).
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New Project"**.
3. Import your **`Taka-Vision`** repository.
4. Keep the Framework Preset as **`Other`** (Root Directory: `./`).
5. Click **"Deploy"**!
6. Once deployed, open the provided `.vercel.app` link on your smartphone's camera browser and grant camera permissions.

---

## 🔮 Roadmap & Future Enhancements

- [ ] **Text-to-Speech (TTS) Integration**: Add dual-language voice readout (Bengali and English) announcing note values for visually impaired accessibility.
- [ ] **ONNX & TensorRT Export**: Export model to ONNX and TensorRT for sub-5ms latency on edge platforms like Raspberry Pi 5 and NVIDIA Jetson.
- [ ] **Dual-side Validation**: Detection of front vs. back note orientations.
- [ ] **Security Feature Verification**: Identification of watermark, latent security threads, and holographic stripes.
- [ ] **Cross-Platform Mobile App**: Porting the pipeline to Flutter / PyTorch Mobile.

---

## 🤝 Contributing

Contributions, feedback, and suggestions are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

<p align="center">
  Developed with ❤️ for Computer Vision and Assistive AI in Bangladesh 🇧🇩
</p>

