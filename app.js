// Taka Vision - Browser-based Real-Time Bangladeshi Banknote Detector
// Powered by MobileNetV3-Large & ONNX Runtime Web

const CLASSES = ['2', '5', '10', '20', '50', '100', '200', '500', '1000'];
const BENGALI_LABELS = {
  '2': 'দুই টাকা',
  '5': 'পাঁচ টাকা',
  '10': 'দশ টাকা',
  '20': 'বিশ টাকা',
  '50': 'পঞ্চাশ টাকা',
  '100': 'একশত টাকা',
  '200': 'দুইশত টাকা',
  '500': 'পাঁচশত টাকা',
  '1000': 'এক হাজার টাকা'
};

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const IMG_SIZE = 224;
const SMOOTH_FRAMES = 6;

// Application State
let session = null;
let currentStream = null;
let currentFacingMode = 'environment';
let isRunning = false;
let useRoi = true;
let soundEnabled = true;
let torchActive = false;
let threshold = 0.75;

let history = [];
let lastSpokenClass = null;
let lastSpokenTime = 0;
let fps = 0;
let lastFrameTime = performance.now();

// DOM Elements
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('crop-canvas');
const ctx = canvasEl.getContext('2d', { willReadFrequently: true });

const fpsBadge = document.getElementById('fps-badge');
const statusBadge = document.getElementById('status-badge');
const roiBox = document.getElementById('roi-box');
const roiHint = document.getElementById('roi-hint');

const resultCard = document.getElementById('result-card');
const noteValue = document.getElementById('note-value');
const noteBengali = document.getElementById('note-bengali');
const noteConfidence = document.getElementById('note-confidence');
const confidenceBar = document.getElementById('confidence-bar');
const candidatesList = document.getElementById('candidates-list');

const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');

const btnFlip = document.getElementById('btn-flip');
const btnTorch = document.getElementById('btn-torch');
const btnSound = document.getElementById('btn-sound');
const btnRoi = document.getElementById('btn-roi');
const loaderOverlay = document.getElementById('loader-overlay');
const loaderStatus = document.getElementById('loader-status');

// ----------------------------------------------------
// 1. Text-to-Speech (Voice Readout)
// ----------------------------------------------------
function speakNote(note) {
  if (!soundEnabled || !('speechSynthesis' in window)) return;
  const now = Date.now();
  if (note === lastSpokenClass && (now - lastSpokenTime) < 3500) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(`${note} Taka`);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);

  lastSpokenClass = note;
  lastSpokenTime = now;
}

// ----------------------------------------------------
// 2. Camera Handling
// ----------------------------------------------------
async function startCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    audio: false,
    video: {
      facingMode: currentFacingMode,
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = currentStream;
    await new Promise(resolve => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    // Check torch capability
    const track = currentStream.getVideoTracks()[0];
    const capabilities = track.getCapabilities ? track.getCapabilities() : {};
    btnTorch.style.display = capabilities.torch ? 'flex' : 'none';
  } catch (err) {
    console.error('Camera error:', err);
    alert('Unable to access camera: ' + err.message + '\nPlease allow camera permissions in your browser.');
  }
}

// ----------------------------------------------------
// 3. Initialize AI Model (ONNX Runtime Web)
// ----------------------------------------------------
async function initModel() {
  try {
    loaderStatus.innerText = 'Loading MobileNetV3 ONNX model (~16MB)...';
    
    // Configure ONNX Runtime Web WASM paths
    if (window.ort && ort.env && ort.env.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
      ort.env.wasm.numThreads = 2;
    }

    // Attempt WebGL first, fallback to WASM
    try {
      session = await ort.InferenceSession.create('./model.onnx', {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all'
      });
      statusBadge.innerText = 'WebGL AI';
    } catch (glErr) {
      console.warn('WebGL failed, falling back to WASM:', glErr);
      session = await ort.InferenceSession.create('./model.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      statusBadge.innerText = 'WASM AI';
    }

    statusBadge.className = 'badge badge-success';
    loaderOverlay.classList.add('hidden');
    
    // Warm up model with blank tensor
    const dummy = new ort.Tensor('float32', new Float32Array(3 * IMG_SIZE * IMG_SIZE), [1, 3, IMG_SIZE, IMG_SIZE]);
    await session.run({ input: dummy });

    isRunning = true;
    requestAnimationFrame(inferenceLoop);
  } catch (err) {
    console.error('Failed to load model:', err);
    loaderStatus.innerText = 'Failed to load model: ' + err.message;
    statusBadge.innerText = 'Model Error';
  }
}

// ----------------------------------------------------
// 4. Preprocessing & Tensor Construction
// ----------------------------------------------------
function extractTensor() {
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  if (!vw || !vh) return null;

  let sx, sy, sw, sh;
  if (useRoi) {
    sw = vw * 0.55;
    sh = vh * 0.55;
    sx = (vw - sw) / 2;
    sy = (vh - sh) / 2;
  } else {
    sx = 0; sy = 0; sw = vw; sh = vh;
  }

  // Draw region to 224x224 offscreen canvas
  ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, IMG_SIZE, IMG_SIZE);
  const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

  // Planar Float32Array [1, 3, 224, 224] with ImageNet normalization
  const channelSize = IMG_SIZE * IMG_SIZE;
  const floatData = new Float32Array(3 * channelSize);

  for (let i = 0; i < channelSize; i++) {
    const r = imageData[i * 4] / 255.0;
    const g = imageData[i * 4 + 1] / 255.0;
    const b = imageData[i * 4 + 2] / 255.0;

    floatData[i] = (r - MEAN[0]) / STD[0];
    floatData[channelSize + i] = (g - MEAN[1]) / STD[1];
    floatData[channelSize * 2 + i] = (b - MEAN[2]) / STD[2];
  }

  return new ort.Tensor('float32', floatData, [1, 3, IMG_SIZE, IMG_SIZE]);
}

// ----------------------------------------------------
// 5. Main Inference Loop
// ----------------------------------------------------
async function inferenceLoop() {
  if (!isRunning) return;

  const now = performance.now();
  const delta = (now - lastFrameTime) / 1000;
  if (delta > 0) {
    fps = 0.9 * fps + 0.1 * (1 / delta);
    fpsBadge.innerText = `${Math.round(fps)} FPS`;
  }
  lastFrameTime = now;

  try {
    const tensor = extractTensor();
    if (tensor && session) {
      const output = await session.run({ input: tensor });
      const probs = output.probabilities.data;

      // Temporal moving-average smoothing
      history.push(Array.from(probs));
      if (history.length > SMOOTH_FRAMES) history.shift();

      const avgProbs = new Array(CLASSES.length).fill(0);
      for (const h of history) {
        for (let j = 0; j < CLASSES.length; j++) avgProbs[j] += h[j];
      }
      for (let j = 0; j < CLASSES.length; j++) avgProbs[j] /= history.length;

      // Sort by confidence
      const indexed = avgProbs.map((p, idx) => ({ idx, p, name: CLASSES[idx] }));
      indexed.sort((a, b) => b.p - a.p);

      const top = indexed[0];
      const isConfident = top.p >= threshold && history.length >= Math.max(2, Math.floor(SMOOTH_FRAMES / 2));

      // Update UI
      if (isConfident) {
        resultCard.className = 'result-card confident';
        roiBox.classList.add('confident');
        roiHint.innerText = `${top.name} Taka (${Math.round(top.p * 100)}%)`;
        noteValue.innerText = `৳ ${top.name} Taka`;
        noteBengali.innerText = BENGALI_LABELS[top.name] || '';
        noteConfidence.innerText = `${Math.round(top.p * 100)}%`;
        confidenceBar.style.width = `${Math.round(top.p * 100)}%`;

        speakNote(top.name);
      } else {
        resultCard.className = 'result-card searching';
        roiBox.classList.remove('confident');
        roiHint.innerText = 'Position note in box';
        noteValue.innerText = 'Scanning...';
        noteBengali.innerText = 'নোট স্ক্যান করা হচ্ছে';
        const bestGuessPct = Math.round(top.p * 100);
        noteConfidence.innerText = `${bestGuessPct}%`;
        confidenceBar.style.width = `${bestGuessPct}%`;
      }

      // Update Top-3 Breakdown
      let candidatesHTML = '';
      for (let i = 0; i < 3; i++) {
        const item = indexed[i];
        const pct = Math.round(item.p * 100);
        candidatesHTML += `
          <div class="candidate-row">
            <span class="c-name">৳ ${item.name}</span>
            <div class="c-bar-bg"><div class="c-bar" style="width: ${pct}%"></div></div>
            <span class="c-pct">${pct}%</span>
          </div>
        `;
      }
      candidatesList.innerHTML = candidatesHTML;
    }
  } catch (err) {
    console.error('Inference error:', err);
  }

  // Next frame
  requestAnimationFrame(inferenceLoop);
}

// ----------------------------------------------------
// 6. Event Listeners & Controls
// ----------------------------------------------------
// Camera Flip
btnFlip.addEventListener('click', async () => {
  currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  history = [];
  await startCamera();
});

// Flashlight (Torch)
btnTorch.addEventListener('click', async () => {
  if (!currentStream) return;
  const track = currentStream.getVideoTracks()[0];
  try {
    torchActive = !torchActive;
    await track.applyConstraints({ advanced: [{ torch: torchActive }] });
    btnTorch.classList.toggle('active', torchActive);
  } catch (e) {
    console.warn('Torch not supported:', e);
  }
});

// Voice readout toggle
btnSound.addEventListener('click', () => {
  soundEnabled = !soundEnabled;
  btnSound.classList.toggle('active', soundEnabled);
  if (!soundEnabled && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
});

// ROI box toggle
btnRoi.addEventListener('click', () => {
  useRoi = !useRoi;
  roiBox.classList.toggle('full-frame', !useRoi);
  btnRoi.classList.toggle('active', useRoi);
  history = [];
});

// Threshold slider
thresholdSlider.addEventListener('input', (e) => {
  threshold = parseInt(e.target.value, 10) / 100.0;
  thresholdVal.innerText = `${e.target.value}%`;
});

// Startup sequence
window.addEventListener('DOMContentLoaded', async () => {
  await startCamera();
  await initModel();
});

