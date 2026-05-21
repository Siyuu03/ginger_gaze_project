const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const cameraButton = document.getElementById("cameraButton");
const captureButton = document.getElementById("captureButton");
const exitButton = document.getElementById("exitButton");
const mediaLayer = document.getElementById("mediaLayer");
const standby = document.getElementById("standby");
const insetMedia = document.getElementById("insetMedia");
const statusText = document.getElementById("statusText");
const matchText = document.getElementById("matchText");
const predictionText = document.getElementById("predictionText");
const dateText = document.getElementById("dateText");
const timeText = document.getElementById("timeText");
const captureCanvas = document.getElementById("captureCanvas");

let cameraStream = null;
let currentVideo = null;
let activeObjectUrls = [];
const typingTimers = new Map();
let lastStatusMessage = statusText.textContent;

uploadButton.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    return;
  }

  stopCamera();
  resetResultText();

  if (file.type.startsWith("image/")) {
    showImageFile(file);
    sendImageBlob(file);
  } else if (file.type.startsWith("video/")) {
    showVideoFile(file);
  } else {
    setStatus("Unsupported file type.");
  }

  fileInput.value = "";
});

cameraButton.addEventListener("click", async () => {
  resetResultText();
  clearMedia();

  try {
    // Performance: request a modest camera stream because inference uses one static frame.
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 540 },
        frameRate: { ideal: 15, max: 15 },
      },
      audio: false,
    });
    const video = document.createElement("video");
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.srcObject = cameraStream;
    currentVideo = video;
    mediaLayer.appendChild(video);
    captureButton.classList.remove("hidden");
    setStatus("CAMERA STREAM ACTIVE. CAPTURE ONE FRAME.");
  } catch (error) {
    setStatus(`Camera error: ${error.message}`);
  }
});

captureButton.addEventListener("click", () => {
  if (!currentVideo) {
    setStatus("No camera stream is active.");
    return;
  }

  captureFrameFromVideo(currentVideo, "CAMERA FRAME CAPTURED.");
});

exitButton.addEventListener("click", () => {
  stopCamera();
  clearMedia();
  resetResultText();
  captureButton.classList.add("hidden");
  setStatus("SYSTEM IDLE");
});

function showImageFile(file) {
  const imageUrl = URL.createObjectURL(file);
  showImageUrl(imageUrl);
  setStatus("IMAGE SIGNAL LOCKED. ANALYZING.");
}

function showVideoFile(file) {
  const videoUrl = URL.createObjectURL(file);
  clearMedia();
  rememberObjectUrl(videoUrl);

  const video = document.createElement("video");
  video.src = videoUrl;
  video.controls = true;
  video.muted = true;
  video.playsInline = true;
  currentVideo = video;
  mediaLayer.appendChild(video);
  setStatus("VIDEO SIGNAL LOADED. CAPTURING ONE FRAME.");

  video.addEventListener("loadedmetadata", () => {
    const captureTime = Math.min(0.5, Math.max(0, video.duration / 3));
    video.currentTime = captureTime;
  });

  // Performance: uploaded video is analyzed once, from one selected frame only.
  video.addEventListener("seeked", () => {
    captureFrameFromVideo(video, "VIDEO FRAME CAPTURED.");
  }, { once: true });

  video.load();
}

function captureFrameFromVideo(video, statusMessage) {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;

  captureCanvas.width = width;
  captureCanvas.height = height;
  const context = captureCanvas.getContext("2d");
  context.drawImage(video, 0, 0, width, height);

  captureCanvas.toBlob((blob) => {
    if (!blob) {
      setStatus("Could not capture frame.");
      return;
    }

    const imageUrl = URL.createObjectURL(blob);
    showImageUrl(imageUrl);
    setStatus(`${statusMessage} ANALYZING.`);
    sendImageBlob(blob);
  }, "image/jpeg", 0.92);
}

function showImageUrl(imageUrl) {
  clearMedia(false);
  rememberObjectUrl(imageUrl);

  const image = document.createElement("img");
  image.src = imageUrl;
  image.alt = "Captured gaze input";
  mediaLayer.appendChild(image);

  const insetImage = document.createElement("img");
  insetImage.src = imageUrl;
  insetImage.alt = "";
  insetMedia.innerHTML = "";
  insetMedia.appendChild(insetImage);
}

async function sendImageBlob(blob) {
  const formData = new FormData();
  formData.append("image", blob, "gaze_frame.jpg");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      setStatus(data.error || "Prediction failed.");
      return;
    }

    showResult(data);
  } catch (error) {
    setStatus(`Network error: ${error.message}`);
  }
}

function showResult(data) {
  const matchPercent = Math.round(data.confidence * 100);
  const timestamp = new Date(data.timestamp);
  const predictedClass = data.predicted_class || data.prediction;
  const displayLabel = formatPredictionLabel(predictedClass);

  typeText(predictionText, displayLabel);
  typeText(matchText, `${matchPercent}% MATCH`);
  typeText(dateText, `DATE ${formatDate(timestamp)}`);
  typeText(timeText, `TIME ${formatTime(timestamp)}`);
  setStatus(`REMOTE FELINE SIGNAL DETECTED: ${displayLabel}`);
}

function typeText(element, text) {
  // Performance: typewriter updates at about 14fps instead of very frequent DOM writes.
  if (typingTimers.has(element)) {
    clearInterval(typingTimers.get(element));
  }

  element.textContent = "";
  let index = 0;

  const timer = setInterval(() => {
    element.textContent += text[index];
    index += 1;

    if (index >= text.length) {
      clearInterval(timer);
      typingTimers.delete(element);
    }
  }, 70);

  typingTimers.set(element, timer);
}

function clearMedia(showStandby = true) {
  releaseObjectUrls();
  mediaLayer.innerHTML = "";
  if (showStandby) {
    mediaLayer.appendChild(standby);
  }
  insetMedia.innerHTML = "";
  currentVideo = null;
}

function resetResultText() {
  typingTimers.forEach((timer) => clearInterval(timer));
  typingTimers.clear();
  predictionText.textContent = "REAL / AI / DRAWN / IMPOSTOR";
  matchText.textContent = "--% MATCH";
  dateText.textContent = "DATE --";
  timeText.textContent = "TIME --";
}

function stopCamera() {
  if (!cameraStream) {
    return;
  }

  cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  captureButton.classList.add("hidden");
}

function setStatus(message) {
  if (message === lastStatusMessage) {
    return;
  }

  statusText.textContent = message;
  lastStatusMessage = message;
}

function rememberObjectUrl(url) {
  activeObjectUrls.push(url);
}

function releaseObjectUrls() {
  activeObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  activeObjectUrls = [];
}

function formatDate(timestamp) {
  return `${timestamp.getFullYear()}/${timestamp.getMonth() + 1}/${timestamp.getDate()}`;
}

function formatTime(timestamp) {
  const hours = String(timestamp.getHours()).padStart(2, "0");
  const minutes = String(timestamp.getMinutes()).padStart(2, "0");
  const seconds = String(timestamp.getSeconds()).padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function formatPredictionLabel(predictedClass) {
  const labels = {
    real: "REAL",
    ai: "AI-GENERATED",
    cartoon: "DRAWN / CARTOON",
    impostor: "IMPOSTOR",
  };

  return labels[predictedClass] || String(predictedClass).toUpperCase();
}
