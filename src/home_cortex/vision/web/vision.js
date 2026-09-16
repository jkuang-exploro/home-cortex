"use strict";

// The browser only talks to Home Cortex; the server owns the fixed upstream URL.
const form = document.getElementById("stream-form");
const input = document.getElementById("vision-key");
const status = document.getElementById("stream-status");
const player = document.getElementById("stream-player");
const disconnect = document.getElementById("stream-disconnect");
let currentImage = null;
let firstFrameTimer = null;
let attempt = 0;
let pending = null;

function stopStream() {
  attempt += 1;
  if (pending) pending.abort();
  pending = null;
  clearInterval(firstFrameTimer);
  firstFrameTimer = null;
  if (currentImage) {
    currentImage.onload = null;
    currentImage.onerror = null;
    currentImage.removeAttribute("src");
    currentImage = null;
  }
  player.replaceChildren();
  disconnect.disabled = true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopStream();
  const version = attempt;
  const controller = new AbortController();
  pending = controller;
  const headers = {};
  if (input.value.trim()) headers.Authorization = `Bearer ${input.value.trim()}`;
  input.value = "";
  disconnect.disabled = false;
  status.textContent = "Connecting through Home Cortex…";
  try {
    const response = await fetch("/vision/session", {
      method: "POST", headers, credentials: "same-origin", signal: controller.signal,
    });
    if (version !== attempt) return;
    if (!response.ok) {
      const message = response.status === 401
        ? "Enter your Cortex API key to unlock the camera preview."
        : "Stream unavailable — configure VISION_STREAM_URL on the Home Cortex server.";
      stopStream();
      status.textContent = message;
      return;
    }
  } catch (error) {
    if (version !== attempt) return;
    stopStream();
    status.textContent = "Cannot reach Home Cortex. Check the server and reconnect.";
    return;
  } finally {
    delete headers.Authorization;
  }
  pending = null;
  const image = document.createElement("img");
  image.alt = "Live camera preview";
  image.referrerPolicy = "no-referrer";
  currentImage = image;
  disconnect.disabled = false;
  status.textContent = "Connecting — waiting for the server to receive camera frames.";
  const failed = () => {
    if (currentImage !== image) return;
    stopStream();
    status.textContent = "Stream unavailable — Home Cortex could not load the camera. Check VISION_STREAM_URL and start EdgeVision with --host 0.0.0.0. If your session expired, enter the API key and reconnect.";
  };
  const ready = () => {
    if (currentImage !== image || image.naturalWidth === 0) return;
    clearInterval(firstFrameTimer);
    firstFrameTimer = null;
    status.textContent = "Preview connected. If the picture stops updating, reconnect.";
  };
  image.onload = ready;
  image.onerror = failed;
  // Some browsers do not fire load until a multipart MJPEG response ends.
  // Detect the first decoded frame without fetching or reading cross-origin pixels.
  const started = Date.now();
  firstFrameTimer = setInterval(() => {
    if (image.naturalWidth > 0) ready();
    else if (Date.now() - started >= 15000) failed();
  }, 250);
  player.replaceChildren(image);
  image.src = "/vision/stream";
});

disconnect.addEventListener("click", () => {
  stopStream();
  status.textContent = "Disconnected. Connect to resume the preview.";
});
window.addEventListener("pagehide", stopStream);
