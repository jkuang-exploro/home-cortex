"use strict";

// Media goes directly from the user-selected source to this browser, never via the API.
const form = document.getElementById("stream-form");
const input = document.getElementById("stream-url");
const status = document.getElementById("stream-status");
const player = document.getElementById("stream-player");
const disconnect = document.getElementById("stream-disconnect");
let currentImage = null;
let firstFrameTimer = null;

function stopStream() {
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

form.addEventListener("submit", (event) => {
  event.preventDefault();
  stopStream();
  let url;
  try {
    url = new URL(input.value.trim());
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
      throw new Error("Unsupported URL");
    }
  } catch {
    status.textContent = "Enter an HTTP or HTTPS stream URL without embedded credentials.";
    return;
  }
  if (location.protocol === "https:" && url.protocol === "http:") {
    status.textContent = "This HTTPS page needs an HTTPS stream. For local development, open Vision over HTTP.";
    return;
  }
  const image = document.createElement("img");
  image.alt = "Live camera preview";
  image.referrerPolicy = "no-referrer";
  currentImage = image;
  disconnect.disabled = false;
  status.textContent = "Connecting — allow local network access if your browser asks.";
  const failed = () => {
    if (currentImage !== image) return;
    stopStream();
    status.textContent = "Stream unavailable — check that EdgeVision is running, the URL is reachable from this browser, and local network access is allowed. Then reconnect.";
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
  image.src = url.href;
});

disconnect.addEventListener("click", () => {
  stopStream();
  status.textContent = "Disconnected. Connect to resume the preview.";
});
window.addEventListener("pagehide", stopStream);
