// Captures the video's audio as it plays, streams it to our server over a
// WebSocket, and renders fact-check results as they come back.

const CHUNK_MS = 200; // how much audio we buffer client-side before sending

const video = document.getElementById("video");
const startButton = document.getElementById("start-button");
const status = document.getElementById("status");
const resultsList = document.getElementById("results-list");
const resultsEmpty = document.getElementById("results-empty");

startButton.addEventListener("click", start, { once: true });

async function start() {
  startButton.disabled = true;
  status.textContent = "CONNECTING...";

  const audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("audio-processor.js");

  const source = audioContext.createMediaElementSource(video);
  const captureNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  source.connect(captureNode); // off to our PCM capture...
  source.connect(audioContext.destination); // ...and also still out to speakers

  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ sample_rate: audioContext.sampleRate }));
    status.textContent = "LISTENING...";
    video.play();
  });

  ws.addEventListener("message", (event) => {
    renderResult(JSON.parse(event.data));
  });

  ws.addEventListener("close", () => {
    status.textContent = "DISCONNECTED";
  });

  // Buffer incoming audio quanta until we have CHUNK_MS worth, then send as
  // one binary frame - sending every ~128-sample quantum individually would be
  // hundreds of tiny websocket messages per second.
  let buffered = [];
  let bufferedLength = 0;
  const samplesPerChunk = Math.floor((audioContext.sampleRate * CHUNK_MS) / 1000);

  captureNode.port.onmessage = (event) => {
    buffered.push(event.data);
    bufferedLength += event.data.length;
    if (bufferedLength >= samplesPerChunk) {
      const chunk = mergeFloat32(buffered, bufferedLength);
      buffered = [];
      bufferedLength = 0;
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(floatTo16BitPCM(chunk).buffer);
      }
    }
  };

  video.addEventListener("ended", () => {
    status.textContent = "DONE";
    ws.close();
  });
}

function mergeFloat32(chunks, totalLength) {
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function floatTo16BitPCM(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

function renderResult(result) {
  resultsEmpty.style.display = "none";

  const card = document.createElement("div");
  card.className = "card";

  if (!result.matched) {
    card.innerHTML = `
      <div class="assertion">${escapeHtml(result.text)}</div>
      <div class="badge">No match</div>
    `;
  } else {
    const badgeClass = `badge-${result.verdict.replace("_", "-")}`;
    const topFact = result.matches[0];
    card.innerHTML = `
      <div class="assertion">${escapeHtml(result.text)}</div>
      <div class="badge ${badgeClass}">${result.verdict.replace("_", " ")}</div>
      <div class="reasoning">${escapeHtml(result.reasoning)}</div>
      <div class="fact-label">Source fact</div>
      <div class="fact">${escapeHtml(topFact.fact)}</div>
    `;
  }

  resultsList.appendChild(card);
  const panel = resultsList.parentElement;
  panel.scrollTop = panel.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
