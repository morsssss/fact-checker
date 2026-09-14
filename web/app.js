/*
 * app.js: the logic that powers the web app.
 * Captures the video's audio as it plays, streams it to our server over a
 * WebSocket, and renders fact-check results as they come back.
 */

const CHUNK_MS = 200; // how much audio we buffer client-side before sending

const video = document.getElementById("video");
const checkingSwitch = document.getElementById("checking-switch");
const resultsList = document.getElementById("results-list");
const resultsEmpty = document.getElementById("results-empty");
const uploadInput = document.getElementById("video-upload-input");
const uploadDialog = document.getElementById("upload-dialog");
const tallyTrueCount = document.getElementById("tally-true-count");
const tallyFalseCount = document.getElementById("tally-false-count");

// Reset to 0 on every page load - there's no persistence, just these two counters.
let trueCount = 0;
let falseCount = 0;

// The audio graph (and createMediaElementSource especially) can only be set
// up once per <video>, so it's built lazily on the first click and then
// reused for every later start/stop cycle - only the websocket, which is the
// actual "checking session", gets torn down and recreated each time.
let audioContext = null;
let ws = null; // the current checking session, or null when stopped

checkingSwitch.addEventListener("click", toggleChecking);

video.addEventListener("ended", stopChecking);

// The pending startChecking() call from page load is stuck waiting on this
// resume() until a user gesture happens - scoped to the video itself (not
// the whole page) so pressing its native play control is what unblocks it.
video.addEventListener("play", () => {
  if (audioContext && audioContext.state === "suspended") {
    audioContext.resume();
  }
});

uploadInput.addEventListener("change", uploadVideo);

async function uploadVideo() {
  const file = uploadInput.files[0];
  if (!file) return;

  stopChecking(); // swapping the video out from under an active session would mix old/new audio
  uploadDialog.showModal();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload-video", { method: "POST", body: formData });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Upload failed (${response.status}).`);
    }
    // Cache-bust: the filename is unchanged, so the browser would otherwise
    // keep showing the old cached video instead of the one we just swapped in.
    video.src = `video/claims.mp4?t=${Date.now()}`;
    video.load();
  } catch (err) {
    alert(`Could not use that video: ${err.message}`);
  } finally {
    uploadDialog.close();
    uploadInput.value = ""; // allow choosing the same filename again later
  }
}

async function toggleChecking() {
  if (ws) {
    stopChecking();
  } else {
    await startChecking();
  }
}

async function ensureAudioGraph() {
  if (audioContext) return;

  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("audio-processor.js");

  const source = audioContext.createMediaElementSource(video);
  const captureNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  source.connect(captureNode); // off to our PCM capture...
  source.connect(audioContext.destination); // ...and also still out to speakers

  // Buffer incoming audio quanta until we have CHUNK_MS worth, then send as
  // one binary frame - sending every ~128-sample quantum individually would be
  // hundreds of tiny websocket messages per second. This runs for the rest of
  // the page's life; it just has nothing to send whenever ws isn't open.
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
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(floatTo16BitPCM(chunk).buffer);
      }
    }
  };
}

async function startChecking() {
  checkingSwitch.disabled = true;

  await ensureAudioGraph();
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  // Held locally (not just via the outer `ws`) so the "close" handler below
  // can tell whether it's reporting on this session or a since-superseded one.
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  ws = socket;

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ sample_rate: audioContext.sampleRate }));
    if (video.paused) {
      // Swallowed: on page load, before any user gesture, the browser's
      // autoplay policy rejects this - the "first gesture anywhere" listener
      // below retries it once that gesture happens.
      video.play().catch(() => {});
    }
    checkingSwitch.setAttribute("aria-checked", "true");
    checkingSwitch.disabled = false;
  });

  socket.addEventListener("message", (event) => {
    renderResult(JSON.parse(event.data));
  });

  socket.addEventListener("close", () => {
    if (ws !== socket) return; // stopChecking() already moved on to a newer session
    ws = null;
    checkingSwitch.setAttribute("aria-checked", "false");
    checkingSwitch.disabled = false;
  });
}

function stopChecking() {
  // Stops fact-checking only - the video (already playing independently of
  // this connection) is left running. Updates the switch immediately rather
  // than waiting for the close handshake to finish, so it feels instant.
  if (!ws) return;
  ws.close();
  ws = null;
  checkingSwitch.setAttribute("aria-checked", "false");
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

// The judge LLM sometimes prefaces its reasoning with this boilerplate lead-in;
// drop it and re-capitalize so the remaining sentence still reads cleanly.
function cleanReasoning(text) {
  const stripped = text.replace(/^The known fact states that\s+/i, "");
  return stripped.charAt(0).toUpperCase() + stripped.slice(1);
}

// The video we've been testing with includes jokes, each of which generates an "unknown" case.
// Thus for now I've suppressed "unknown"s. Feel free to add them back in!
function renderResult(result) {
  if (result.verdict === "unknown")
    return;

  resultsEmpty.style.display = "none";

  const card = document.createElement("div");
  card.className = "card";

  if (!result.matched) {
    card.innerHTML = `
      <div class="assertion">${escapeHtml(result.text)}</div>
      <div class="badge">No match</div>
    `;
  } else {
    if (result.verdict === "true" || result.verdict === "likely_true") {
      tallyTrueCount.textContent = ++trueCount;
    } else if (result.verdict === "false" || result.verdict === "likely_false") {
      tallyFalseCount.textContent = ++falseCount;
    }

    const badgeClass = `badge-${result.verdict.replace("_", "-")}`;
    const topFact = result.matches[0];
    card.innerHTML = `
      <div class="assertion">${escapeHtml(result.text)}</div>
      <div class="verdict-row">
        <span class="badge ${badgeClass}">${result.verdict.replace("_", " ")}</span>
        <span class="reasoning">${escapeHtml(cleanReasoning(result.reasoning))}</span>
      </div>
      <details class="fact-toggle">
        <summary>
          <svg class="chevron" viewBox="0 0 12 12" width="10" height="10" aria-hidden="true">
            <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          Source fact
        </summary>
        <p class="fact">${renderFactCitation(topFact)}</p>
      </details>
    `;
  }

  resultsList.appendChild(card);
  const panel = resultsList.parentElement;
  panel.scrollTop = panel.scrollHeight;
}

// A matched fact, plus its source in parentheses when we have one - a link
// to it when we also have a URL, otherwise just plain text.
function renderFactCitation(fact) {
  let html = escapeHtml(fact.fact);
  if (fact.source) {
    const source = fact.url
      ? `<a href="${escapeHtml(fact.url)}" target="_blank" rel="noopener noreferrer">(${escapeHtml(fact.source)})</a>`
      : escapeHtml(fact.source);
    html += ` ${source}`;
  }
  return html;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Fact checking starts on page load rather than waiting for a click on the
// switch. Browsers won't let a fresh AudioContext produce sound (or even run
// its worklet) until a user gesture happens somewhere on the page, so this
// first pass typically leaves the video paused and the context suspended
// until the user presses play on the video themselves - at which point this
// same pending startChecking() call picks back up and finishes connecting.
startChecking();
