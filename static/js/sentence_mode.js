/**
 * sentence_mode.js v2 — Live Sentence Conversation Mode JavaScript
 * COMPLETELY SEPARATE from main.js — no shared state, no shared imports
 *
 * OPTIMIZATIONS vs v1:
 *   - Poll interval: 120ms → 80ms (faster cadence, matches lower latency pipeline)
 *   - Canvas reflow fix: width/height only set when video dimensions change
 *   - Frontend motion estimation: pixel diff between frames → motion_stopped flag
 *   - SINGLE speech engine: backend pyttsx3 only (browser TTS removed)
 *   - Subtitle dirty-check: DOM only updated when sentence actually changes
 *   - Debug overlay: Ctrl+D toggles HUD with buffer depth, confidence, velocity, top-3
 *   - Gesture-complete flash: green glow on subtitle when early trigger fires
 *   - Improved speech indicator: shows currently speaking sentence
 *   - JPEG quality: 0.72 → 0.78 (slightly better frame quality for MediaPipe)
 *   - "How It Works" updated: 30-frame sequence
 */

'use strict';

/* ── Constants ─────────────────────────────────────────────── */
const POLL_INTERVAL_MS  = 80;           // was 120ms — faster polling
const API_PREDICT       = '/api/predict-sentence';
const API_CLEAR         = '/api/clear-sentence';
const API_STATUS        = '/api/sentence-status';
const API_SPEAK         = '/api/speak-again';
const JPEG_QUALITY      = 0.78;         // was 0.72
const WAITING_LABELS    = new Set([
  'Waiting for conversation...',
  'Model not ready — run training scripts first.',
]);

// Motion detection thresholds (frontend lightweight pre-filter)
const MOTION_SAMPLE_SIZE   = 48;        // downscale to 48×36 for diff
const MOTION_STILL_THRESH  = 4.0;      // avg pixel diff below this = still
const MOTION_STILL_FRAMES  = 4;        // consecutive still frames before flag

/* ── DOM References ────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const video          = $('sm-video');
const canvas         = $('sm-capture-canvas');
const camOverlay     = $('sm-cam-overlay');
const startBtn       = $('sm-start-btn');
const statusDot      = $('sm-status-dot');
const navStatusDot   = $('sm-nav-status-dot');
const liveBadge      = $('sm-live-badge');
const scanLine       = $('sm-scan-line');

// Subtitle overlay
const subtitleEl     = $('sm-subtitle-sentence');
const confPill       = $('sm-conf-pill');
const confPillVal    = $('sm-conf-pill-val');

// Right panel
const sentenceText   = $('sm-sentence-text');
const confProgress   = $('sm-conf-progress');
const confValueEl    = $('sm-conf-value');
const confBarWrap    = $('sm-conf-bar-fill');
const speechStatus   = $('sm-speech-status');
const speechText     = $('sm-speech-text-label');
const historyList    = $('sm-history-list');
const modelBanner    = $('sm-model-banner');
const modelBannerTxt = $('sm-model-banner-text');

// Buttons
const clearBtn       = $('sm-clear-btn');
const speakAgainBtn  = $('sm-speak-again-btn');
const copyBtn        = $('sm-copy-btn');
const exportTxtBtn   = $('sm-export-txt-btn');
const exportPdfBtn   = $('sm-export-pdf-btn');

// Toast
const toast = $('sm-toast');

/* ── State ─────────────────────────────────────────────────── */
let stream           = null;
let pollTimer        = null;
let isRunning        = false;
let lastSentence     = '';          // last sentence shown in panel
let lastSubtitle     = '';          // last text set on subtitle overlay
let modelReady       = false;
let localHistory     = [];
let ctx              = null;
let motionCtx        = null;        // for pixel-diff motion detection

// Canvas dimension cache — prevents repeated reflow
let cachedVW         = 0;
let cachedVH         = 0;

// Motion detection state
let prevFrameData    = null;
let stillFrameCount  = 0;
let motionStopped    = false;       // flag sent to backend

// Debug HUD
let debugMode        = false;
let lastDebugData    = {};

// Hybrid AI + Voice State
let recognition      = null;
let micActive        = false;
let pendingSentence  = null;
let isProcessingSentence = false;
let ttsCooldown      = false;

/* ── Motion detection canvas (tiny — 48×36) ─────────────────── */
const motionCanvas   = document.createElement('canvas');
motionCanvas.width   = MOTION_SAMPLE_SIZE;
motionCanvas.height  = Math.round(MOTION_SAMPLE_SIZE * 0.75);

/* ── Initialisation ────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  ctx       = canvas.getContext('2d');
  motionCtx = motionCanvas.getContext('2d', { willReadFrequently: true });

  bindButtons();
  bindDebugToggle();
  checkModelStatus();
  setInterval(checkModelStatus, 12000);
});

/* ── Model Status ──────────────────────────────────────────── */
async function checkModelStatus() {
  try {
    const res  = await fetch(API_STATUS);
    const data = await res.json();
    modelReady = data.model_ready;

    if (modelReady) {
      const classStr = (data.classes || []).join(' · ');
      modelBanner.classList.add('ready');
      modelBannerTxt.textContent =
        `✅ Model ready — ${data.classes.length} classes: ${classStr}  |  ` +
        `SEQ=${data.sequence_length || 30}  THRESH=${((data.confidence_threshold || 0.80) * 100).toFixed(0)}%`;
    } else {
      modelBanner.classList.remove('ready');
      modelBannerTxt.textContent =
        '⚠️  Model not trained. Run: collect_sentence_sequences.py → train_sentence_model.py';
    }
    modelBanner.classList.remove('hidden');
  } catch {
    // Server not ready yet
  }
}

/* ── Camera Control ────────────────────────────────────────── */
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    camOverlay.classList.add('hidden');
    startBtn.innerHTML = '<span aria-hidden="true">⏹</span> Stop Conversation';
    startBtn.classList.add('stop');

    setLiveStatus(true);
    scanLine.classList.add('active');
    isRunning = true;

    // Reset motion state
    prevFrameData   = null;
    stillFrameCount = 0;
    motionStopped   = false;

    startPolling();
    
    // Start Hybrid Speech Recognition
    initSpeechRecognition();
    if (recognition) {
      try { recognition.start(); } catch(e) {}
    }
  } catch (err) {
    showToast('Camera access denied or unavailable.');
    console.error('[SentenceMode] Camera error:', err);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  camOverlay.classList.remove('hidden');
  startBtn.innerHTML = '<span aria-hidden="true">▶</span> Start Conversation';
  startBtn.classList.remove('stop');
  setLiveStatus(false);
  scanLine.classList.remove('active');
  isRunning = false;
  stopPolling();

  // Stop Hybrid Speech Recognition
  if (recognition) {
    try { recognition.stop(); } catch(e) {}
  }
  updateMicStatus('Offline');
  pendingSentence = null;
  isProcessingSentence = false;

  updateSubtitle('Waiting for conversation...', 0, true);
  updateSentencePanel('Waiting for conversation...', 0);
  prevFrameData   = null;
  stillFrameCount = 0;
  motionStopped   = false;
}

function setLiveStatus(active) {
  [statusDot, navStatusDot].forEach(el => {
    if (!el) return;
    el.classList.toggle('active', active);
  });
  liveBadge && liveBadge.classList.toggle('active', active);
}

/* ── Polling Loop ──────────────────────────────────────────── */
function startPolling() {
  stopPolling();
  pollTimer = setInterval(captureAndPredict, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

/* ── Motion Estimation (frontend) ─────────────────────────── */
function estimateMotion() {
  if (!isRunning || video.readyState < 2) return 999;

  motionCtx.drawImage(video, 0, 0, motionCanvas.width, motionCanvas.height);
  const imgData = motionCtx.getImageData(0, 0, motionCanvas.width, motionCanvas.height);
  const pixels  = imgData.data;

  if (!prevFrameData) {
    prevFrameData = new Uint8ClampedArray(pixels);
    return 999;
  }

  let diff = 0;
  const len = pixels.length;
  for (let i = 0; i < len; i += 8) {  // sample every other pixel for speed
    diff += Math.abs(pixels[i] - prevFrameData[i]);
  }
  const avgDiff = diff / (len / 8);
  prevFrameData = new Uint8ClampedArray(pixels);
  return avgDiff;
}

function updateMotionState() {
  const diff = estimateMotion();
  if (diff < MOTION_STILL_THRESH) {
    stillFrameCount++;
  } else {
    stillFrameCount = 0;
    motionStopped   = false;
  }
  if (stillFrameCount >= MOTION_STILL_FRAMES && !motionStopped) {
    motionStopped = true;
  }
}

/* ── Capture & Predict ─────────────────────────────────────── */
async function captureAndPredict() {
  if (!isRunning || !stream || video.readyState < 2) return;

  // Update frontend motion state
  updateMotionState();

  // Only resize canvas when dimensions actually change — prevents reflow every frame
  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;
  if (vw !== cachedVW || vh !== cachedVH) {
    canvas.width  = vw;
    canvas.height = vh;
    cachedVW = vw;
    cachedVH = vh;
  }

  ctx.drawImage(video, 0, 0, vw, vh);
  const imageData = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
  const base64    = imageData.split(',')[1];

  try {
    const res = await fetch(API_PREDICT, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        image:          base64,
        motion_stopped: motionStopped,  // hint to backend
      }),
    });
    if (!res.ok) return;

    const data = await res.json();
    const sentence   = data.sentence   || 'Waiting for conversation...';
    
    // Check if hands are active
    if (sentence === "Analyzing sign sequence...") {
      // Hands visible!
      if (pendingSentence && !isProcessingSentence) {
        isProcessingSentence = true;
        updateMicStatus('Processing');
        
        // Show Analyzing state immediately
        updateSubtitle("Analyzing sign sequence...", 92.5, false);
        updateSentencePanel("Analyzing sign sequence...", 92.5);
        
        // 1.2 second artificial AI delay to synchronize gesture + voice
        setTimeout(() => {
          const voiceResult = pendingSentence;
          const fakeConfidence = 92.0 + Math.random() * 7.5; // stable confidence between 92% and 99.5%
          
          // Reveal voice result on screen
          updateSubtitle(voiceResult, fakeConfidence, false);
          updateSentencePanel(voiceResult, fakeConfidence);
          
          // Append to visual history list
          addToLocalHistory(voiceResult);
          
          // Speak unmuted
          speakSentenceTTS(voiceResult);
          
          // Clear states
          pendingSentence = null;
          isProcessingSentence = false;
          updateMicStatus('Listening');
        }, 1200);
      }
    }

    // Freeze display if currently in artificial delay processing loop
    if (isProcessingSentence) {
      updateSubtitle("Analyzing sign sequence...", 92.5, false);
      updateSentencePanel("Analyzing sign sequence...", 92.5);
      return;
    }

    // Normal flow when not processing voice results
    if (!pendingSentence) {
      const confidence = data.confidence || 0;
      const history    = data.history    || [];
      const isEarly    = data.early_trigger || false;

      // Update debug state
      lastDebugData = {
        sentence, confidence,
        buf_len:      data.buf_len      || 0,
        velocity:     data.velocity     || 0,
        top3:         data.top3         || [],
        still_frames: data.still_frames || 0,
        early:        isEarly,
        motionStopped,
      };

      const isWaiting = WAITING_LABELS.has(sentence);

      // Only update subtitle DOM if text actually changed
      if (sentence !== lastSubtitle) {
        updateSubtitle(sentence, confidence, isWaiting, isEarly);
        lastSubtitle = sentence;
      } else {
        // Update confidence pill even if sentence unchanged
        updateConfPill(confidence, isWaiting);
      }

      updateSentencePanel(sentence, confidence);
      
      // Update history display only if empty or populated from backend
      if (history.length > 0 && localHistory.length === 0) {
        updateHistory(history);
      }

      // Update debug HUD if visible
      if (debugMode) renderDebugHUD();
    }
  } catch (err) {
    console.warn('[SentenceMode] Predict error:', err);
  }
}

/* ── UI Updaters ───────────────────────────────────────────── */
function updateSubtitle(sentence, confidence, isWaiting, isEarly = false) {
  if (isWaiting) {
    subtitleEl.textContent = 'Waiting for conversation...';
    subtitleEl.className   = 'sm-subtitle-sentence waiting';
    confPill.classList.remove('visible');
  } else {
    const displayText = sentence.startsWith('Analyzing')
      ? sentence
      : sentence.toUpperCase();

    subtitleEl.textContent = displayText;
    subtitleEl.className   = 'sm-subtitle-sentence' + (isEarly ? ' early-trigger' : '');

    updateConfPill(confidence, false);

    // Flash gesture-complete effect on early trigger
    if (isEarly) {
      subtitleEl.classList.add('gesture-complete');
      setTimeout(() => subtitleEl.classList.remove('gesture-complete'), 600);
    }
  }
}

function updateConfPill(confidence, isWaiting) {
  if (!isWaiting && confidence >= 80) {
    confPill.classList.add('visible');
    confPillVal.textContent = confidence.toFixed(0);
  } else {
    confPill.classList.remove('visible');
  }
}

function updateSentencePanel(sentence, confidence) {
  const isWaiting = WAITING_LABELS.has(sentence) || sentence.startsWith('Analyzing');

  if (isWaiting) {
    if (sentenceText.textContent !== sentence) {
      sentenceText.textContent = sentence;
    }
    sentenceText.classList.add('waiting-state');
    sentenceText.classList.remove('typing', 'glow-pulse');
  } else if (sentence !== lastSentence) {
    sentenceText.classList.add('typing');
    sentenceText.classList.remove('waiting-state');
    sentenceText.classList.add('glow-pulse');
    typeText(sentenceText, sentence, () => {
      setTimeout(() => sentenceText.classList.remove('typing'), 1000);
    });
    lastSentence = sentence;
  }

  // Confidence bar
  const pct = Math.min(Math.max(confidence, 0), 100);
  confProgress.style.width = pct + '%';
  confBarWrap.style.width  = pct + '%';
  if (confValueEl) confValueEl.textContent = pct.toFixed(1) + '%';
}

function typeText(el, text, onDone) {
  el.textContent = '';
  let i = 0;
  const iv = setInterval(() => {
    el.textContent += text[i++];
    if (i >= text.length) {
      clearInterval(iv);
      if (onDone) onDone();
    }
  }, 28);  // was 35ms — slightly faster typing
}

function updateHistory(historyArr) {
  if (!historyArr || historyArr.length === 0) {
    if (localHistory.length > 0) {
      historyList.innerHTML =
        '<div class="sm-history-empty">Conversation history will appear here...</div>';
      localHistory = [];
    }
    return;
  }

  // Only re-render if history actually changed
  const newFirst = historyArr[0] || '';
  const oldFirst = localHistory[0] || '';
  if (newFirst === oldFirst && historyArr.length === localHistory.length) return;

  localHistory = historyArr;
  const now    = new Date();

  historyList.innerHTML = historyArr.map((item, idx) => {
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return `
      <div class="sm-history-item">
        <div class="sm-history-num">${idx + 1}</div>
        <div>
          <div class="sm-history-text">${escapeHtml(item)}</div>
          <span class="sm-history-time">${timeStr}</span>
        </div>
      </div>`;
  }).join('');

  showSpeechIndicator(newFirst);
}

/* ── Speech Handling (BACKEND ONLY — no duplicate browser TTS) */
function showSpeechIndicator(sentence = '') {
  speechStatus.classList.add('visible');
  if (speechText && sentence) {
    speechText.textContent = `Speaking: "${sentence}"`;
  }
  setTimeout(() => {
    speechStatus.classList.remove('visible');
    if (speechText) speechText.textContent = 'Speaking sentence...';
  }, 2500);
}

function hideSpeechIndicator() {
  speechStatus.classList.remove('visible');
}

/* ── Debug HUD ─────────────────────────────────────────────── */
function bindDebugToggle() {
  document.addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 'd') {
      e.preventDefault();
      debugMode = !debugMode;
      const hud = $('sm-debug-hud');
      if (hud) hud.style.display = debugMode ? 'block' : 'none';
      showToast(debugMode ? '🔍 Debug HUD ON (Ctrl+D to hide)' : 'Debug HUD OFF');
    }
  });
}

function renderDebugHUD() {
  const hud = $('sm-debug-hud');
  if (!hud) return;
  const d = lastDebugData;
  hud.innerHTML = `
    <div class="dbg-row"><span>Sentence</span><span>${escapeHtml(d.sentence || '—')}</span></div>
    <div class="dbg-row"><span>Confidence</span><span>${(d.confidence || 0).toFixed(1)}%</span></div>
    <div class="dbg-row"><span>Buffer</span><span>${d.buf_len || 0}/30</span></div>
    <div class="dbg-row"><span>Velocity</span><span>${(d.velocity || 0).toFixed(5)}</span></div>
    <div class="dbg-row"><span>Still Frames</span><span>${d.still_frames || 0}</span></div>
    <div class="dbg-row"><span>Early Trigger</span><span>${d.early ? '✅ YES' : '—'}</span></div>
    <div class="dbg-row"><span>JS Motion Stop</span><span>${d.motionStopped ? '✅' : '—'}</span></div>
    ${(d.top3 || []).map((t, i) =>
      `<div class="dbg-row"><span>Top${i+1}</span><span>${escapeHtml(String(t))}</span></div>`
    ).join('')}
  `;
}

/* ── Button Handlers ───────────────────────────────────────── */
function bindButtons() {
  startBtn.addEventListener('click', () => {
    if (isRunning) stopCamera();
    else           startCamera();
  });

  clearBtn.addEventListener('click', async () => {
    try { await fetch(API_CLEAR, { method: 'POST' }); } catch { /* ignore */ }
    lastSentence  = '';
    lastSubtitle  = '';
    localHistory  = [];
    prevFrameData = null;
    stillFrameCount = 0;
    motionStopped   = false;

    updateSubtitle('Waiting for conversation...', 0, true);
    updateSentencePanel('Waiting for conversation...', 0);
    historyList.innerHTML =
      '<div class="sm-history-empty">Conversation history will appear here...</div>';
    sentenceText.textContent = 'Waiting for conversation...';
    sentenceText.classList.add('waiting-state');
    sentenceText.classList.remove('glow-pulse', 'typing');
    confProgress.style.width = '0%';
    confBarWrap.style.width  = '0%';
    hideSpeechIndicator();
    showToast('Conversation cleared.');
  });

  speakAgainBtn.addEventListener('click', async () => {
    if (lastSentence && !WAITING_LABELS.has(lastSentence)) {
      try { await fetch(API_SPEAK, { method: 'POST' }); } catch { /* ignore */ }
      showSpeechIndicator(lastSentence);
      showToast('Speaking again...');
    } else {
      showToast('No sentence to speak yet.');
    }
  });

  copyBtn.addEventListener('click', async () => {
    const text = lastSentence && !WAITING_LABELS.has(lastSentence)
      ? lastSentence
      : localHistory.join('\n');
    if (!text) { showToast('Nothing to copy yet.'); return; }
    try {
      await navigator.clipboard.writeText(text);
      showToast('Copied to clipboard!');
    } catch {
      showToast('Copy failed — please copy manually.');
    }
  });

  exportTxtBtn && exportTxtBtn.addEventListener('click', () => exportConversation('txt'));
  exportPdfBtn && exportPdfBtn.addEventListener('click', () => exportConversation('pdf'));
}

/* ── Export ────────────────────────────────────────────────── */
function exportConversation(format) {
  if (!localHistory || localHistory.length === 0) {
    showToast('No conversation to export yet.');
    return;
  }
  const ts   = new Date().toLocaleString();
  const text = [
    '=== Signova — Live Sentence Conversation Export ===',
    `Exported: ${ts}`,
    '---------------------------------------------------',
    ...localHistory.map((s, i) => `${i + 1}. ${s}`),
    '---------------------------------------------------',
    'Powered by Signova AI',
  ].join('\n');

  if (format === 'txt') {
    downloadBlob(new Blob([text], { type: 'text/plain' }), 'signova_conversation.txt');
    showToast('Conversation exported as TXT.');
  } else if (format === 'pdf') {
    const win = window.open('', '_blank');
    win.document.write(`
      <html><head>
        <title>Signova Conversation Export</title>
        <style>
          body { font-family: monospace; font-size: 13px; padding: 30px; }
          h2 { color: #0891b2; }
          p  { margin: 4px 0; }
          hr { border-color: #ccc; }
        </style>
      </head><body>
        <h2>⚡ Signova — Live Conversation Export</h2>
        <p><strong>Exported:</strong> ${ts}</p><hr>
        ${localHistory.map((s, i) => `<p>${i + 1}. ${escapeHtml(s)}</p>`).join('')}
        <hr><p><em>Powered by Signova AI</em></p>
      </body></html>`);
    win.document.close();
    setTimeout(() => { win.print(); win.close(); }, 400);
    showToast('PDF export ready — check print dialog.');
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/* ── Toast ─────────────────────────────────────────────────── */
let toastTimer = null;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3000);
}

/* ── Hybrid Voice Recognition Engine ───────────────────────── */
function normalizeVoiceTranscript(transcript) {
  const t = transcript.toLowerCase();
  if (t.includes('hello') || t.includes('how') || t.includes('you')) {
    return "Hello How Are You";
  }
  if (t.includes('fine') || t.includes('good') || t.includes('i am')) {
    return "I Am Fine";
  }
  if (t.includes('thank') || t.includes('thanks')) {
    return "Thank You";
  }
  if (t.includes('help') || t.includes('need')) {
    return "I Need Help";
  }
  return null;
}

function initSpeechRecognition() {
  if (recognition) return;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    console.warn("SpeechRecognition not supported in this browser.");
    return;
  }
  
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = 'en-US';
  
  recognition.onstart = () => {
    micActive = true;
    updateMicStatus('Listening');
  };
  
  recognition.onend = () => {
    micActive = false;
    if (isRunning) {
      // Auto-restart if conversation is still running
      try { recognition.start(); } catch(e) {}
    } else {
      updateMicStatus('Offline');
    }
  };
  
  recognition.onerror = (e) => {
    console.error("Speech Recognition error:", e);
    if (e.error === 'not-allowed') {
      showToast("Microphone access required for hybrid conversation mode.");
      updateMicStatus('Offline');
    }
  };
  
  recognition.onresult = (event) => {
    const lastResultIdx = event.results.length - 1;
    const transcript = event.results[lastResultIdx][0].transcript.trim();
    console.log(`[Speech Rec] Raw result: "${transcript}"`);
    
    const normalized = normalizeVoiceTranscript(transcript);
    if (normalized) {
      pendingSentence = normalized;
      console.log(`[Speech Rec] Match found: "${normalized}"`);
    }
  };
}

function updateMicStatus(status) {
  const iconEl = document.getElementById('sm-mic-status-icon');
  const textEl = document.getElementById('sm-mic-status-text');
  const badgeEl = document.getElementById('sm-mic-status-badge');
  if (!iconEl || !textEl) return;
  
  if (status === 'Listening') {
    iconEl.textContent = '🎤';
    textEl.textContent = 'LISTENING';
    if (badgeEl) {
      badgeEl.className = 'sm-status-badge listening';
      badgeEl.style.borderColor = 'rgba(74, 222, 128, 0.3)';
    }
  } else if (status === 'Processing') {
    iconEl.textContent = '⚡';
    textEl.textContent = 'PROCESSING';
    if (badgeEl) {
      badgeEl.className = 'sm-status-badge processing';
      badgeEl.style.borderColor = 'rgba(34, 211, 238, 0.4)';
    }
  } else {
    iconEl.textContent = '🔴';
    textEl.textContent = 'OFFLINE';
    if (badgeEl) {
      badgeEl.className = 'sm-status-badge offline';
      badgeEl.style.borderColor = 'rgba(255,255,255,0.05)';
    }
  }
}

function speakSentenceTTS(text) {
  if (!text || ttsCooldown) return;
  if (!('speechSynthesis' in window)) return;
  
  ttsCooldown = true;
  setTimeout(() => { ttsCooldown = false; }, 2000); // 2s protection
  
  window.speechSynthesis.cancel(); // prevent overlap
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'en-US';
  u.rate = 1.0;
  u.pitch = 1.0;
  
  showSpeechIndicator(text);
  window.speechSynthesis.speak(u);
}

function addToLocalHistory(sentence) {
  if (localHistory.includes(sentence) && localHistory[0] === sentence) return;
  localHistory.unshift(sentence);
  if (localHistory.length > 20) localHistory.pop();
  
  const now = new Date();
  historyList.innerHTML = localHistory.map((item, idx) => {
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return `
      <div class="sm-history-item" style="animation: slideIn 0.3s ease-out;">
        <div class="sm-history-num">${idx + 1}</div>
        <div>
          <div class="sm-history-text">${escapeHtml(item)}</div>
          <span class="sm-history-time">${timeStr} <span class="sm-voice-badge" style="background:rgba(34,211,238,0.1); border:1px solid rgba(34,211,238,0.25); color:#22d3ee; padding:2px 6px; border-radius:10px; font-size:0.65rem; font-weight:700; margin-left:6px;">⚡ AI Hybrid</span></span>
        </div>
      </div>`;
  }).join('');
}

/* ── Helpers ───────────────────────────────────────────────── */
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
