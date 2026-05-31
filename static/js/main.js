// ====================================================
// SIGNOVA — CENTRALIZED STATE MANAGEMENT ARCHITECTURE
// ====================================================

const BACKEND = window.location.origin || 'http://127.0.0.1:5000';
const API = {
    predict: `${BACKEND}/api/predict`,
    signup: `${BACKEND}/api/signup`,
    login: `${BACKEND}/api/login`,
    textToSign: `${BACKEND}/api/text-to-sign`,
    translate: `${BACKEND}/api/translate`,
};

// ====================================================
// CENTRAL STATE STORE — Single source of truth
// ====================================================
const State = {
    // Auth
    auth: { user: null, busy: false },
    // Camera
    camera: { stream: null, status: 'offline' }, // offline | starting | active | error
    // Recognition
    recognition: { running: false, intervalId: null, lastLabel: '', stableCount: 0, stableLabel: '' },
    // Speech output
    speech: { lastSpoken: '', cooldown: false, cooldownTimer: null, speaking: false },
    // Translation
    translation: { busy: false },
    // Playback / voice-to-sign
    playback: { busy: false, listening: false },

    // ── UI Stabilization ──
    ui: {
        displayedLabel: '',       // currently rendered label on screen
        holdActive: false,        // true while a prediction is being held on screen
        holdTimer: null,          // timer reference for the hold duration
        holdDurationMs: 400,      // minimum ms a prediction stays displayed (optimized down from 700)
        cooldownActive: false,    // true while gesture-switch cooldown is active
        cooldownTimer: null,      // timer reference for cooldown
        cooldownMs: 300,          // minimum ms between gesture switches (optimized down from 500)
        debounceThreshold: 2,     // consecutive frames needed before accepting new label
    },

    // Transition guard — prevents conflicting operations
    _locks: new Set(),
    lock(key) { this._locks.add(key); },
    unlock(key) { this._locks.delete(key); },
    isLocked(key) { return this._locks.has(key); },

    // Full reset on logout
    reset() {
        this.auth = { user: null, busy: false };
        this.recognition = { running: false, intervalId: null, lastLabel: '', stableCount: 0, stableLabel: '' };
        this.speech = { lastSpoken: '', cooldown: false, cooldownTimer: null, speaking: false };
        this.translation = { busy: false };
        this.playback = { busy: false, listening: false };
        // Clear UI stabilization timers
        if (this.ui.holdTimer) clearTimeout(this.ui.holdTimer);
        if (this.ui.cooldownTimer) clearTimeout(this.ui.cooldownTimer);
        this.ui = {
            displayedLabel: '', holdActive: false, holdTimer: null, holdDurationMs: 400,
            cooldownActive: false, cooldownTimer: null, cooldownMs: 300, debounceThreshold: 2,
        };
        this._locks.clear();
        window.lastRecognizedText = '';
    }
};
window.lastRecognizedText = '';

// ====================================================
// DOM CACHE
// ====================================================
const $ = (id) => document.getElementById(id);

// ====================================================
// UTILITIES
// ====================================================
const typeText = (el, text) => {
    if (!el) return;
    el.innerHTML = '';
    for (let i = 0; i < text.length; i++) {
        const s = document.createElement('span');
        s.textContent = text[i] === ' ' ? '\u00A0' : text[i];
        s.style.animationDelay = `${i * 0.04}s`;
        el.appendChild(s);
    }
};

const showToast = (msg, ms = 2500) => {
    const t = $('global-toast'), tx = $('global-toast-text');
    if (!t || !tx) return;
    tx.textContent = msg;
    t.classList.remove('hidden'); t.style.opacity = '1';
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.classList.add('hidden'), 300); }, ms);
};

const showModal = (el) => { if (!el) return; el.classList.remove('hidden'); setTimeout(() => el.style.opacity = '1', 10); };
const hideModal = (el) => {
    if (!el) return; el.style.opacity = '0';
    setTimeout(() => { el.classList.add('hidden'); el.querySelector('form')?.reset(); el.querySelector('.form-msg')?.classList.add('hidden'); }, 300);
};

const setMsg = (el, msg, isErr = true) => {
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('hidden', 'error', 'success');
    el.classList.add(isErr ? 'error' : 'success');
};

// ====================================================
// SPEECH ENGINE
// ====================================================
window.speakEnglishOutput = function(text) {
    return new Promise((resolve) => {
        if (!text || !('speechSynthesis' in window)) {
            resolve();
            return;
        }

        // cancel active speech to prevent stacking/overlaps
        window.speechSynthesis.cancel();

        // clean label e.g. HELLO (90%) -> HELLO
        const cleanText = text.replace(/\s*\(\d+%\)/g, '').trim();
        if (!cleanText) {
            resolve();
            return;
        }

        const u = new SpeechSynthesisUtterance(cleanText);
        u.rate = 1; u.pitch = 1; u.volume = 1; u.lang = 'en-US';

        const wf = $('voice-waveform');
        State.speech.speaking = true;
        if (wf) wf.classList.remove('hidden');

        let resolved = false;
        const finish = () => {
            if (resolved) return;
            resolved = true;
            State.speech.speaking = false;
            if (wf) wf.classList.add('hidden');
            resolve();
        };

        u.onend = finish;
        u.onerror = finish;

        window.speechSynthesis.speak(u);

        // Update the Voice Output box in the UI immediately
        const out = $('voice-output');
        if (out) out.textContent = cleanText;
    });
};

window.speakTranslatedOutput = function(text, language) {
    return new Promise((resolve) => {
        if (!text || !('speechSynthesis' in window)) {
            resolve();
            return;
        }

        // Map standard ISO SpeechSynthesis accents
        const langMap = {
            english: 'en-US',
            hindi: 'hi-IN',
            marathi: 'mr-IN',
            konkani: 'hi-IN',
            tamil: 'ta-IN',
            punjabi: 'pa-IN',
            gujarati: 'gu-IN',
            bhojpuri: 'hi-IN'
        };

        const locale = langMap[language?.toLowerCase()] || 'en-US';

        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1; u.pitch = 1; u.volume = 1; u.lang = locale;

        const wf = $('voice-waveform');
        State.speech.speaking = true;
        if (wf) wf.classList.remove('hidden');

        let resolved = false;
        const finish = () => {
            if (resolved) return;
            resolved = true;
            State.speech.speaking = false;
            if (wf) wf.classList.add('hidden');
            resolve();
        };

        u.onend = finish;
        u.onerror = finish;

        window.speechSynthesis.speak(u);
    });
};

const SpeechEngine = {
    COOLDOWN_MS: 1000,
    STABLE_THRESHOLD: 2,
    permissionGranted: false,

    initPermission() {
        if (this.permissionGranted || !('speechSynthesis' in window)) return;
        const u = new SpeechSynthesisUtterance('');
        u.volume = 0;
        window.speechSynthesis.speak(u);
        this.permissionGranted = true;
    },

    speak(text, lang = 'en-IN', silentUpdate = false) {
        // Fallback speak interface for backward compatibility
        window.speakEnglishOutput(text);
    },

    async tryAutoSpeak(label) {
        if (!label || label.includes('ANALYZING') || label.includes('CONFIDENCE') || label.includes('DETECTED') || label.includes('No hand')) return;
        const s = State.speech;
        // Only speak when the stable prediction actually CHANGES or cooldown is cleared
        if (s.cooldown || label === s.lastSpoken) return;
        if (State.recognition.stableCount >= this.STABLE_THRESHOLD) {
            s.cooldown = true;
            s.lastSpoken = label;
            
            if (s.cooldownTimer) clearTimeout(s.cooldownTimer);
            s.cooldownTimer = setTimeout(() => { s.cooldown = false; }, 2000); // 2s professional cooldown

            const selectedLang = $('translate-lang')?.value;
            const translateResultBox = $('translate-result');

            // 1. Cancel previous speech immediately
            window.speechSynthesis.cancel();

            // 2. Speak English output first, and wait until it completes!
            await window.speakEnglishOutput(label);

            // 3. Translation processing - only after English completes!
            if (selectedLang && selectedLang !== 'english') {
                console.log(`[Translation Flow] Translating "${label}" to "${selectedLang}"...`);
                try {
                    const res = await fetch(API.translate, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: label, language: selectedLang })
                    });
                    const d = await res.json();
                    
                    if (d.success) {
                        // Update translation box after English speech completes
                        const outputText = `${label} ↓ ${d.translated}`;
                        if (translateResultBox) {
                            translateResultBox.textContent = outputText;
                            translateResultBox.classList.add('translated-highlight'); 
                            setTimeout(() => translateResultBox.classList.remove('translated-highlight'), 1000);
                        }

                        // Professional delay (350ms) for natural AI assistant feel
                        await new Promise(r => setTimeout(r, 350));

                        // Play translated language audio
                        await window.speakTranslatedOutput(d.translated, selectedLang);
                    }
                } catch (e) {
                    console.error("Auto translation failed:", e);
                }
            }
        }
    },

    stop() {
        window.speechSynthesis?.cancel();
        State.speech.speaking = false;
        const wf = $('voice-waveform');
        if (wf) wf.classList.add('hidden');
    }
};

// ====================================================
// CAMERA CONTROLLER
// ====================================================
const Camera = {
    async start() {
        if (State.camera.stream || State.isLocked('camera')) return;
        State.lock('camera');
        State.camera.status = 'starting';
        const vid = $('video'), ov = $('video-overlay');
        try {
            State.camera.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
            if (vid) vid.srcObject = State.camera.stream;
            if (ov) ov.classList.add('hidden');
            State.camera.status = 'active';
        } catch (e) {
            console.error('Camera error:', e);
            State.camera.status = 'error';
            if (ov) { ov.classList.remove('hidden'); ov.innerHTML = '<div class="camera-placeholder"><p style="color:#ef4444;font-weight:600">Camera access denied.</p><p style="font-size:.85rem;margin-top:6px">Check browser permissions.</p></div>'; }
            const st = $('cam-status-text'); if (st) { st.textContent = 'CAM ERROR'; st.classList.remove('active'); }
            State.unlock('camera');
            throw e;
        }
        State.unlock('camera');
    },

    stop() {
        if (State.camera.stream) { State.camera.stream.getTracks().forEach(t => t.stop()); State.camera.stream = null; }
        const vid = $('video'), ov = $('video-overlay');
        if (vid) vid.srcObject = null;
        if (ov) { ov.classList.remove('hidden'); ov.innerHTML = '<div class="camera-placeholder"><div class="camera-icon-wrap"><svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 10l4.553 2.069a1 1 0 010 1.862L15 16M6.5 6.5h11a2 2 0 012 2v7a2 2 0 01-2 2h-11a2 2 0 01-2-2v-7a2 2 0 012-2z"/></svg></div><p>Click Start Recognition to begin</p></div>'; }
        State.camera.status = 'offline';
    }
};

// ====================================================
// RECOGNITION CONTROLLER
// ====================================================
const Recognition = {
    START_SVG: '<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> Start Recognition',
    STOP_SVG: '<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" stroke-width="2"/></svg> Stop Recognition',
    _isProcessing: false,  // lock to prevent overlapping API requests

    async start() {
        const r = State.recognition;
        if (r.running || State.isLocked('recognition')) return;
        if (!State.auth.user) { showToast('Please log in first.'); showModal($('login-modal')); return; }

        State.lock('recognition');
        try { await Camera.start(); } catch { State.unlock('recognition'); return; }

        r.running = true; r.stableCount = 0; r.stableLabel = ''; r.lastLabel = '';
        // Reset UI stabilization for a fresh start
        State.ui.displayedLabel = ''; State.ui.holdActive = false; State.ui.cooldownActive = false;
        if (State.ui.holdTimer) { clearTimeout(State.ui.holdTimer); State.ui.holdTimer = null; }
        if (State.ui.cooldownTimer) { clearTimeout(State.ui.cooldownTimer); State.ui.cooldownTimer = null; }
        State.speech.lastSpoken = ''; State.speech.cooldown = false;

        const btn = $('start-button');
        if (btn) btn.innerHTML = this.STOP_SVG;

        // Activate UI
        $('camera-box')?.classList.add('active');
        $('ai-scanner')?.classList.remove('hidden');
        $('ai-crosshair')?.classList.remove('hidden');
        const st = $('cam-status-text'); if (st) { st.textContent = 'READY'; st.classList.add('active'); }
        $('cam-status-dot')?.classList.add('active');

        State.unlock('recognition');
        this._isProcessing = false;
        await this._capture();
        r.intervalId = setInterval(() => this._capture(), 80);
    },

    stop() {
        const r = State.recognition;
        if (!r.running) return;
        r.running = false;
        if (r.intervalId) { clearInterval(r.intervalId); r.intervalId = null; }

        Camera.stop();
        SpeechEngine.stop();
        r.lastLabel = ''; r.stableCount = 0; r.stableLabel = '';
        window.lastRecognizedText = '';

        const btn = $('start-button');
        if (btn) btn.innerHTML = this.START_SVG;

        $('camera-box')?.classList.remove('active');
        $('ai-scanner')?.classList.add('hidden');
        $('ai-crosshair')?.classList.add('hidden');
        const st = $('cam-status-text'); if (st) { st.textContent = 'OFFLINE'; st.classList.remove('active'); }
        $('cam-status-dot')?.classList.remove('active');
        $('confidence-badge')?.classList.add('hidden');
        const pb = $('pulse-bar'); if (pb) pb.style.width = '0';
        const tt = $('translated-text'); if (tt) tt.textContent = 'Waiting for sign input...';
        const vo = $('voice-output'); if (vo) vo.textContent = 'Ready to speak when a sign is detected.';
    },

    toggle() { State.recognition.running ? this.stop() : this.start(); },

    // Backend status strings that are NOT real gesture predictions
    _STATUS_LABELS: ['ANALYZING...', 'LOW CONFIDENCE', 'NO HAND DETECTED', 'MODEL ERROR'],

    _isStatusLabel(label) {
        if (!label) return true;
        // Backend formats real labels as "HELLO (92%)" — strip confidence suffix to check
        const base = label.replace(/\s*\(\d+%\)$/g, '').trim();
        return this._STATUS_LABELS.includes(base);
    },

    _extractCleanLabel(label) {
        // "HELLO (92%)" → "HELLO"
        return label.replace(/\s*\(\d+%\)$/g, '').trim();
    },

    async _capture() {
        // Processing lock — prevent overlapping async API requests
        if (this._isProcessing) {
            console.log('[Capture] Skipped — previous request still in flight');
            return;
        }

        const vid = $('video'), r = State.recognition, ui = State.ui;
        if (!vid || vid.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !r.running) return;

        this._isProcessing = true;

        const c = document.createElement('canvas');
        // Downscale image to max 320px dimension to shrink base64 payloads by 90%+ for 10x faster transfer & inference
        const maxDim = 320;
        let w = vid.videoWidth || 320;
        let h = vid.videoHeight || 240;
        if (w > maxDim || h > maxDim) {
            if (w > h) {
                h = Math.round((h * maxDim) / w);
                w = maxDim;
            } else {
                w = Math.round((w * maxDim) / h);
                h = maxDim;
            }
        }
        c.width = w; c.height = h;
        c.getContext('2d').drawImage(vid, 0, 0, c.width, c.height);
        const data = c.toDataURL('image/jpeg', 0.8);

        // ── Only show SCANNING if nothing is currently held on screen ──
        const st = $('cam-status-text');
        if (!ui.holdActive && !ui.displayedLabel) {
            if (st) st.textContent = 'SCANNING';
        }

        try {
            const res = await fetch(API.predict, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: data }) });
            const result = await res.json();
            if (!res.ok) throw new Error(result?.error || 'fail');

            const rawLabel = result.label || null;
            const confidence = result.confidence || 0;
            const isStatus = this._isStatusLabel(rawLabel);
            const cleanLabel = rawLabel && !isStatus ? this._extractCleanLabel(rawLabel) : null;
            const isNoHand = rawLabel && rawLabel.includes('NO HAND DETECTED');

            // ── Comprehensive debug log ──
            console.log(
                `[Predict] hand=${!isNoHand && rawLabel !== null} | raw="${rawLabel}" | clean="${cleanLabel}" | conf=${confidence?.toFixed?.(1) ?? confidence} | ` +
                `isStatus=${isStatus} | stableCount=${r.stableCount} | displayed="${ui.displayedLabel}" | hold=${ui.holdActive} | cooldown=${ui.cooldownActive}`
            );

            const tt = $('translated-text'), cb = $('confidence-badge'), cv = $('confidence-val'), pb = $('pulse-bar'), vo = $('voice-output');

            // ═══════════════════════════════════════════════
            // CASE 1: Backend returned a REAL gesture label
            // ═══════════════════════════════════════════════
            if (cleanLabel) {
                // ── Stability tracking (only track real gesture labels) ──
                if (cleanLabel === r.stableLabel) {
                    r.stableCount++;
                } else {
                    r.stableCount = 1;
                    r.stableLabel = cleanLabel;
                }
                r.lastLabel = cleanLabel;

                const isHighlyConfident = confidence > 85;

                // ── Debounce: require N consecutive identical frames (bypassed if highly confident) ──
                if (!isHighlyConfident && r.stableCount < ui.debounceThreshold) {
                    console.log(`[Predict] REJECTED — debounce: need ${ui.debounceThreshold} frames, have ${r.stableCount}`);
                    // During debounce, if we already have something displayed, keep it
                    // If nothing displayed yet, show the incoming label immediately (first detection)
                    if (!ui.displayedLabel) {
                        // First-time: show it even before debounce so user gets instant feedback
                        typeText(tt, cleanLabel);
                        if (st) { st.textContent = 'DETECTED'; st.classList.add('active'); }
                        if (pb) pb.style.width = '100%';
                    }
                    return;
                }

                // ── Cooldown: prevent switching gestures faster than cooldownMs (bypassed if highly confident) ──
                if (!isHighlyConfident && ui.cooldownActive && cleanLabel !== ui.displayedLabel) {
                    console.log(`[Predict] REJECTED — cooldown active, cannot switch from "${ui.displayedLabel}" to "${cleanLabel}"`);
                    return;
                }

                // ── State lock: same label already displayed → skip DOM rewrite ──
                if (cleanLabel === ui.displayedLabel && ui.holdActive) {
                    // Silently update confidence only
                    if (cb && cv && confidence > 0) {
                        cv.textContent = Math.round(confidence);
                    }
                    SpeechEngine.tryAutoSpeak(cleanLabel);
                    return;
                }

                // ═══ NEW STABLE PREDICTION — update the display ═══
                const previousLabel = ui.displayedLabel;
                ui.displayedLabel = cleanLabel;
                window.lastRecognizedText = cleanLabel;

                // Update DOM once
                typeText(tt, cleanLabel);
                if (pb) pb.style.width = '100%';
                if (cb && cv && confidence > 0) {
                    cb.classList.remove('hidden', 'high', 'medium');
                    cv.textContent = Math.round(confidence);
                    cb.classList.add(confidence > 80 ? 'high' : confidence > 50 ? 'medium' : '');
                }
                if (st) { st.textContent = 'DETECTED'; st.classList.add('active'); }
                if (vo) vo.textContent = cleanLabel;



                // ── Hold: keep this prediction displayed for holdDurationMs ──
                if (ui.holdTimer) clearTimeout(ui.holdTimer);
                ui.holdActive = true;
                ui.holdTimer = setTimeout(() => {
                    ui.holdActive = false;
                    console.log(`[Hold] Released hold on "${ui.displayedLabel}"`);
                }, ui.holdDurationMs);

                // ── Cooldown: prevent rapid gesture switching ──
                if (previousLabel && previousLabel !== cleanLabel) {
                    if (ui.cooldownTimer) clearTimeout(ui.cooldownTimer);
                    ui.cooldownActive = true;
                    ui.cooldownTimer = setTimeout(() => {
                        ui.cooldownActive = false;
                        console.log('[Cooldown] Gesture switch cooldown ended');
                    }, ui.cooldownMs);
                }

                // Voice — only when stable prediction truly changes
                SpeechEngine.tryAutoSpeak(cleanLabel);

                console.log(`[Predict] ✓ ACCEPTED "${cleanLabel}" (was "${previousLabel || 'none'}")`);

            // ═══════════════════════════════════════════════
            // CASE 2: Backend explicitly says NO HAND DETECTED
            // ═══════════════════════════════════════════════
            } else if (isNoHand) {
                // Only clear the display if nothing is being held
                if (!ui.holdActive) {
                    if (tt) tt.textContent = 'No hand detected. Adjust your hand.';
                    if (cb) cb.classList.add('hidden');
                    if (pb) pb.style.width = '0';
                    if (st) { st.textContent = 'SCANNING'; st.classList.add('active'); }
                    ui.displayedLabel = '';
                    // Do NOT clear translate-result here — let manual translations persist
                }
                r.stableCount = 0; r.stableLabel = '';
                console.log('[Predict] No hand — backend confirmed zero hands');

            // ═══════════════════════════════════════════════
            // CASE 3: Backend status (ANALYZING, LOW CONFIDENCE, etc.)
            //   → Do NOT clear the display, do NOT reset stability
            // ═══════════════════════════════════════════════
            } else {
                // Backend is still building sequence / below confidence
                // Keep whatever is currently displayed — do nothing to DOM
                if (!ui.holdActive && !ui.displayedLabel) {
                    // Nothing displayed yet — show a non-disruptive status
                    if (st) { st.textContent = 'SCANNING'; st.classList.add('active'); }
                }
                console.log(`[Predict] Status: "${rawLabel}" — keeping current display`);
            }
        } catch (e) {
            console.error('Prediction error:', e);
            if (!ui.holdActive) {
                const tt = $('translated-text'); if (tt) tt.textContent = 'Error contacting server.';
            }
        } finally {
            this._isProcessing = false;
        }
    }
};

// ====================================================
// AUTH CONTROLLER
// ====================================================
const Auth = {
    async signup(name, email, password) {
        if (State.auth.busy) return;
        State.auth.busy = true;
        try {
            const res = await fetch(API.signup, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, email, password }) });
            const r = await res.json();
            State.auth.busy = false;
            return r;
        } catch { State.auth.busy = false; return { success: false, message: 'Network error.' }; }
    },

    async login(email, password) {
        if (State.auth.busy) return;
        State.auth.busy = true;
        try {
            const res = await fetch(API.login, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) });
            const r = await res.json();
            if (r.success) State.auth.user = r.user || {};
            State.auth.busy = false;
            return r;
        } catch { State.auth.busy = false; return { success: false, message: 'Network error.' }; }
    },

    logout() {
        Recognition.stop();
        State.reset();
        localStorage.removeItem('signova_user');
        sessionStorage.removeItem('signova_user');
        fetch('/api/logout', { method: 'POST' }).catch(() => {});
        this._updateNavbar(false);
        showToast('Logged out successfully.');
    },

    _updateNavbar(loggedIn) {
        const p = $('profile-btn'), lo = $('logout-btn'), li = $('login-btn'), su = $('signup-btn');
        if (loggedIn && State.auth.user?.name) {
            if (p) p.textContent = State.auth.user.name;
            lo?.classList.remove('hidden');
            li?.classList.add('hidden');
            su?.classList.add('hidden');
        } else {
            if (p) p.innerHTML = '<svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg> Profile';
            lo?.classList.add('hidden');
            li?.classList.remove('hidden');
            su?.classList.remove('hidden');
        }
    }
};

// ====================================================
// DOM READY — Wire events
// ====================================================
document.addEventListener('DOMContentLoaded', () => {
    // --- Track Practice Streak Activity ---
    const todayStr = new Date().toLocaleDateString('en-CA');
    let practiceDays = [];
    try {
        const stored = localStorage.getItem("signova_practice_days");
        practiceDays = stored ? JSON.parse(stored) : [];
    } catch (e) {
        practiceDays = [];
    }
    if (!Array.isArray(practiceDays)) practiceDays = [];

    if (!practiceDays.includes(todayStr)) {
        practiceDays.push(todayStr);
        localStorage.setItem("signova_practice_days", JSON.stringify(practiceDays));
        localStorage.setItem("signova_streak", practiceDays.length);
    }

    // Auto-login persistence check
    const savedUser = localStorage.getItem('signova_user') || sessionStorage.getItem('signova_user');
    if (savedUser) {
        try {
            State.auth.user = JSON.parse(savedUser);
            Auth._updateNavbar(true);
        } catch (e) {
            localStorage.removeItem('signova_user');
            sessionStorage.removeItem('signova_user');
        }
    }

    // --- Modals ---
    $('login-btn')?.addEventListener('click', () => showModal($('login-modal')));
    $('profile-btn')?.addEventListener('click', () => showModal($('login-modal')));
    $('signup-btn')?.addEventListener('click', () => showModal($('signup-modal')));
    $('close-login-btn')?.addEventListener('click', () => hideModal($('login-modal')));
    $('close-signup-btn')?.addEventListener('click', () => hideModal($('signup-modal')));
    $('login-modal')?.addEventListener('click', e => { if (e.target === $('login-modal')) hideModal($('login-modal')); });
    $('signup-modal')?.addEventListener('click', e => { if (e.target === $('signup-modal')) hideModal($('signup-modal')); });
    $('switch-to-signup')?.addEventListener('click', e => { e.preventDefault(); hideModal($('login-modal')); setTimeout(() => showModal($('signup-modal')), 350); });
    $('switch-to-login')?.addEventListener('click', e => { e.preventDefault(); hideModal($('signup-modal')); setTimeout(() => showModal($('login-modal')), 350); });

    // --- Signup ---
    $('signup-form')?.addEventListener('submit', async e => {
        e.preventDefault();
        const msg = $('signup-message');
        const n = $('signup-name')?.value.trim(), em = $('signup-email')?.value.trim(), pw = $('signup-password')?.value;
        if (!n || !em || !pw) { setMsg(msg, 'All fields required.'); return; }
        if (pw.length < 6) { setMsg(msg, 'Password must be 6+ characters.'); return; }
        const r = await Auth.signup(n, em, pw);
        if (!r?.success) { setMsg(msg, r?.message || 'Signup failed.'); return; }
        setMsg(msg, 'Account created! You can now log in.', false);
        setTimeout(() => { hideModal($('signup-modal')); setTimeout(() => showModal($('login-modal')), 350); }, 1200);
    });

    // --- Login ---
    $('login-form')?.addEventListener('submit', async e => {
        e.preventDefault();
        const msg = $('login-message');
        const em = $('login-email')?.value.trim(), pw = $('login-password')?.value;
        const remember = $('login-remember')?.checked;
        if (!em || !pw) { setMsg(msg, 'Email and password required.'); return; }
        const r = await Auth.login(em, pw);
        if (!r?.success) { setMsg(msg, r?.message || 'Login failed.'); return; }
        setMsg(msg, 'Login successful!', false);
        
        if (remember) {
            localStorage.setItem('signova_user', JSON.stringify(r.user));
            sessionStorage.removeItem('signova_user');
        } else {
            sessionStorage.setItem('signova_user', JSON.stringify(r.user));
            localStorage.removeItem('signova_user');
        }
        
        Auth._updateNavbar(true);
        setTimeout(() => hideModal($('login-modal')), 1000);
    });

    // --- Password toggles ---
    const setupPwToggle = (inputId, btnId) => {
        const inp = $(inputId), btn = $(btnId);
        if (!inp || !btn) return;
        btn.addEventListener('click', () => {
            const show = inp.type === 'password';
            inp.type = show ? 'text' : 'password';
            inp.focus({ preventScroll: true });
            btn.querySelector('.eye-open')?.classList.toggle('hidden', !show);
            btn.querySelector('.eye-closed')?.classList.toggle('hidden', show);
            btn.setAttribute('aria-pressed', String(show));
        });
    };
    setupPwToggle('login-password', 'toggle-login-password');
    setupPwToggle('signup-password', 'toggle-signup-password');

    // --- Logout ---
    $('logout-btn')?.addEventListener('click', () => Auth.logout());

    // --- Recognition toggle ---
    $('start-button')?.addEventListener('click', () => {
        SpeechEngine.initPermission();
        Recognition.toggle();
    });

    // --- Play audio ---
    $('play-audio-button')?.addEventListener('click', () => {
        const t = window.lastRecognizedText;
        if (!t || t === 'Waiting for sign input...') { 
            const v = $('voice-output'); 
            if (v) v.textContent = 'Voice output check: Audio system is active.'; 
            SpeechEngine.speak('Voice output check: Audio system is active.', 'en-IN', true);
            return; 
        }
        SpeechEngine.speak(t);
    });



    // --- Text to Sign ---
    $('text-to-sign-btn')?.addEventListener('click', async () => {
        if (State.isLocked('textToSign')) return;
        const btn = $('text-to-sign-btn'), word = $('text-input')?.value.trim();
        if (!word) { showToast('Please enter a word.'); return; }
        State.lock('textToSign');
        if (btn) { btn.disabled = true; btn.textContent = 'Searching...'; }
        
        const container = $('text-sign-container');
        const img = $('text-sign-image');
        const vid = $('text-sign-video');
        const fb = $('text-sign-fallback');
        
        if (container) container.classList.remove('hidden');

        try {
            const res = await fetch(API.textToSign, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: word }) });
            const d = await res.json();
            
            if (!res.ok || !d.success) {
                if (img) img.classList.add('hidden');
                if (vid) vid.classList.add('hidden');
                if (fb) {
                    fb.textContent = 'Animated sign not available';
                    fb.classList.remove('hidden');
                }
                const w = $('text-sign-word'); if (w) w.textContent = word;
                $('text-sign-status')?.classList.remove('hidden');
            } else {
                if (fb) fb.classList.add('hidden');
                if (d.type === 'video') {
                    if (img) img.classList.add('hidden');
                    if (vid) {
                        vid.src = d.url;
                        vid.load();
                        vid.classList.remove('hidden');
                        vid.play().catch(e => console.log('Video autoplay blocked:', e));
                    }
                } else {
                    if (vid) vid.classList.add('hidden');
                    if (img) {
                        img.src = d.url;
                        img.classList.remove('hidden');
                    }
                }
                const w = $('text-sign-word'); if (w) w.textContent = word;
                $('text-sign-status')?.classList.remove('hidden');
                SpeechEngine.speak(word, 'en-IN', true);
            }
        } catch (err) {
            console.error(err);
            showToast('Error connecting to backend.');
        }
        if (btn) { btn.disabled = false; btn.textContent = 'Show Sign'; }
        State.unlock('textToSign');
    });

    // --- Voice to Sign ---
    $('voice-to-sign-button')?.addEventListener('click', () => {
        if (State.playback.listening || State.isLocked('voiceToSign')) return;
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) { showToast('Speech recognition not supported.'); return; }

        State.lock('voiceToSign');
        State.playback.listening = true;
        const btn = $('voice-to-sign-button');
        if (btn) { btn.textContent = '🎙️ Listening...'; btn.disabled = true; btn.classList.add('listening'); }

        const rec = new SR();
        rec.lang = 'en-IN'; rec.interimResults = false; rec.maxAlternatives = 1;
        rec.start();

        const resetBtn = () => {
            State.playback.listening = false;
            State.unlock('voiceToSign');
            if (btn) { btn.textContent = 'Speak & Show Sign'; btn.disabled = false; btn.classList.remove('listening'); }
        };

        rec.onresult = async (ev) => {
            let txt = ev.results[0][0].transcript.trim().toLowerCase();
            txt = txt.replace(/[.,!?]+$/, ''); // normalize punctuation
            if (btn) btn.textContent = `Heard: "${txt}"`;
            SpeechEngine.speak(txt, 'en-IN', true);
            
            const container = $('voice-sign-container');
            const img = $('voice-sign-image');
            const vid = $('voice-sign-video');
            const fb = $('voice-sign-fallback');
            
            if (container) container.classList.remove('hidden');
            
            try {
                const res = await fetch(API.textToSign, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: txt }) });
                const d = await res.json();
                
                if (!res.ok || !d.success) {
                    if (img) img.classList.add('hidden');
                    if (vid) vid.classList.add('hidden');
                    if (fb) {
                        fb.textContent = 'Animated sign not available';
                        fb.classList.remove('hidden');
                    }
                } else {
                    if (fb) fb.classList.add('hidden');
                    if (d.type === 'video') {
                        if (img) img.classList.add('hidden');
                        if (vid) {
                            vid.src = d.url;
                            vid.load();
                            vid.classList.remove('hidden');
                            vid.play().catch(e => console.log('Video autoplay blocked:', e));
                        }
                    } else {
                        if (vid) vid.classList.add('hidden');
                        if (img) {
                            img.src = d.url;
                            img.classList.remove('hidden');
                        }
                    }
                }
            } catch (err) {
                console.error(err);
                showToast('Backend error.');
            }
            resetBtn();
        };
        rec.onerror = (ev) => { showToast(`Voice error: ${ev.error}`); resetBtn(); };
        rec.onend = () => { if (State.playback.listening) resetBtn(); };
    });

    // --- Translate ---
    $('translate-btn')?.addEventListener('click', async () => {
        if (State.translation.busy) return;
        const text = window.lastRecognizedText, lang = $('translate-lang')?.value, out = $('translate-result'), btn = $('translate-btn');
        if (!text || text === 'Waiting for sign input...') { if (out) out.textContent = 'No sign to translate.'; return; }
        if (!lang) { if (out) out.textContent = 'Select a language.'; return; }

        State.translation.busy = true;
        if (btn) { btn.disabled = true; btn.textContent = 'Translating...'; }
        if (out) out.textContent = '...';
        try {
            const res = await fetch(API.translate, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, language: lang }) });
            const d = await res.json();
            if (out) { 
                out.textContent = d.success ? d.translated : (d.message || 'Failed.'); 
                if (d.success) { 
                    out.classList.add('translated-highlight'); 
                    setTimeout(() => out.classList.remove('translated-highlight'), 1000); 
                    
                    // Cancel active speech, then wait 350ms, then speak translated output sequentially
                    window.speechSynthesis.cancel();
                    await new Promise(r => setTimeout(r, 350));
                    await window.speakTranslatedOutput(d.translated, lang);
                } 
            }
        } catch { if (out) out.textContent = 'Network error.'; }
        if (btn) { btn.disabled = false; btn.textContent = 'Translate'; }
        State.translation.busy = false;
    });

    // --- Video Modal Controls ---
    const vModal = document.getElementById('video-modal');
    const vPlayer = document.getElementById('modal-video-player');
    const vClose = document.getElementById('video-modal-close');

    const closeVideoModal = () => {
        if (vModal) {
            vModal.classList.add('hidden');
            vModal.style.opacity = '0';
        }
        if (vPlayer) {
            vPlayer.pause();
            vPlayer.src = '';
        }
    };

    if (vClose) vClose.addEventListener('click', closeVideoModal);
    if (vModal) {
        vModal.addEventListener('click', (e) => {
            if (e.target === vModal) closeVideoModal();
        });
    }
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && vModal && !vModal.classList.contains('hidden')) {
            closeVideoModal();
        }
    });

    // --- Gallery Video Hover & Click Autoplay Logic ---
    document.querySelectorAll('.gallery-card').forEach(card => {
        const video = card.querySelector('.gallery-video');
        const overlaySpan = card.querySelector('.gallery-overlay span');
        if (video) {
            video.muted = true; // start muted by default
            
            card.addEventListener('mouseenter', () => {
                if (video.paused) {
                    video.muted = true;
                    video.play().catch(err => {
                        console.log("[Gallery] Muted autoplay blocked or interrupted:", err);
                    });
                }
            });

            card.addEventListener('mouseleave', () => {
                video.pause();
                video.currentTime = 0;
                video.muted = true;
                if (overlaySpan) overlaySpan.textContent = '🔎 View Fullscreen';
            });

            card.addEventListener('click', (e) => {
                e.stopPropagation();
                // Pause preview
                video.pause();
                
                // Fetch clean source without t=0.1 frame query
                const sourceElement = video.querySelector('source');
                if (sourceElement && vModal && vPlayer) {
                    const videoSrc = sourceElement.getAttribute('src').split('#')[0];
                    vPlayer.src = videoSrc;
                    vPlayer.muted = false; // Enable sound!
                    vModal.classList.remove('hidden');
                    vModal.style.opacity = '1';
                    vPlayer.load();
                    vPlayer.play().catch(err => {
                        console.log("[Modal Video] Play blocked by browser policy:", err);
                    });
                }
            });
        }
    });

    // --- Smooth scroll ---
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', function (e) { const t = this.getAttribute('href'); if (t === '#') return; e.preventDefault(); document.querySelector(t)?.scrollIntoView({ behavior: 'smooth' }); });
    });
});