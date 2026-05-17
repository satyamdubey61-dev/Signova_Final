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
const SpeechEngine = {
    COOLDOWN_MS: 1200,
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
        const out = $('voice-output'), wf = $('voice-waveform');
        if (!('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1; u.pitch = 1; u.volume = 1; u.lang = lang;
        State.speech.speaking = true;
        if (wf) wf.classList.remove('hidden');
        u.onend = u.onerror = () => { State.speech.speaking = false; if (wf) wf.classList.add('hidden'); };
        window.speechSynthesis.speak(u);
        if (!silentUpdate && out) out.textContent = text;
    },

    tryAutoSpeak(label) {
        const s = State.speech;
        if (s.cooldown || label === s.lastSpoken) return;
        if (State.recognition.stableCount >= this.STABLE_THRESHOLD) {
            s.cooldown = true;
            s.lastSpoken = label;
            this.speak(label);
            if (s.cooldownTimer) clearTimeout(s.cooldownTimer);
            s.cooldownTimer = setTimeout(() => { s.cooldown = false; s.lastSpoken = ''; }, this.COOLDOWN_MS);
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

    async start() {
        const r = State.recognition;
        if (r.running || State.isLocked('recognition')) return;
        if (!State.auth.user) { showToast('Please log in first.'); showModal($('login-modal')); return; }

        State.lock('recognition');
        try { await Camera.start(); } catch { State.unlock('recognition'); return; }

        r.running = true; r.stableCount = 0; r.stableLabel = ''; r.lastLabel = '';
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
        await this._capture();
        r.intervalId = setInterval(() => this._capture(), 1500);
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

    async _capture() {
        const vid = $('video'), r = State.recognition;
        if (!vid || vid.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !r.running) return;

        const c = document.createElement('canvas');
        c.width = vid.videoWidth || 300; c.height = vid.videoHeight || 300;
        c.getContext('2d').drawImage(vid, 0, 0, c.width, c.height);
        const data = c.toDataURL('image/jpeg', 0.8);

        const st = $('cam-status-text'); if (st) st.textContent = 'ANALYZING...';

        try {
            const res = await fetch(API.predict, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: data }) });
            const result = await res.json();
            if (!res.ok) throw new Error(result?.error || 'fail');

            const tt = $('translated-text'), cb = $('confidence-badge'), cv = $('confidence-val'), pb = $('pulse-bar'), vo = $('voice-output');

            if (result.label) {
                // Stability tracking
                if (result.label === r.stableLabel) { r.stableCount++; } else { r.stableCount = 1; r.stableLabel = result.label; }
                r.lastLabel = result.label;
                window.lastRecognizedText = result.label;

                typeText(tt, result.label);
                if (pb) pb.style.width = '100%';
                if (cb && cv && result.confidence !== undefined) {
                    cb.classList.remove('hidden', 'high', 'medium');
                    cv.textContent = Math.round(result.confidence);
                    cb.classList.add(result.confidence > 80 ? 'high' : result.confidence > 50 ? 'medium' : '');
                }
                if (st) { st.textContent = 'DETECTED'; st.classList.add('active'); }

                if (vo) vo.textContent = result.label;
                SpeechEngine.tryAutoSpeak(result.label);
            } else {
                if (tt) tt.textContent = 'No hand detected. Adjust your hand.';
                if (cb) cb.classList.add('hidden');
                if (pb) pb.style.width = '0';
                if (st) { st.textContent = 'SCANNING'; st.classList.add('active'); }
                r.stableCount = 0; r.stableLabel = '';
            }
        } catch (e) {
            console.error('Prediction error:', e);
            const tt = $('translated-text'); if (tt) tt.textContent = 'Error contacting server.';
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
        if (!em || !pw) { setMsg(msg, 'Email and password required.'); return; }
        const r = await Auth.login(em, pw);
        if (!r?.success) { setMsg(msg, r?.message || 'Login failed.'); return; }
        setMsg(msg, 'Login successful!', false);
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
        try {
            const res = await fetch(API.textToSign, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: word }) });
            if (!res.ok) { const e = await res.json().catch(() => ({})); showToast(e.message || `"${word}" not found.`); $('text-sign-image')?.classList.add('hidden'); $('text-sign-status')?.classList.add('hidden'); }
            else {
                const url = URL.createObjectURL(await res.blob());
                const img = $('text-sign-image'); if (img) { img.src = url; img.classList.remove('hidden'); }
                const w = $('text-sign-word'); if (w) w.textContent = word;
                $('text-sign-status')?.classList.remove('hidden');
                SpeechEngine.speak(word, 'en-IN', true);
            }
        } catch { showToast('Error connecting to backend.'); }
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
            try {
                const res = await fetch(API.textToSign, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: txt }) });
                if (!res.ok) { showToast(`"${txt}" not found.`); $('voice-sign-image')?.classList.add('hidden'); }
                else { 
                    const img = $('voice-sign-image'); 
                    if (img) { img.src = URL.createObjectURL(await res.blob()); img.classList.remove('hidden'); } 
                }
            } catch { showToast('Backend error.'); }
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
                    const langMap = { english: 'en-IN', hindi: 'hi-IN', marathi: 'mr-IN', konkani: 'hi-IN', tamil: 'ta-IN' };
                    SpeechEngine.speak(d.translated, langMap[lang] || 'en-IN', true);
                } 
            }
        } catch { if (out) out.textContent = 'Network error.'; }
        if (btn) { btn.disabled = false; btn.textContent = 'Translate'; }
        State.translation.busy = false;
    });

    // --- Smooth scroll ---
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', function (e) { const t = this.getAttribute('href'); if (t === '#') return; e.preventDefault(); document.querySelector(t)?.scrollIntoView({ behavior: 'smooth' }); });
    });
});