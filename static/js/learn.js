/* ============================================================
   learn.js — Premium Interactive Learning Suite Engine
   Handles tabs, visual modal preview, sequential sentence playback,
   Web Audio synthesized quiz gamification, and real-time webcam practice.
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  // --- Data Dictionaries ---
  const ALPHABET_DATA = [];
  const alphabetLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const animatedAlphabets = {
    "A": { pron: "ay", inst: "Clench your hand into a standard fist, keeping your thumb upright and pressed flat against the outer side of your index finger." },
    "C": { pron: "see", inst: "Curve all four fingers and your thumb together in a smooth semi-circle, forming a visible shape of the letter 'C'." },
    "I": { pron: "eye", inst: "Clench your hand into a fist but extend only your pinky finger straight up into the air." },
    "V": { pron: "vee", inst: "Extend your index and middle fingers upward in a spread 'V' (peace) shape, keeping other fingers folded flat." }
  };

  alphabetLetters.forEach(l => {
    if (animatedAlphabets[l]) {
      ALPHABET_DATA.push({
        id: l,
        title: `Alphabet ${l}`,
        pron: animatedAlphabets[l].pron,
        inst: animatedAlphabets[l].inst,
        hasAnim: true
      });
    } else {
      // Standard static/holographic letters
      ALPHABET_DATA.push({
        id: l,
        title: `Alphabet ${l}`,
        pron: l.toLowerCase(),
        inst: `Form the manual hand shape representing the letter '${l}' by placing your fingers in alignment with the manual ISL blueprint.`,
        hasAnim: false
      });
    }
  });

  const WORDS_DATA = [
    { id: "Hello", title: "Hello", desc: "Friendly universal greeting", inst: "Bring your flat dominant hand near your forehead and wave it outward with a gentle saluting motion.", hasAnim: true },
    { id: "Yes", title: "Yes", desc: "Affirmative confirmation", inst: "Form a soft fist with your dominant hand and tilt it forward and back, mimicking a nodding head.", hasAnim: true },
    { id: "No", title: "No", desc: "Crisp negation gesture", inst: "Extend your index and middle fingers together, then snap them down firmly to touch your thumb.", hasAnim: true },
    { id: "ThankYou", title: "Thank You", desc: "Expression of deep gratitude", inst: "Touch your flat dominant hand to your mouth, then move it forward and down toward the receiver.", hasAnim: true },
    { id: "Sorry", title: "Sorry", desc: "Sincere expression of regret", inst: "Make a closed fist and rub it gently in a warm circular motion directly over your heart.", hasAnim: true },
    { id: "Help", title: "Help", desc: "Assistance or support request", inst: "Place a thumbs-up hand shape on top of your flat open hand, lifting both hands upward together.", hasAnim: true },
    { id: "ILoveYou", title: "I Love You", desc: "Universal sign of affection", inst: "Raise your thumb, index, and pinky fingers together, keeping middle and ring fingers folded flat.", hasAnim: true }
  ];

  const SENTENCES_DATA = [
    { id: "HelloHowAreYou", title: "Hello How Are You", desc: "Standard conversational inquiry", sequence: ["Hello"], inst: "Greet with an open saluting hand wave, followed by inviting palms-up gestures." },
    { id: "IAmFine", title: "I Am Fine", desc: "Polite response to greetings", sequence: ["Yes"], inst: "Nod with your hand shape, followed by placing your open palm flat on your chest." },
    { id: "ThankYou", title: "Thank You", desc: "Expression of thankfulness", sequence: ["ThankYou"], inst: "Touch flat hand to lips and sweep it forward elegantly." },
    { id: "INeedHelp", title: "I Need Help", desc: "Urgent call for assistance", sequence: ["Help"], inst: "Perform the support lifting gesture directly in front of your chest." }
  ];

  // --- State Variables ---
  let activeTab = "alphabets";
  let activeQuizQuestions = [];
  let currentQuestionIdx = 0;
  let quizScore = 0;
  let ttsSynth = window.speechSynthesis;
  
  // Webcam Practice parameters
  let webcamStream = null;
  let captureInterval = null;
  let isWebcamRunning = false;
  let selectedPracticeTarget = { type: "alphabet", id: "A" };
  let practiceCooldown = false;

  // Audio Context for UI Sound FX
  let audioCtx = null;

  // --- DOM Elements ---
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const alphabetGrid = document.getElementById("alphabet-grid-container");
  const wordsGrid = document.getElementById("words-grid-container");
  const sentencesGrid = document.getElementById("sentences-grid-container");

  // Modal elements
  const modal = document.getElementById("gesture-modal");
  const modalTitle = document.getElementById("modal-gesture-title");
  const modalSub = document.getElementById("modal-gesture-description");
  const modalGif = document.getElementById("modal-gesture-gif");
  const modalVideo = document.getElementById("modal-gesture-video");
  const modalInst = document.getElementById("modal-gesture-instructions");
  const modalClose = document.getElementById("close-modal-btn");
  const playbackContainer = document.getElementById("modal-step-playback-controls");
  const playbackPrev = document.getElementById("modal-playback-prev");
  const playbackNext = document.getElementById("modal-playback-next");
  const playbackIndicator = document.getElementById("modal-playback-indicator");

  // Quiz elements
  const startQuizBtn = document.getElementById("start-quiz-btn");
  const quizStartScreen = document.getElementById("quiz-start-screen");
  const quizQuestionScreen = document.getElementById("quiz-question-screen");
  const quizResultScreen = document.getElementById("quiz-result-screen");
  const quizQuestionText = document.getElementById("quiz-question-text");
  const quizOptionsContainer = document.getElementById("quiz-options-container");
  const quizMediaContainer = document.getElementById("quiz-media-container");
  const quizGifElement = document.getElementById("quiz-gif-element");
  const quizVideoElement = document.getElementById("quiz-video-element");
  const quizCurrentNum = document.getElementById("quiz-current-num");
  const quizLiveScore = document.getElementById("quiz-live-score");
  const quizScoreValue = document.getElementById("quiz-score-value");
  const retryQuizBtn = document.getElementById("retry-quiz-btn");
  const quizToDashboardBtn = document.getElementById("quiz-to-dashboard-btn");

  // Webcam elements
  const practiceVideo = document.getElementById("practice-video");
  const practiceCanvas = document.getElementById("practice-capture-canvas");
  const practiceCameraToggle = document.getElementById("practice-camera-toggle");
  const practiceStatusOverlay = document.getElementById("practice-status-overlay");
  const practiceTargetHud = document.getElementById("practice-target-hud");
  const practiceTargetName = document.getElementById("practice-target-name");
  const practiceSuccessOverlay = document.getElementById("practice-success-overlay");

  const sidebarAlphabets = document.getElementById("practice-alphabets-chips");
  const sidebarWords = document.getElementById("practice-words-chips");
  const sidebarSentences = document.getElementById("practice-sentences-chips");

  // --- Sound Effects Synthesizer ---
  function playSound(type) {
    try {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      if (type === "correct") {
        osc.frequency.setValueAtTime(523.25, audioCtx.currentTime); // C5
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
      } else if (type === "wrong") {
        osc.frequency.setValueAtTime(146.83, audioCtx.currentTime); // D3
        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.4);
      } else if (type === "success") {
        // Double tone beep
        osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        osc.start();
        
        setTimeout(() => {
          osc.frequency.setValueAtTime(880.00, audioCtx.currentTime); // A5
        }, 120);

        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.45);
        osc.stop(audioCtx.currentTime + 0.45);
      }
    } catch (e) {
      console.warn("Audio Context synth not supported on this browser:", e);
    }
  }

  // --- Toast HUD Utility ---
  function showToast(msg) {
    const toast = document.getElementById("global-toast");
    const toastText = document.getElementById("global-toast-text");
    if (toast && toastText) {
      toastText.textContent = msg;
      toast.classList.remove("hidden");
      setTimeout(() => toast.classList.add("hidden"), 3000);
    }
  }

  // --- Save Achievement Progress ---
  function saveToDashboard(action, score = 0) {
    // Sync with browser localStorage for the Dashboard metrics
    let streak = parseInt(localStorage.getItem("signova_streak") || "3");
    let cards = parseInt(localStorage.getItem("signova_cards") || "4");
    let accuracy = parseFloat(localStorage.getItem("signova_accuracy") || "92.5");
    let quizzes = parseInt(localStorage.getItem("signova_quizzes") || "2");

    if (action === "quiz") {
      quizzes += 1;
      localStorage.setItem("signova_quizzes", quizzes);
      // Accuracy weighting
      accuracy = parseFloat(((accuracy * 3 + score) / 4).toFixed(1));
      localStorage.setItem("signova_accuracy", accuracy);
    } else if (action === "practice") {
      cards = Math.min(11, cards + 1);
      localStorage.setItem("signova_cards", cards);
      streak = Math.min(30, streak + 1);
      localStorage.setItem("signova_streak", streak);
    }
  }

  // --- Tabs Implementation ---
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = `tab-${btn.dataset.tab}`;
      const panel = document.getElementById(targetId);
      if (panel) panel.classList.add("active");

      activeTab = btn.dataset.tab;
      
      // Stop webcam if moving away from live practice tab
      if (activeTab !== "practice" && isWebcamRunning) {
        stopWebcam();
      }
    });
  });

  // --- Render Alphabet Grid ---
  function renderAlphabetGrid() {
    alphabetGrid.innerHTML = "";
    ALPHABET_DATA.forEach(item => {
      const card = document.createElement("div");
      card.className = "card-glow";
      
      card.innerHTML = `
        <div class="letter-title">${item.id}</div>
        <div class="pronunciation">Pron: "${item.pron}"</div>
        <p class="instruction-text">${item.inst}</p>
        <div class="card-action-bar">
          ${item.hasAnim ? `
            <button class="btn-tiny orange view-anim-btn" data-id="${item.id}" data-type="alphabet">Watch</button>
            <button class="btn-tiny green practice-direct-btn" data-id="${item.id}" data-type="alphabet">Practice</button>
          ` : `
            <button class="btn-tiny" style="opacity:0.45; cursor:not-allowed;" disabled>Watch</button>
            <button class="btn-tiny green practice-direct-btn" data-id="${item.id}" data-type="alphabet">Practice</button>
          `}
        </div>
      `;
      alphabetGrid.appendChild(card);
    });
  }

  // --- Render Common Words Grid ---
  function renderWordsGrid() {
    wordsGrid.innerHTML = "";
    WORDS_DATA.forEach(item => {
      const card = document.createElement("div");
      card.className = "card-glow";
      
      card.innerHTML = `
        <div class="letter-title" style="font-size:1.6rem; margin-bottom:8px;">${item.title}</div>
        <div class="pronunciation" style="color:var(--purple); font-size:0.75rem;">${item.desc}</div>
        <p class="instruction-text">${item.inst}</p>
        <div class="card-action-bar">
          <button class="btn-tiny purple view-anim-btn" data-id="${item.id}" data-type="word">Watch</button>
          <button class="btn-tiny green practice-direct-btn" data-id="${item.id}" data-type="word">Practice</button>
        </div>
      `;
      wordsGrid.appendChild(card);
    });
  }

  // --- Render Sentences Grid ---
  function renderSentencesGrid() {
    sentencesGrid.innerHTML = "";
    SENTENCES_DATA.forEach(item => {
      const card = document.createElement("div");
      card.className = "card-glow";
      
      card.innerHTML = `
        <div class="letter-title" style="font-size:1.35rem; margin-bottom:8px;">${item.title}</div>
        <div class="pronunciation" style="color:var(--cyan); font-size:0.75rem;">Sentence Concept</div>
        <p class="instruction-text">${item.inst}</p>
        <div class="card-action-bar">
          <button class="btn-tiny cyan view-anim-btn" data-id="${item.id}" data-type="sentence">Watch Playback</button>
          <button class="btn-tiny green practice-direct-btn" data-id="${item.id}" data-type="sentence">Practice</button>
        </div>
      `;
      sentencesGrid.appendChild(card);
    });
  }

  // --- Register Visual Modal Events ---
  function registerCardButtons() {
    document.body.addEventListener("click", e => {
      const target = e.target;
      if (target.classList.contains("view-anim-btn")) {
        const id = target.dataset.id;
        const type = target.dataset.type;
        openVisualModal(type, id);
      } else if (target.classList.contains("practice-direct-btn")) {
        const id = target.dataset.id;
        const type = target.dataset.type;
        
        // Switch tab to Practice
        const practiceTabBtn = document.querySelector(".tab-btn[data-tab='practice']");
        if (practiceTabBtn) {
          practiceTabBtn.click();
          selectPracticeTarget(type, id);
          
          // If webcam isn't running, start it
          if (!isWebcamRunning) {
            practiceCameraToggle.click();
          }
        }
      }
    });
  }

  // Explicit UI to Sign Asset video map
  const GESTURE_VIDEO_MAP = {
    "Hello How Are You": "HowAreYou.mp4",
    "I Am Fine": "IAmFine.mp4",
    "Thank You": "ThankYou.mp4",
    "I Need Help": "INeedHelp.mp4",
    "A": "A.mp4",
    "C": "C.mp4",
    "Hello": "Hello.mp4",
    "Help": "Help.mp4",
    "I": "I.mp4",
    "ILoveYou": "ILoveYou.mp4",
    "No": "No.mp4",
    "Sorry": "Sorry.mp4",
    "V": "V.mp4",
    "Yes": "Yes.mp4"
  };

  function getMappedVideoFile(titleOrId) {
    if (!titleOrId) return null;
    
    // 1. Direct key match
    if (GESTURE_VIDEO_MAP[titleOrId]) {
      return GESTURE_VIDEO_MAP[titleOrId];
    }
    
    // 2. Normalize (remove spaces and special characters, keep case)
    const normalizedInput = titleOrId.replace(/[\s\-_'"?,.!]/g, "");
    
    // Match against normalized keys
    for (const [key, val] of Object.entries(GESTURE_VIDEO_MAP)) {
      const normalizedKey = key.replace(/[\s\-_'"?,.!]/g, "");
      if (normalizedKey === normalizedInput) {
        return val;
      }
    }
    
    // 3. Fallback
    return normalizedInput + ".mp4";
  }

  /**
   * Load a sign asset into the modal, auto-detecting mp4 vs image.
   * Tries .mp4 first (current asset format), falls back to .gif.
   */
  function loadModalAsset(assetId) {
    const videoFile = getMappedVideoFile(assetId);
    
    // Use absolute paths: /static/sign_assets/
    const mp4Url = videoFile ? `/static/sign_assets/${videoFile}` : `/static/sign_assets/${assetId}.mp4`;
    const gifUrl = `/static/sign_assets/${assetId}.gif`;

    // Restore original instructions text if cached
    const modalInstructions = document.getElementById("modal-gesture-instructions");
    if (modalInstructions && modalInstructions.dataset.originalText) {
      modalInstructions.textContent = modalInstructions.dataset.originalText;
    }

    // Try mp4 first by checking if video can load
    if (modalVideo) {
      modalVideo.src = mp4Url;
      modalVideo.load();
      modalVideo.classList.remove('hidden');
      modalVideo.play().catch(() => {});
    }
    if (modalGif) {
      modalGif.classList.add('hidden');
      modalGif.src = '';
    }

    // Handle video load error — fall back to gif/image or show error
    if (modalVideo) {
      modalVideo.onerror = () => {
        modalVideo.classList.add('hidden');
        modalVideo.src = '';
        if (modalGif && gifUrl) {
          modalGif.src = gifUrl;
          modalGif.classList.remove('hidden');
          modalGif.onerror = () => {
            modalGif.classList.add('hidden');
            modalGif.src = '';
            if (modalInstructions) {
              modalInstructions.textContent = "Playback asset not found.";
            }
          };
        } else {
          if (modalInstructions) {
            modalInstructions.textContent = "Playback asset not found.";
          }
        }
      };
    }
  }

  // --- Global Preview Modal Handling ---
  let activeSequence = [];
  let currentSequenceIdx = 0;

  function openVisualModal(type, id) {
    playbackContainer.classList.add("hidden");
    const modalInstructions = document.getElementById("modal-gesture-instructions");
    
    if (type === "alphabet") {
      const item = ALPHABET_DATA.find(a => a.id === id);
      modalTitle.textContent = item.title;
      modalSub.textContent = `A-Z manual alphabet guide`;
      modalInst.textContent = item.inst;
      if (modalInstructions) modalInstructions.dataset.originalText = item.inst;
      loadModalAsset(item.id);
    } else if (type === "word") {
      const item = WORDS_DATA.find(w => w.id === id);
      modalTitle.textContent = item.title;
      modalSub.textContent = `Common vocabulary animation preview`;
      modalInst.textContent = item.inst;
      if (modalInstructions) modalInstructions.dataset.originalText = item.inst;
      loadModalAsset(item.id);
    } else if (type === "sentence") {
      const item = SENTENCES_DATA.find(s => s.id === id);
      modalTitle.textContent = item.title;
      modalSub.textContent = `Visual Sentence Combination (Step-by-Step Playback)`;
      modalInst.textContent = item.inst;
      if (modalInstructions) modalInstructions.dataset.originalText = item.inst;

      // Force play exact mapped video for this sentence
      activeSequence = [item.title];
      currentSequenceIdx = 0;
      updateSentencePlayback();
      
      // Hide sequence step playback controls since it's a unified sentence video
      playbackContainer.classList.add("hidden");
    }

    modal.classList.remove("hidden");
  }

  function updateSentencePlayback() {
    const wordKey = activeSequence[currentSequenceIdx];
    loadModalAsset(wordKey);
    playbackIndicator.textContent = `Step ${currentSequenceIdx + 1} of ${activeSequence.length}: "${wordKey}"`;
  }

  playbackPrev.addEventListener("click", () => {
    if (currentSequenceIdx > 0) {
      currentSequenceIdx--;
      updateSentencePlayback();
    }
  });

  playbackNext.addEventListener("click", () => {
    if (currentSequenceIdx < activeSequence.length - 1) {
      currentSequenceIdx++;
      updateSentencePlayback();
    }
  });

  modalClose.addEventListener("click", () => {
    modal.classList.add("hidden");
    if (modalGif) modalGif.src = ""; // reset image source
    if (modalVideo) { modalVideo.pause(); modalVideo.src = ""; modalVideo.classList.add('hidden'); } // reset video source
  });

  // --- Quiz Mode Implementation ---
  startQuizBtn.addEventListener("click", () => {
    quizStartScreen.classList.add("hidden");
    quizQuestionScreen.classList.remove("hidden");
    currentQuestionIdx = 0;
    quizScore = 0;
    fetchQuizQuestions();
  });

  async function fetchQuizQuestions() {
    try {
      quizQuestionText.textContent = "Loading next challenge...";
      quizOptionsContainer.innerHTML = "";
      quizMediaContainer.classList.add("hidden");

      const response = await fetch("/api/quiz-questions");
      const data = await response.json();
      
      if (data.success && data.questions) {
        activeQuizQuestions = data.questions;
        renderQuizQuestion();
      } else {
        showToast("Error connecting to Quiz server. Retrying...");
      }
    } catch (e) {
      showToast("Network exception loading quiz questions.");
    }
  }

  function renderQuizQuestion() {
    const q = activeQuizQuestions[currentQuestionIdx];
    quizCurrentNum.textContent = currentQuestionIdx + 1;
    quizLiveScore.textContent = quizScore;

    quizQuestionText.textContent = q.question;
    quizOptionsContainer.innerHTML = "";

    // Show media preview if available (e.g. video/GIF matching question)
    if (q.asset_url) {
      quizMediaContainer.classList.remove("hidden");
      const isVideo = q.asset_url.toLowerCase().endsWith(".mp4");
      
      if (isVideo) {
        if (quizGifElement) quizGifElement.classList.add("hidden");
        if (quizVideoElement) {
          quizVideoElement.src = q.asset_url;
          quizVideoElement.classList.remove("hidden");
          quizVideoElement.load();
          quizVideoElement.play().catch(e => console.log("Quiz video autoplay blocked:", e));
        }
      } else {
        if (quizVideoElement) {
          quizVideoElement.classList.add("hidden");
          quizVideoElement.src = "";
        }
        if (quizGifElement) {
          quizGifElement.src = q.asset_url;
          quizGifElement.classList.remove("hidden");
        }
      }
    } else {
      quizMediaContainer.classList.add("hidden");
      if (quizGifElement) quizGifElement.src = "";
      if (quizVideoElement) {
        quizVideoElement.src = "";
        quizVideoElement.classList.add("hidden");
      }
    }

    // Render option buttons
    q.options.forEach((opt, idx) => {
      const btn = document.createElement("button");
      btn.className = "quiz-opt-btn";
      btn.innerHTML = `<span class="opt-label">${String.fromCharCode(65 + idx)}.</span> ${opt}`;
      
      btn.addEventListener("click", () => handleQuizAnswer(idx, btn));
      quizOptionsContainer.appendChild(btn);
    });
  }

  function handleQuizAnswer(selectedIndex, selectedBtn) {
    // Disable all option buttons
    const allButtons = quizOptionsContainer.querySelectorAll(".quiz-opt-btn");
    allButtons.forEach(btn => btn.disabled = true);

    const q = activeQuizQuestions[currentQuestionIdx];
    const isCorrect = (selectedIndex === q.correct_index);

    if (isCorrect) {
      selectedBtn.classList.add("correct");
      quizScore += 20;
      playSound("correct");
    } else {
      selectedBtn.classList.add("wrong");
      allButtons[q.correct_index].classList.add("correct");
      playSound("wrong");
    }

    setTimeout(() => {
      if (currentQuestionIdx < 4) {
        currentQuestionIdx++;
        renderQuizQuestion();
      } else {
        showQuizResults();
      }
    }, 1800);
  }

  function showQuizResults() {
    quizQuestionScreen.classList.add("hidden");
    quizResultScreen.classList.remove("hidden");

    quizScoreValue.textContent = `${quizScore}/100`;
    saveToDashboard("quiz", quizScore);

    if (quizScore >= 80) {
      quizResultScreen.querySelector(".quiz-trophy").textContent = "🏆";
      document.getElementById("quiz-feedback-text").textContent = "Outstanding work! You've achieved highly on Indian Sign Language visual signs!";
    } else if (quizScore >= 50) {
      quizResultScreen.querySelector(".quiz-trophy").textContent = "⭐";
      document.getElementById("quiz-feedback-text").textContent = "Great job! A bit more visual review will make you absolute master.";
    } else {
      quizResultScreen.querySelector(".quiz-trophy").textContent = "📚";
      document.getElementById("quiz-feedback-text").textContent = "Practice makes perfect. Click below to review letters and common words again.";
    }
  }

  retryQuizBtn.addEventListener("click", () => {
    quizResultScreen.classList.add("hidden");
    quizQuestionScreen.classList.remove("hidden");
    currentQuestionIdx = 0;
    quizScore = 0;
    fetchQuizQuestions();
  });

  quizToDashboardBtn.addEventListener("click", () => {
    window.location.href = "/dashboard";
  });

  // --- Webcam Live Practice Implementation ---
  function selectPracticeTarget(type, id) {
    selectedPracticeTarget = { type, id };
    
    // Highlight sidebar chip
    const allChips = document.querySelectorAll(".chip-btn");
    allChips.forEach(chip => {
      if (chip.dataset.id === id && chip.dataset.type === type) {
        chip.classList.add("selected");
      } else {
        chip.classList.remove("selected");
      }
    });

    // Update HUD
    const displayLabel = WORDS_DATA.find(w => w.id === id)?.title || 
                         SENTENCES_DATA.find(s => s.id === id)?.title || 
                         `Letter ${id}`;
    practiceTargetName.textContent = displayLabel;
    practiceTargetHud.classList.remove("hidden");
  }

  function renderSidebarSelectorChips() {
    sidebarAlphabets.innerHTML = "";
    sidebarWords.innerHTML = "";
    sidebarSentences.innerHTML = "";

    // Letters
    LETTERS = ["A", "C", "I", "V"];
    LETTERS.forEach(l => {
      const chip = document.createElement("button");
      chip.className = "chip-btn";
      chip.textContent = l;
      chip.dataset.id = l;
      chip.dataset.type = "alphabet";
      
      chip.addEventListener("click", () => selectPracticeTarget("alphabet", l));
      sidebarAlphabets.appendChild(chip);
    });

    // Common Words
    WORDS_DATA.forEach(w => {
      const chip = document.createElement("button");
      chip.className = "chip-btn";
      chip.textContent = w.title;
      chip.dataset.id = w.id;
      chip.dataset.type = "word";
      
      chip.addEventListener("click", () => selectPracticeTarget("word", w.id));
      sidebarWords.appendChild(chip);
    });

    // Sentences
    SENTENCES_DATA.forEach(s => {
      const chip = document.createElement("button");
      chip.className = "chip-btn";
      chip.textContent = s.title;
      chip.dataset.id = s.id;
      chip.dataset.type = "sentence";
      
      chip.addEventListener("click", () => selectPracticeTarget("sentence", s.id));
      sidebarSentences.appendChild(chip);
    });

    // Default select letter A
    selectPracticeTarget("alphabet", "A");
  }

  practiceCameraToggle.addEventListener("click", () => {
    if (isWebcamRunning) {
      stopWebcam();
    } else {
      startWebcam();
    }
  });

  async function startWebcam() {
    try {
      practiceStatusOverlay.querySelector("p").textContent = "Activating camera streams...";
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      practiceVideo.srcObject = webcamStream;
      practiceVideo.play();
      
      isWebcamRunning = true;
      practiceCameraToggle.textContent = "Disable Camera Feed";
      practiceCameraToggle.classList.add("danger");
      practiceStatusOverlay.classList.add("hidden");

      // Start capture polling
      startCapturePolling();
    } catch (e) {
      showToast("Unable to access user camera device.");
      practiceStatusOverlay.querySelector("p").textContent = "Camera access denied. Enable permissions in your browser to practice.";
    }
  }

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach(t => t.stop());
    }
    practiceVideo.srcObject = null;
    isWebcamRunning = false;
    practiceCameraToggle.textContent = "Activate Camera Feed";
    practiceCameraToggle.classList.remove("danger");
    practiceStatusOverlay.classList.remove("hidden");
    practiceTargetHud.classList.add("hidden");
    
    stopCapturePolling();
  }

  function startCapturePolling() {
    stopCapturePolling();
    captureInterval = setInterval(pollFrameToBackend, 100);
  }

  function stopCapturePolling() {
    if (captureInterval) {
      clearInterval(captureInterval);
      captureInterval = null;
    }
  }

  async function pollFrameToBackend() {
    if (practiceCooldown || !isWebcamRunning) return;

    const ctx = practiceCanvas.getContext("2d");
    practiceCanvas.width = 320;
    practiceCanvas.height = 240;
    
    // Draw mirrored video
    ctx.translate(320, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(practiceVideo, 0, 0, 320, 240);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    const base64Img = practiceCanvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    
    // Determine route endpoint
    const isSentenceMode = (selectedPracticeTarget.type === "sentence");
    const url = isSentenceMode ? "/api/predict-sentence" : "/api/predict";

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Img })
      });

      const data = await response.json();
      
      // Check for matching target sign
      if (data) {
        let detected = "";
        
        if (isSentenceMode) {
          detected = (data.sentence || "").replace(/\s+/g, "").toLowerCase();
        } else {
          detected = (data.label || "").toLowerCase();
        }

        const target = selectedPracticeTarget.id.toLowerCase();
        
        // Clean matching comparison
        if (detected && (detected === target || detected.includes(target))) {
          triggerPracticeSuccess();
        }
      }
    } catch (e) {
      // Slient errors to prevent cluttering the practice loop
    }
  }

  function triggerPracticeSuccess() {
    practiceCooldown = true;
    playSound("success");
    practiceSuccessOverlay.classList.remove("hidden");
    
    saveToDashboard("practice");
    showToast(`Achievement unlocked! Mastered gesture '${selectedPracticeTarget.id}'!`);

    // Let the green overlay flash beautifully, then clear
    setTimeout(() => {
      practiceSuccessOverlay.classList.add("hidden");
      practiceCooldown = false;
    }, 2800);
  }

  // --- Initializers ---
  renderAlphabetGrid();
  renderWordsGrid();
  renderSentencesGrid();
  renderSidebarSelectorChips();
  registerCardButtons();
});
