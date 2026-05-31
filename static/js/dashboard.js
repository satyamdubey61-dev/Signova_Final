/* ============================================================
   dashboard.js — Student Progress Dashboard Logic
   Parses synced localStorage learning metrics, unlocks accessibility badges,
   and visualizes accuracy trendlines using customized Chart.js graphs.
   ============================================================ */

// --- Practice Streak Reset Handler ---
window.resetPracticeStreak = function() {
  localStorage.setItem("signova_practice_days", JSON.stringify([]));
  localStorage.setItem("signova_streak", "0");
  const el = document.getElementById("db-streak-val");
  if (el) {
    el.textContent = "0 Days";
  }
};

document.addEventListener("DOMContentLoaded", async () => {
  // --- Real Usage Practice Streak Tracking ---
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

  let streak = practiceDays.length;
  let cards = parseInt(localStorage.getItem("signova_cards") || "4");
  let accuracy = parseFloat(localStorage.getItem("signova_accuracy") || "92.5");
  let quizzes = parseInt(localStorage.getItem("signova_quizzes") || "2");

  // Load from API as default/fallback sync
  try {
    const res = await fetch("/api/progress-summary");
    const data = await res.json();
    if (data.success && data.default_stats) {
      const defaults = data.default_stats;
      // If user hasn't set anything locally, sync with backend (except streak, which is real usage)
      if (!localStorage.getItem("signova_cards")) {
        cards = defaults.cards_learned;
        accuracy = defaults.accuracy_avg;
        quizzes = defaults.quizzes_taken;
        
        localStorage.setItem("signova_cards", cards);
        localStorage.setItem("signova_accuracy", accuracy);
        localStorage.setItem("signova_quizzes", quizzes);
      }
    }
  } catch (e) {
    console.warn("Unable to fetch sync summary stats:", e);
  }

  // --- Populate Stats in DOM ---
  document.getElementById("db-streak-val").textContent = `${streak} ${streak === 1 ? 'Day' : 'Days'}`;
  document.getElementById("db-accuracy-val").textContent = `${accuracy}%`;
  document.getElementById("db-cards-val").textContent = `${cards} / 11`;
  document.getElementById("db-quizzes-val").textContent = quizzes;

  // --- Dynamic Badge Unlocking Engine ---
  const BADGES = [
    {
      id: "first_step",
      name: "First Steps",
      desc: "Practice your first gesture or take your first quiz evaluation.",
      icon: "🌱",
      condition: () => (quizzes > 0 || cards > 4)
    },
    {
      id: "quiz_champ",
      name: "Quiz Scholar",
      desc: "Complete a gamified quiz session with an accuracy average of 80% or higher.",
      icon: "🎓",
      condition: () => (quizzes > 0 && accuracy >= 80.0)
    },
    {
      id: "gesture_novice",
      name: "Gesture Novice",
      desc: "Successfully master 5 or more distinct manual ISL gestures.",
      icon: "🤟",
      condition: () => (cards >= 5)
    },
    {
      id: "velocity_master",
      name: "Accuracy Enthusiast",
      desc: "Maintain an overall average accuracy metric above 92%.",
      icon: "⚡",
      condition: () => (accuracy >= 92.0)
    },
    {
      id: "absolute_maestro",
      name: "Absolute Maestro",
      desc: "Master 8+ gestures, complete 3+ quizzes, and maintain 94% accuracy.",
      icon: "🏆",
      condition: () => (cards >= 8 && quizzes >= 3 && accuracy >= 94.0)
    }
  ];

  const badgesContainer = document.getElementById("badges-list-container");
  badgesContainer.innerHTML = "";

  BADGES.forEach(badge => {
    const isUnlocked = badge.condition();
    const item = document.createElement("div");
    item.className = `badge-item ${isUnlocked ? 'unlocked' : ''}`;
    
    item.innerHTML = `
      <div class="badge-icon">${badge.icon}</div>
      <div class="badge-info">
        <h4>${badge.name}</h4>
        <p>${badge.desc}</p>
      </div>
      <div class="badge-status-marker" style="font-size:0.75rem; font-weight:700; color:${isUnlocked ? 'var(--green)' : 'var(--text-muted)'};">
        ${isUnlocked ? 'UNLOCKED' : 'LOCKED'}
      </div>
    `;
    badgesContainer.appendChild(item);
  });

  // --- Render Recent Activity Timeline ---
  const timelineContainer = document.getElementById("timeline-list");
  timelineContainer.innerHTML = "";

  const TIMELINE_DATA = [
    { title: "Gamified Quiz Evaluation", date: "Today", status: quizzes > 0 ? "Completed" : "Planned" },
    { title: "Greetings & Common Phrases", date: "Yesterday", status: cards >= 4 ? "Completed" : "In Progress" },
    { title: "Manual Alphabets practice", date: "3 Days ago", status: cards >= 2 ? "Completed" : "In Progress" },
    { title: "Live Conversation Dialogs", date: "Planned", status: "In Progress" }
  ];

  TIMELINE_DATA.forEach(t => {
    const item = document.createElement("div");
    item.className = `timeline-item ${t.status === 'Completed' ? 'completed' : 'in-progress'}`;
    
    item.innerHTML = `
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-info">
          <h4>${t.title}</h4>
          <span class="date">${t.date}</span>
        </div>
        <span class="timeline-status">${t.status}</span>
      </div>
    `;
    timelineContainer.appendChild(item);
  });

  // --- Chart.js Trendline Graph Renderer ---
  const ctx = document.getElementById("accuracyChart").getContext("2d");
  
  // Custom neon gradient setup
  const glowGradient = ctx.createLinearGradient(0, 0, 0, 300);
  glowGradient.addColorStop(0, "rgba(34, 211, 238, 0.2)");
  glowGradient.addColorStop(1, "rgba(168, 85, 247, 0.0)");

  // Seed trendline values using current accuracy
  const chartData = [85, 88, 90, 89, 93, 91, accuracy];

  new Chart(ctx, {
    type: "line",
    data: {
      labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
      datasets: [{
        label: "Posture Match Accuracy %",
        data: chartData,
        borderColor: "#22d3ee",
        borderWidth: 3,
        pointBackgroundColor: "#a855f7",
        pointBorderColor: "#fff",
        pointHoverRadius: 8,
        fill: true,
        backgroundColor: glowGradient,
        tension: 0.35
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.03)" },
          ticks: { color: "#64748b", font: { family: "Inter" } }
        },
        y: {
          min: 60,
          max: 100,
          grid: { color: "rgba(255, 255, 255, 0.03)" },
          ticks: { color: "#64748b", font: { family: "Inter" } }
        }
      }
    }
  });
});
