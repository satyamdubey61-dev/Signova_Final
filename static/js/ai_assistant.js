/* ============================================================
   ai_assistant.js — Interactive conversational assistant logic
   Implements conversational threads, typewriter animations, and
   dynamic sign GIF media embeds.
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chat-input-form");
  const userInput = document.getElementById("chat-user-input");
  const historyContainer = document.getElementById("chat-history-container");
  const quickPrompts = document.querySelectorAll(".quick-prompt-btn");

  // --- Scroll to Bottom ---
  function scrollToBottom() {
    historyContainer.scrollTo({
      top: historyContainer.scrollHeight,
      behavior: "smooth"
    });
  }

  // --- Create Typing Indicator ---
  function showTypingIndicator() {
    const indicator = document.createElement("div");
    indicator.className = "typing-indicator animate-bubble";
    indicator.id = "typing-indicator-bubble";
    
    indicator.innerHTML = `
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    `;
    historyContainer.appendChild(indicator);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const indicator = document.getElementById("typing-indicator-bubble");
    if (indicator) indicator.remove();
  }

  // --- Create Message Bubble ---
  // --- Create Embed Card ---
  function renderEmbedCard(embed) {
    if (!embed) return;
    const embedCard = document.createElement("div");
    embedCard.className = "msg-embed animate-bubble";
    
    // Premium glowing card styles
    embedCard.style.cssText = `
      border: 2px solid var(--purple, #a855f7);
      border-radius: 16px;
      overflow: hidden;
      margin-top: 10px;
      margin-bottom: 10px;
      box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
      transition: all 0.3s ease;
      background: #1e1b4b;
      max-width: 320px;
      cursor: pointer;
    `;
    embedCard.addEventListener("mouseenter", () => {
      embedCard.style.transform = "scale(1.03)";
      embedCard.style.boxShadow = "0 0 25px rgba(168, 85, 247, 0.6)";
    });
    embedCard.addEventListener("mouseleave", () => {
      embedCard.style.transform = "scale(1)";
      embedCard.style.boxShadow = "0 0 15px rgba(168, 85, 247, 0.3)";
    });

    const isVideo = embed.type === "video" || embed.url.toLowerCase().endsWith(".mp4");
    if (isVideo) {
      embedCard.innerHTML = `
        <video src="${embed.url}" autoplay muted loop playsinline style="width: 100%; height: auto; display: block; border-bottom: 1px solid rgba(168, 85, 247, 0.2);"></video>
        <div class="embed-title" style="padding: 10px 15px; font-weight: 600; color: #fff; font-family: 'Space Grotesk', sans-serif;">ISL Playback: ${embed.title}</div>
      `;
    } else {
      embedCard.innerHTML = `
        <img src="${embed.url}" alt="ISL Sign: ${embed.title}" style="width: 100%; height: auto; display: block; border-bottom: 1px solid rgba(168, 85, 247, 0.2);">
        <div class="embed-title" style="padding: 10px 15px; font-weight: 600; color: #fff; font-family: 'Space Grotesk', sans-serif;">ISL Visual: ${embed.title}</div>
      `;
    }
    
    historyContainer.appendChild(embedCard);
    scrollToBottom();
  }

  // --- Create Message Bubble ---
  function appendMessage(sender, text, embed = null) {
    const bubble = document.createElement("div");
    bubble.className = `msg-bubble ${sender} animate-bubble`;
    
    // We parse a very lightweight markdown bold format **text** to HTML <strong>text</strong>
    const formattedText = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<strong>$1</strong>");

    bubble.innerHTML = `<div class="text">${formattedText}</div>`;
    
    historyContainer.appendChild(bubble);
    
    if (embed) {
      renderEmbedCard(embed);
    } else {
      scrollToBottom();
    }
  }

  // --- Typewriter Effect for AI ---
  function appendTypewriterMessage(text, embed = null) {
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble mentor animate-bubble";
    
    const textNode = document.createElement("div");
    textNode.className = "text";
    bubble.appendChild(textNode);
    historyContainer.appendChild(bubble);
    scrollToBottom();

    // Format text
    const formattedText = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<strong>$1</strong>");

    // Clean word-by-word animation
    const words = formattedText.split(" ");
    let currentIdx = 0;

    function typeNextWord() {
      if (currentIdx < words.length) {
        textNode.innerHTML = words.slice(0, currentIdx + 1).join(" ");
        currentIdx++;
        scrollToBottom();
        setTimeout(typeNextWord, 25 + Math.random() * 15); // Natural fluid cadence
      } else {
        if (embed) {
          renderEmbedCard(embed);
        }
      }
    }

    typeNextWord();
  }

  // --- Form Submission Handler ---
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // 1. Add User bubble
    appendMessage("user", query);
    userInput.value = "";

    // 2. Show Typing dots
    showTypingIndicator();

    // Check if it's a playback request
    const queryLower = query.toLowerCase();
    const playbackKeywords = ["show me", "play", "teach", "display"];
    let isPlaybackRequest = false;
    let gestureKey = "";

    for (const kw of playbackKeywords) {
      if (queryLower.includes(kw)) {
        isPlaybackRequest = true;
        // Extract what follows the keyword
        const idx = queryLower.indexOf(kw);
        gestureKey = query.substring(idx + kw.length).trim();
        // Remove helper words like "alphabet", "sign for", "gesture for", "the sign", "the"
        gestureKey = gestureKey.replace(/^(alphabet|sign for|gesture for|the sign|the)\s+/i, "").trim();
        break;
      }
    }

    if (isPlaybackRequest && gestureKey) {
      // Map names to clean slugs
      const cleanKey = gestureKey.toLowerCase().replace(/[\s\-_'"?,.!]/g, "");
      
      const AVAILABLE_ASSETS = {
        "a": { file: "A.mp4", display: "Alphabet A" },
        "c": { file: "C.mp4", display: "Alphabet C" },
        "i": { file: "I.mp4", display: "Alphabet I" },
        "v": { file: "V.mp4", display: "Alphabet V" },
        "hello": { file: "Hello.mp4", display: "Hello" },
        "help": { file: "Help.mp4", display: "Help" },
        "howareyou": { file: "HowAreYou.mp4", display: "How Are You" },
        "iamfine": { file: "IAmFine.mp4", display: "I Am Fine" },
        "iloveyou": { file: "ILoveYou.mp4", display: "I Love You" },
        "ineedhelp": { file: "INeedHelp.mp4", display: "I Need Help" },
        "no": { file: "No.mp4", display: "No" },
        "sorry": { file: "Sorry.mp4", display: "Sorry" },
        "thankyou": { file: "ThankYou.mp4", display: "Thank You" },
        "yes": { file: "Yes.mp4", display: "Yes" }
      };

      const matched = AVAILABLE_ASSETS[cleanKey];

      setTimeout(() => {
        removeTypingIndicator();
        if (matched) {
          const replyText = `Certainly! I've loaded the official Indian Sign Language (ISL) playback reference for **'${matched.display}'** below:`;
          appendTypewriterMessage(replyText, {
            type: "video",
            url: `/static/sign_assets/${matched.file}`,
            title: matched.display
          });
        } else {
          appendMessage("mentor", "Gesture asset not available yet.");
        }
      }, 800); // realistic typing latency
      return;
    }

    try {
      // 3. Request AI reply from backend
      const response = await fetch("/api/ai-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query })
      });
      const data = await response.json();

      // 4. Remove Typing dots
      removeTypingIndicator();

      if (data.success) {
        // 5. Typewriter response
        appendTypewriterMessage(data.reply, data.embed);
      } else {
        appendMessage("mentor", "I'm sorry, I encountered an issue interpreting your request. Please try again.");
      }
    } catch (e) {
      removeTypingIndicator();
      appendMessage("mentor", "Network exception. Please check your local server connection.");
    }
  });

  // --- Register Quick Prompt Chips ---
  quickPrompts.forEach(btn => {
    btn.addEventListener("click", () => {
      userInput.value = btn.dataset.query;
      chatForm.dispatchEvent(new Event("submit"));
    });
  });
});
