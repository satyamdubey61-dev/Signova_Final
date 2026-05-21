import threading
from typing import Any, Optional
from utils.logger import logger


class SpeechService:
    def __init__(self) -> None:
        """Initializes the backend Text-to-Speech service."""
        self.engine: Any = None
        self.enabled: bool = False

        # Try to initialize pyttsx3
        try:
            import pyttsx3
            # Initialize inside a thread-safe property or create a new engine when needed
            self.engine = pyttsx3.init()
            # Set standard rate and volume
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.enabled = True
            logger.info("Backend SpeechService (pyttsx3) initialized successfully.")
        except ImportError:
            logger.warning("pyttsx3 library not found. Backend text-to-speech will operate in fallback mock mode (logging only).")
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {e}. Operating in fallback mock mode.")

    def _speak_worker(self, text: str) -> None:
        """Worker function to speak text in a separate thread."""
        try:
            import pyttsx3
            # Note: pyttsx3 engines must be initialized and run in the same thread
            engine: Any = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in threaded text-to-speech: {e}")

    def speak(self, text: str) -> None:
        """Speaks the provided text asynchronously without blocking the main thread."""
        if not text:
            return

        logger.info(f"[SPEECH] Speaking: '{text}'")

        if not self.enabled:
            # Fallback mock mode: already logged
            return

        try:
            # Run in a separate thread so it doesn't block Flask requests
            thread: threading.Thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
            thread.start()
        except Exception as e:
            logger.error(f"Failed to start speech thread: {e}")


# Instantiate a global service instance
speech_service: SpeechService = SpeechService()
