from typing import Dict, Optional

class TranslationService:
    # 1. Dictionary-based offline translations for supported languages
    TRANSLATIONS: Dict[str, Dict[str, str]] = {
        "hello": {
            "english": "Hello",
            "hindi": "नमस्ते",
            "marathi": "नमस्कार",
            "konkani": "नमस्कार",
            "tamil": "வணக்கம்",
            "punjabi": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
            "gujarati": "નમસ્તે",
            "bhojpuri": "प्रणाम"
        },
        "yes": {
            "english": "Yes",
            "hindi": "हाँ",
            "marathi": "हो",
            "konkani": "हय",
            "tamil": "ஆம்",
            "punjabi": "ਹਾਂ",
            "gujarati": "હા",
            "bhojpuri": "हाँ"
        },
        "no": {
            "english": "No",
            "hindi": "नहीं",
            "marathi": "नाही",
            "konkani": "ना",
            "tamil": "இல்லை",
            "punjabi": "ਨਹੀਂ",
            "gujarati": "ના",
            "bhojpuri": "ना"
        },
        "thankyou": {
            "english": "Thank You",
            "hindi": "धन्यवाद",
            "marathi": "धन्यवाद",
            "konkani": "देव बरें करूं",
            "tamil": "நன்றி",
            "punjabi": "ਧੰਨਵਾਦ",
            "gujarati": "આભાર",
            "bhojpuri": "धन्यवाद"
        },
        "sorry": {
            "english": "Sorry",
            "hindi": "माफ़ करें",
            "marathi": "माफ करा",
            "konkani": "माफ करात",
            "tamil": "மன்னிக்கவும்",
            "punjabi": "ਮਾਫ਼ ਕਰਨਾ",
            "gujarati": "માફ કરજો",
            "bhojpuri": "माफ करीं"
        },
        "help": {
            "english": "Help",
            "hindi": "मदद",
            "marathi": "मदत",
            "konkani": "कुमक",
            "tamil": "உதவி",
            "punjabi": "ਮਦਦ",
            "gujarati": "મદદ",
            "bhojpuri": "मदद"
        },
        "iloveyou": {
            "english": "I Love You",
            "hindi": "मैं तुमसे प्यार करता हूँ",
            "marathi": "मी तुझ्यावर प्रेम करतो",
            "konkani": "हांव तुका मोग करतां",
            "tamil": "நான் உன்னை காதலிக்கிறேன்",
            "punjabi": "ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ",
            "gujarati": "હું તને પ્રેમ કરું છું",
            "bhojpuri": "हम तोहरा से प्यार करीं ला"
        }
    }

    @classmethod
    def translate(cls, text: str, to_lang: str) -> str:
        """
        Translate a word or alphabet offline using static dictionaries.
        Guarantees that alphabets are NOT translated into regional scripts, keeping as is.
        """
        clean_text = text.lower().replace(" ", "").replace("-", "")
        lang_key = to_lang.lower().strip()

        # 2. ALPHABET HANDLING: Do NOT translate single alphabets, keep them as is (e.g. A -> A, C -> C)
        if len(clean_text) == 1 and clean_text.isalpha():
            return text.upper()

        # Check if the word is in our dictionary
        word_translations = cls.TRANSLATIONS.get(clean_text)
        if word_translations:
            translated_word = word_translations.get(lang_key)
            if translated_word:
                return translated_word

        # Fallback to English/original text if not matched
        return text
