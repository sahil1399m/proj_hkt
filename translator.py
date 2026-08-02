"""
translator.py — Gemini Flash Marathi Translation
"""

from __future__ import annotations
import os
import logging
import time
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

_PRESERVE_TERMS = [
    "HTE", "DTE", "AICTE", "UGC", "MSBTE", "DPIIT", "MSME",
    "Maharashtra", "CAP", "MHT-CET", "JEE", "NEET", "MCA", "MBA",
    "B.Tech", "M.Tech", "BE", "ME", "Ph.D", "TFWS", "EBC", "OBC",
    "SC", "ST", "VJ", "DT", "NT", "SEBC", "EWS",
    "Fee Regulating Authority", "FRA",
]


def translate(text: str, language: str = "marathi") -> str:
    """
    Translate English text to the requested language using Gemini.
    Tries new SDK first, falls back to old SDK.
    Returns original text on failure.
    """
    if not text or not text.strip():
        return text

    # Split long texts into chunks to avoid token limits
    MAX_CHARS = 3000
    if len(text) > MAX_CHARS:
        chunks = _split_text(text, MAX_CHARS)
        translated_chunks = [_translate_chunk(c, language) for c in chunks]
        return "\n\n".join(translated_chunks)

    return _translate_chunk(text, language)


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks or [text]


def _translate_chunk(text: str, language: str = "marathi") -> str:
    """Translate a single chunk — tries new SDK then old SDK."""
    preserve_note = ", ".join(_PRESERVE_TERMS[:15])

    prompt = f"""You are a professional translator. Translate the following English text to {language.capitalize()} (Devanagari script).

STRICT RULES:
1. Output ONLY the translated text. Do not include English, explanations, headings, or any preamble.
2. Keep these terms exactly as-is in English: {preserve_note}, and any similar education/government acronyms.
3. Keep all numbers, percentages, fees (₹), dates, and years unchanged.
4. Keep citation markers [DOC 1], [WEB 2] exactly as written.
5. Keep URL links unchanged.
6. Translate naturally and fluently — not word-for-word.

Text to translate:
{text}

{language.capitalize()} translation:"""

    # ── Try new google-genai SDK ──────────────────────────────────────────────
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        result = response.text.strip() if response.text else ""
        if result and _looks_like_devanagari(result):
            logger.info("Translation OK (new SDK): %d chars → %d chars", len(text), len(result))
            return result
        elif result:
            logger.warning("Translation returned but doesn't look like Devanagari — retrying with old SDK")
    except Exception as exc:
        logger.warning("New SDK translation failed: %s", exc)

    # ── Fallback: old google-generativeai SDK ─────────────────────────────────
    try:
        import google.generativeai as genai_old
        genai_old.configure(api_key=GOOGLE_API_KEY)
        model = genai_old.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        result = response.text.strip() if response.text else ""
        if result and _looks_like_devanagari(result):
            logger.info("Translation OK (old SDK): %d chars → %d chars", len(text), len(result))
            return result
        logger.warning("Old SDK translation didn't return valid Devanagari text")
    except Exception as exc:
        logger.warning("Old SDK translation failed: %s", exc)

    # ── Fallback: Groq ────────────────────────────────────────────────────────
    try:
        from groq import Groq
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional English translator. "
                            "Translate to the language requested by the user. "
                            "Output ONLY the translated text in Devanagari script. "
                            "Keep acronyms like HTE, DTE, AICTE, OBC, SC, ST, EBC, CAP, MHT-CET, "
                            "TFWS, EWS in English. Keep numbers, dates, fees unchanged. "
                            "Keep [DOC N] and [WEB N] markers unchanged."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Translate to {language}:\n\n{text}"
                    },
                ],
                max_tokens=2048,
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
            if result and _looks_like_devanagari(result):
                logger.info("Translation OK (Groq fallback)")
                return result
    except Exception as exc:
        logger.warning("Groq translation fallback failed: %s", exc)

    logger.error("All translation backends failed — returning original text")
    return text


def _looks_like_devanagari(text: str) -> bool:
    """
    Check if text contains Devanagari characters.
    Devanagari Unicode block: U+0900–U+097F
    Returns True if at least 10% of alpha chars are Devanagari.
    """
    if not text:
        return False
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    ratio = devanagari_count / total_alpha
    return ratio > 0.1  # at least 10% Devanagari chars

def translate_to_marathi(text: str) -> str:
    return translate(text, "marathi")


def translate_to_hindi(text: str) -> str:
    return translate(text, "hindi")

def detect_language(text: str) -> str:
    """Detect language of input text."""
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f'Detect the language. Reply with ONE word only: english, marathi, hindi, or other.\n\nText: "{text[:200]}"',
        )
        lang = response.text.strip().lower()
        return lang if lang in {"english", "marathi", "hindi", "other"} else "english"
    except Exception:
        return "english"
