"""
Text Processing Utilities
=========================
Functions for processing and manipulating text.
"""

import re
from typing import List, Tuple

from book_translator.config import config
from book_translator.utils.logging import debug_print


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and standardizing line endings.
    Preserves paragraph structure and dialogue formatting.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive blank lines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, max_length: int = None) -> List[str]:
    """
    Split text into smaller chunks for translation.
    Ensures clean boundaries at paragraph level when possible.
    Preserves original formatting (paragraphs, dialogue, etc.)

    Args:
        text: Text to split
        max_length: Maximum chunk length (uses config if not specified)

    Returns:
        List of text chunks
    """
    if max_length is None:
        max_length = config.translation.max_prompt_length

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        para_length = len(paragraph)

        # If single paragraph is too long, split by sentences
        if para_length > max_length:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split long paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sentence_chunk = []
            sentence_length = 0

            for sentence in sentences:
                if sentence_length + len(sentence) > max_length and sentence_chunk:
                    chunks.append(" ".join(sentence_chunk))
                    sentence_chunk = []
                    sentence_length = 0
                sentence_chunk.append(sentence)
                sentence_length += len(sentence) + 1

            if sentence_chunk:
                chunks.append(" ".join(sentence_chunk))
            continue

        # Check if adding this paragraph would exceed the limit
        if current_length + para_length + 2 > max_length and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(paragraph)
        current_length += para_length + 2

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    result = chunks if chunks else [text]

    # Debug output for chunking
    debug_print(f"[CHUNKING] Split text into {len(result)} chunks", "DEBUG", "TEXT")
    debug_print(
        f"  Input: {len(text)} chars, {len(paragraphs)} paragraphs", "DEBUG", "TEXT"
    )
    debug_print(f"  Max chunk size: {max_length} chars", "DEBUG", "TEXT")
    for i, chunk in enumerate(result):
        preview = chunk[:60].replace("\n", " ")
        debug_print(
            f"  Chunk {i+1}: {len(chunk)} chars - {preview}...", "DEBUG", "TEXT"
        )

    return result


def clean_translation_response(translation: str, previous_chunk: str = "") -> str:
    """
    Clean the LLM response to remove unwanted content.
    Removes thinking tags, instruction echoes, and other artifacts.

    Args:
        translation: Raw translation from LLM
        previous_chunk: Previous chunk for detecting repetition

    Returns:
        Cleaned translation
    """
    if not translation:
        return ""

    original_len = len(translation)
    translation = translation.strip()
    debug_print(f"[CLEAN] Starting cleanup of {original_len} chars", "DEBUG", "TEXT")

    # ========== PHASE 1: Remove thinking/reasoning tags ==========
    # Remove <think> tags and their content (common in reasoning models like DeepSeek)
    translation = re.sub(
        r"<think>.*?</think>", "", translation, flags=re.DOTALL | re.IGNORECASE
    )
    translation = re.sub(
        r"<thinking>.*?</thinking>", "", translation, flags=re.DOTALL | re.IGNORECASE
    )
    translation = re.sub(
        r"<reasoning>.*?</reasoning>", "", translation, flags=re.DOTALL | re.IGNORECASE
    )
    translation = re.sub(
        r"<reflection>.*?</reflection>",
        "",
        translation,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove unclosed thinking tags (if model was cut off)
    translation = re.sub(
        r"<think>.*$", "", translation, flags=re.DOTALL | re.IGNORECASE
    )
    translation = re.sub(
        r"<thinking>.*$", "", translation, flags=re.DOTALL | re.IGNORECASE
    )

    translation = translation.strip()

    # ========== PHASE 2: Remove instruction echoes ==========
    # These are patterns where the model echoes parts of the prompt
    unwanted_patterns = [
        # English instruction echoes
        r"IMPORTANT:\s*Return ONLY the translation[^\n]*\n*",
        r"IMPORTANT:\s*Devolver SOLO la traducción[^\n]*\n*",
        r"IMPORTANTE:\s*Devolver SOLO la traducción[^\n]*\n*",
        r"IMPORTANTE:\s*Devuelve SOLO la traducción[^\n]*\n*",
        r"IMPORTANTE:\s*Return ONLY[^\n]*\n*",
        r"Return ONLY the translation[^\n]*\n*",
        r"Devolver SOLO la traducción[^\n]*\n*",
        r"No repita contenido previo[^\n]*\n*",
        r"Do not repeat previous content[^\n]*\n*",
        # Section headers from prompts
        r"TEXT TO TRANSLATE:.*?\n+",
        r"TEXTO A TRADUCIR:.*?\n+",
        r"ORIGINAL TEXT:.*?\n+",
        r"TEXTO ORIGINAL:.*?\n+",
        r"CONTEXT \(previous translation[^\)]*\):.*?\n+",
        r"CONTEXTO \(traducción anterior[^\)]*\):.*?\n+",
        # Requirements section echoes
        r"REQUIREMENTS:.*?(?=\n[A-Z]|\n\n|\Z)",
        r"REQUISITOS:.*?(?=\n[A-Z]|\n\n|\Z)",
        r"GENRE:.*?\n+",
        r"GÉNERO:.*?\n+",
        # Common LLM prefixes
        r"^\s*Here is the translation:?\s*\n*",
        r"^\s*Here\'s the translation:?\s*\n*",
        r"^\s*Aquí está la traducción:?\s*\n*",
        r"^\s*La traducción es:?\s*\n*",
        r"^\s*Translation:?\s*\n*",
        r"^\s*Traducción:?\s*\n*",
        r"^\s*Translated text:?\s*\n*",
        r"^\s*Texto traducido:?\s*\n*",
        r"^\s*\*\*Translation:?\*\*\s*\n*",
        r"^\s*\*\*Traducción:?\*\*\s*\n*",
        # Markdown artifacts
        r"^\s*---+\s*\n*",
        r"^\s*\*\*\*+\s*\n*",
        r"^\s*```[a-z]*\s*\n*",
        r"\s*```\s*$",
        # Notes/explanations wrapped in brackets (bounded, safe anywhere)
        r"\n+\[Note:.*?\]",
        r"\n+\[Nota:.*?\]",
    ]

    for pattern in unwanted_patterns:
        translation = re.sub(
            pattern, "", translation, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

    translation = translation.strip()

    # ========== PHASE 2B: Remove a trailing meta-commentary note ==========
    # Models sometimes append a short "Note: I kept the names as-is" remark
    # after the real translation. Only strip it if it is the LAST block of
    # text and looks like a short remark - never scan mid-text, since a
    # story can legitimately contain a paragraph starting with "Nota:"
    # (a letter, a message, etc.) and everything after it must be kept.
    _note_re = re.compile(r"^\*{0,2}(?:Note|Nota)\s*:\s*.+$", re.IGNORECASE | re.DOTALL)
    for _sep in ("\n\n", "\n"):
        _parts = translation.split(_sep)
        if len(_parts) > 1:
            _last = _parts[-1].strip()
            if _note_re.match(_last) and len(_last) < 300:
                translation = _sep.join(_parts[:-1]).strip()
                break

    # ========== PHASE 3: Remove prompt context that leaked ==========
    # Sometimes the model includes the "previous translation for continuity" context
    context_markers = [
        r"^.*?(?:for continuity|para continuidad):?\s*\n+",
        r"^.*?(?:previous translation|traducción anterior):?\s*\n+",
        r"INICIOS?\s*\n+",  # Sometimes models output this marker
        r"BEGINNINGS?\s*\n+",
    ]

    for pattern in context_markers:
        translation = re.sub(
            pattern, "", translation, flags=re.IGNORECASE | re.MULTILINE
        )

    translation = translation.strip()

    # ========== PHASE 4: Remove quotes wrapping ==========
    # Remove leading/trailing quotes that might have been added
    if len(translation) > 2:
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]
        if translation.startswith("'") and translation.endswith("'"):
            translation = translation[1:-1]
        if translation.startswith("«") and translation.endswith("»"):
            translation = translation[1:-1]

    # ========== PHASE 5: Remove repetition from previous chunk ==========
    if previous_chunk and len(previous_chunk) > 50:
        prev_lines = previous_chunk.strip().split("\n")

        # Check if translation starts with content from previous chunk
        for i in range(min(5, len(prev_lines))):
            check_text = "\n".join(prev_lines[-(i + 1) :]).strip()
            if len(check_text) > 50 and translation.startswith(check_text):
                translation = translation[len(check_text) :].strip()
                break

        # Check for partial sentence duplicates
        if len(prev_lines) > 0:
            last_prev_line = prev_lines[-1].strip()
            if len(last_prev_line) > 30:
                for check_len in range(len(last_prev_line), 30, -10):
                    check_segment = last_prev_line[-check_len:]
                    if translation.startswith(check_segment):
                        translation = translation[len(check_segment) :].strip()
                        break

    final_len = len(translation.strip())
    if original_len != final_len:
        debug_print(
            f"[CLEAN] Removed {original_len - final_len} chars ({original_len} -> {final_len})",
            "DEBUG",
            "TEXT",
        )
    else:
        debug_print(f"[CLEAN] No changes needed ({final_len} chars)", "DEBUG", "TEXT")

    return translation.strip()


def clean_for_epub(text: str) -> str:
    """
    Clean text specifically for EPUB output.
    Removes problematic characters and ensures valid XHTML content.

    Args:
        text: Text to clean

    Returns:
        EPUB-safe text
    """
    if not text:
        return ""

    # Remove null characters and other control characters (except newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Remove any remaining thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove instruction artifacts
    instruction_patterns = [
        r"IMPORTANT:.*?(?:\n|$)",
        r"IMPORTANTE:.*?(?:\n|$)",
        r"REQUIREMENTS:.*?(?:\n\n|\Z)",
        r"REQUISITOS:.*?(?:\n\n|\Z)",
        r"\[⚠️[^\]]*\]",
        r"Note:.*?(?:\n|$)",
        r"Nota:.*?(?:\n|$)",
    ]

    for pattern in instruction_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Escape special XML characters (for XHTML compatibility)
    # But preserve already-escaped entities
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", text)

    # Ensure proper paragraph separation
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def preserve_formatting(original: str, translated: str) -> str:
    """
    Attempt to preserve the formatting structure of the original text
    in the translated version.

    Args:
        original: Original text with formatting
        translated: Translated text

    Returns:
        Translated text with preserved formatting
    """
    # Count leading/trailing whitespace in original
    leading_spaces = len(original) - len(original.lstrip())
    trailing_spaces = len(original) - len(original.rstrip())

    # Preserve leading newlines
    leading_newlines = 0
    for char in original:
        if char == "\n":
            leading_newlines += 1
        else:
            break

    # Apply formatting
    result = translated.strip()
    if leading_newlines > 0:
        result = "\n" * leading_newlines + result
    if leading_spaces > leading_newlines:
        result = " " * (leading_spaces - leading_newlines) + result
    if trailing_spaces > 0:
        result = result + " " * trailing_spaces

    return result


def extract_proper_nouns(text: str) -> List[str]:
    """
    Extract proper nouns (capitalized words/phrases) from text.

    Args:
        text: Text to analyze

    Returns:
        List of unique proper nouns
    """
    # Match capitalized words that are not at sentence start
    pattern = r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    nouns = re.findall(pattern, text, re.MULTILINE)
    return list(set(nouns))


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """Count characters in text."""
    if include_spaces:
        return len(text)
    return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
