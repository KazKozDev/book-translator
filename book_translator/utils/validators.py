"""
Validation Utilities
====================
Functions for validating input data.
"""

import os
from typing import List, Optional, Tuple

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from book_translator.config import SUPPORTED_LANGUAGES, config


def validate_file(file: FileStorage) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate an uploaded file.

    Args:
        file: The uploaded file

    Returns:
        Tuple of (is_valid, error_message, secure_filename)
    """
    if not file or not file.filename:
        return False, "No file provided", None

    # Secure the filename
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename", None

    # Check extension
    allowed_extensions = config.file.allowed_extensions
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        return (
            False,
            f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
            None,
        )

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    max_size = config.file.max_file_size_bytes
    if file_size > max_size:
        max_mb = max_size // (1024 * 1024)
        return False, f"File too large. Maximum size: {max_mb}MB", None

    return True, None, filename


def validate_language(lang_code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a language code.

    Args:
        lang_code: The language code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not lang_code:
        return False, "Language code is required"

    if lang_code not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES.keys())
        return False, f"Unsupported language: {lang_code}. Supported: {supported}"

    return True, None


def validate_model_name(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a model name.

    Args:
        model_name: The model name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_name:
        return False, "Model name is required"

    # Basic validation - model name should be alphanumeric with some special
    # chars. Includes "/" since models pulled from Hugging Face are named
    # like "hf.co/mradermacher/Qwen3-8B-heretic-i1-GGUF:Q4_0".
    import re

    if not re.match(r"^[a-zA-Z0-9._:/-]+$", model_name):
        return False, "Invalid model name format"

    return True, None


def validate_translation_request(
    source_lang: str, target_lang: str, model_name: str
) -> Tuple[bool, List[str]]:
    """
    Validate a complete translation request.

    Args:
        source_lang: Source language code
        target_lang: Target language code
        model_name: Model name

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    valid, error = validate_language(source_lang)
    if not valid:
        errors.append(f"Source language: {error}")

    valid, error = validate_language(target_lang)
    if not valid:
        errors.append(f"Target language: {error}")

    valid, error = validate_model_name(model_name)
    if not valid:
        errors.append(f"Model: {error}")

    return len(errors) == 0, errors
