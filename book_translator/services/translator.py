"""
Book Translator Service
=======================
Main translation service with two-stage translation approach.
"""
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

from book_translator.config import config
from book_translator.config.constants import TranslationStatus, SUPPORTED_LANGUAGES
from book_translator.models.translation import TranslationProgress
from book_translator.services.cache_service import TranslationCache, get_cache
from book_translator.services.ollama_client import OllamaClient, get_ollama_client
from book_translator.services.terminology import TerminologyManager
from book_translator.utils.language_detection import is_likely_translated
from book_translator.utils.logging import debug_print, get_logger
from book_translator.utils.text_processing import (
    clean_translation_response,
    normalize_text,
    split_into_chunks,
)


@dataclass
class ChunkResult:
    """Result of translating a single chunk."""
    chunk_index: int
    original: str
    translation: str
    success: bool
    error: Optional[str] = None
    from_cache: bool = False


def _normalize_custom_instructions(custom_instructions: str) -> str:
    """Normalize user-provided translation instructions."""
    if not custom_instructions:
        return ""
    return custom_instructions.strip()


class BookTranslator:
    """
    Two-stage book translator.
    
    Stage 1: Primary translation
    Stage 2: Reflection and improvement
    """
    
    def __init__(
        self, 
        model_name: str = None, 
        chunk_size: int = None,
        ollama_client: OllamaClient = None,
        cache: TranslationCache = None
    ):
        self.model_name = model_name or config.ollama.default_model
        self.chunk_size = chunk_size or config.translation.chunk_size
        self.client = ollama_client or get_ollama_client()
        self.cache = cache or get_cache()
        self.terminology = TerminologyManager()
        self.logger = get_logger().translation_logger
    
    def _build_stage1_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        previous_chunk: str = "",
        genre: str = "general",
        custom_instructions: str = ""
    ) -> str:
        """Build prompt for stage 1 (primary translation)."""
        context_section = ""
        if previous_chunk:
            context_preview = previous_chunk[-200:] if len(previous_chunk) > 200 else previous_chunk
            context_section = f"""
CONTEXT (for continuity only - do NOT include in output):
{context_preview}
---
"""
        
        terminology_section = self.terminology.get_context_for_prompt()
        if terminology_section:
            terminology_section = f"\n{terminology_section}\n"

        custom_instructions = _normalize_custom_instructions(custom_instructions)
        instructions_section = ""
        if custom_instructions:
            instructions_section = f"""
USER TRANSLATION INSTRUCTIONS:
{custom_instructions}
"""
        return f"""You are a professional literary translator. Translate the following {SUPPORTED_LANGUAGES[source_lang]} ({source_lang}) text to {SUPPORTED_LANGUAGES[target_lang]} ({target_lang}).

CRITICAL RULES:
1. Output ONLY the translated text - nothing else
2. PRESERVE all original formatting: paragraphs, line breaks, dialogue formatting, indentation
3. Do NOT add notes, explanations, comments, or headers
4. Do NOT repeat the prompt or instructions
5. Do NOT include "Translation:", "Here is:", or similar prefixes
6. Do NOT add [brackets] or markers of any kind
7. Maintain the author's style, tone, and voice exactly
8. Keep proper nouns and names consistent
{terminology_section}{instructions_section}{context_section}
TEXT TO TRANSLATE:
{text}

OUTPUT (translated text only, preserving all formatting):"""
    
    def _build_stage2_prompt(
        self,
        original: str,
        draft: str,
        source_lang: str,
        target_lang: str,
        genre: str = "general",
        custom_instructions: str = ""
    ) -> str:
        """Build prompt for stage 2 (reflection and improvement)."""
        custom_instructions = _normalize_custom_instructions(custom_instructions)
        instructions_section = ""
        if custom_instructions:
            instructions_section = f"""
USER TRANSLATION INSTRUCTIONS:
{custom_instructions}
"""

        return f"""You are a professional literary editor. Review and improve this {target_lang} translation.

ORIGINAL ({source_lang}):
{original}

DRAFT TRANSLATION ({target_lang}):
{draft}

TASK: Review for accuracy, fluency, style preservation, and consistency.
{instructions_section}

CRITICAL RULES:
1. Output ONLY the improved translated text - nothing else
2. PRESERVE all original formatting: paragraphs, line breaks, dialogue formatting
3. Do NOT add notes, explanations, or comments
4. Do NOT include prefixes like "Improved translation:" or similar
5. If the draft is already good, return it unchanged

OUTPUT (final translation only):"""
    
    def _translate_chunk_stage1(
        self,
        chunk: str,
        source_lang: str,
        target_lang: str,
        previous_chunk: str = "",
        genre: str = "general",
        custom_instructions: str = ""
    ) -> str:
        """Translate a single chunk (stage 1)."""
        prompt = self._build_stage1_prompt(
            chunk, source_lang, target_lang, previous_chunk, genre, custom_instructions
        )

        # Debug: Show prompt being sent
        debug_print(f"[PROMPT S1] Length: {len(prompt)} chars", 'DEBUG', 'LLM')
        debug_print(f"[PROMPT S1] Input text ({len(chunk)} chars): {chunk[:150]}...", 'DEBUG', 'LLM')

        last_cleaned = None
        for attempt in range(config.translation.max_retries):
            debug_print(f"[LLM] Sending request to {self.model_name} (attempt {attempt + 1})", 'INFO', 'LLM')
            start_time = time.time()

            response = self.client.generate(prompt, model=self.model_name)

            elapsed = time.time() - start_time
            debug_print(f"[LLM] Response received in {elapsed:.2f}s", 'INFO', 'LLM')

            if response.success and response.text:
                debug_print(f"[RAW RESPONSE] Length: {len(response.text)} chars", 'DEBUG', 'LLM')
                debug_print(f"[RAW RESPONSE] Preview: {response.text[:200]}...", 'DEBUG', 'LLM')

                cleaned = clean_translation_response(response.text, previous_chunk)
                last_cleaned = cleaned

                debug_print(f"[CLEANED] Length: {len(cleaned)} chars", 'DEBUG', 'LLM')
                debug_print(f"[CLEANED] Preview: {cleaned[:200]}...", 'DEBUG', 'LLM')

                # Validate translation
                if is_likely_translated(chunk, cleaned, source_lang, target_lang):
                    debug_print(f"[VALIDATION] PASSED - Translation accepted", 'INFO', 'LLM')
                    return cleaned
                else:
                    debug_print(f"[VALIDATION] FAILED - Translation rejected (attempt {attempt + 1})", 'WARNING', 'LLM')
                    self.logger.warning(f"Translation validation failed, attempt {attempt + 1}")
            else:
                debug_print(f"[LLM ERROR] {response.error}", 'ERROR', 'LLM')
                self.logger.error(f"Generation failed: {response.error}")

            if attempt < config.translation.max_retries - 1:
                time.sleep(config.translation.retry_delay * (attempt + 1))

        debug_print(f"[TRANSLATION] FAILED after {config.translation.max_retries} attempts", 'ERROR', 'LLM')
        # Cambiado: devolver el último intento aunque no pase la validación
        return last_cleaned if last_cleaned is not None else f"[TRANSLATION_FAILED: {chunk[:50]}...]"
    
    def _translate_chunk_stage2(
        self,
        original: str,
        draft: str,
        source_lang: str,
        target_lang: str,
        genre: str = "general",
        custom_instructions: str = ""
    ) -> str:
        """Improve a translation (stage 2)."""
        prompt = self._build_stage2_prompt(
            original, draft, source_lang, target_lang, genre, custom_instructions
        )

        # Debug: Show prompt being sent
        debug_print(f"[PROMPT S2] Length: {len(prompt)} chars", 'DEBUG', 'LLM')
        debug_print(f"[PROMPT S2] Original ({len(original)} chars): {original[:100]}...", 'DEBUG', 'LLM')
        debug_print(f"[PROMPT S2] Draft ({len(draft)} chars): {draft[:100]}...", 'DEBUG', 'LLM')

        for attempt in range(config.translation.max_retries):
            debug_print(f"[LLM S2] Sending refinement request (attempt {attempt + 1})", 'INFO', 'LLM')
            start_time = time.time()

            response = self.client.generate(prompt, model=self.model_name)

            elapsed = time.time() - start_time
            debug_print(f"[LLM S2] Response received in {elapsed:.2f}s", 'INFO', 'LLM')

            if response.success and response.text:
                debug_print(f"[RAW S2] Length: {len(response.text)} chars", 'DEBUG', 'LLM')
                debug_print(f"[RAW S2] Preview: {response.text[:200]}...", 'DEBUG', 'LLM')

                cleaned = clean_translation_response(response.text, "")

                debug_print(f"[CLEANED S2] Length: {len(cleaned)} chars", 'DEBUG', 'LLM')
                debug_print(f"[CLEANED S2] Preview: {cleaned[:200]}...", 'DEBUG', 'LLM')

                if is_likely_translated(original, cleaned, source_lang, target_lang):
                    debug_print(f"[VALIDATION S2] PASSED - Refinement accepted", 'INFO', 'LLM')
                    return cleaned
                else:
                    debug_print(f"[VALIDATION S2] FAILED - Using original draft", 'WARNING', 'LLM')
                    self.logger.warning(f"Stage 2 validation failed, using draft")
                    return draft
            else:
                debug_print(f"[LLM S2 ERROR] {response.error}", 'ERROR', 'LLM')
                self.logger.error(f"Stage 2 generation failed: {response.error}")

            if attempt < config.translation.max_retries - 1:
                time.sleep(config.translation.retry_delay * (attempt + 1))

        debug_print(f"[S2 FALLBACK] Using draft after {config.translation.max_retries} failed attempts", 'WARNING', 'LLM')
        # Fall back to draft if stage 2 fails
        return draft
    
    def _get_context_hash(self, previous_chunk: str, custom_instructions: str = "") -> str:
        """Generate cache context hash from continuity context and user instructions."""
        hash_input = "\n".join(
            part for part in [previous_chunk.strip(), _normalize_custom_instructions(custom_instructions)] if part
        )
        if not hash_input:
            return ""
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:config.cache.context_hash_length]
    
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation_id: int = None,
        genre: str = "general",
        custom_instructions: str = ""
    ) -> Generator[TranslationProgress, None, None]:
        """
        Translate text using the two-stage approach.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            translation_id: Optional ID for database tracking
            genre: Genre of the text
        
        Yields:
            TranslationProgress updates
        """
        # Normalize and split text
        text = normalize_text(text)
        chunks = split_into_chunks(text)
        total_chunks = len(chunks)

        self.logger.info(f"Starting translation: {total_chunks} chunks, {source_lang} -> {target_lang}")

        # Detailed debug output
        debug_print(f"{'='*60}", 'INFO', 'TRANS')
        debug_print(f"[TRANSLATION START]", 'INFO', 'TRANS')
        debug_print(f"  Model: {self.model_name}", 'INFO', 'TRANS')
        debug_print(f"  Source: {source_lang} -> Target: {target_lang}", 'INFO', 'TRANS')
        debug_print(f"  Total text length: {len(text)} chars", 'INFO', 'TRANS')
        debug_print(f"  Chunks: {total_chunks}", 'INFO', 'TRANS')
        debug_print(f"{'='*60}", 'INFO', 'TRANS')

        # Show chunk breakdown
        for idx, chunk in enumerate(chunks):
            preview = chunk[:80].replace('\n', ' ')
            debug_print(f"  [CHUNK {idx+1}] {len(chunk)} chars: {preview}...", 'DEBUG', 'TRANS')
        
        # Stage 1: Primary translations
        draft_translations: List[str] = []
        
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            previous_chunk = draft_translations[-1] if draft_translations else ""
            context_hash = self._get_context_hash(previous_chunk, custom_instructions)

            debug_print(f"", 'INFO', 'TRANS')
            debug_print(f"{'='*60}", 'INFO', 'TRANS')
            debug_print(f"[STAGE 1] Chunk {chunk_num}/{total_chunks}", 'INFO', 'TRANS')
            debug_print(f"  Chunk size: {len(chunk)} chars", 'DEBUG', 'TRANS')
            debug_print(f"  Context hash: {context_hash[:16] if context_hash else 'none'}...", 'DEBUG', 'TRANS')

            # Check cache
            cached = self.cache.get(
                chunk, source_lang, target_lang,
                f"{self.model_name}_stage1", context_hash
            )

            if cached and not cached['translated_text'].startswith('[TRANSLATION_FAILED'):
                draft = cached['machine_translation'] or cached['translated_text']
                if is_likely_translated(chunk, draft, source_lang, target_lang):
                    debug_print(f"[CACHE HIT] Using cached translation ({len(draft)} chars)", 'INFO', 'CACHE')
                    debug_print(f"  Cached text: {draft[:100]}...", 'DEBUG', 'CACHE')
                    draft_translations.append(draft)

                    yield TranslationProgress(
                        progress=(chunk_num / (total_chunks * 2)) * 100,
                        stage='primary_translation',
                        original_text='\n\n'.join(chunks),
                        machine_translation='\n\n'.join(draft_translations),
                        current_chunk=chunk_num,
                        total_chunks=total_chunks * 2
                    )
                    continue

            debug_print(f"[CACHE MISS] Requesting new translation", 'INFO', 'CACHE')

            # Translate
            draft = self._translate_chunk_stage1(
                chunk, source_lang, target_lang, previous_chunk, genre, custom_instructions
            )

            if not draft.startswith('[TRANSLATION_FAILED'):
                debug_print(f"[CACHE SAVE] Storing translation ({len(draft)} chars)", 'DEBUG', 'CACHE')
                # Cache successful translation
                self.cache.set(
                    chunk, draft, draft,
                    source_lang, target_lang,
                    f"{self.model_name}_stage1", context_hash
                )

            draft_translations.append(draft)
            progress_pct = (chunk_num / (total_chunks * 2)) * 100
            debug_print(f"[PROGRESS] {progress_pct:.1f}% complete", 'INFO', 'TRANS')

            yield TranslationProgress(
                progress=progress_pct,
                stage='primary_translation',
                original_text='\n\n'.join(chunks),
                machine_translation='\n\n'.join(draft_translations),
                current_chunk=chunk_num,
                total_chunks=total_chunks * 2
            )

            # Delay between chunks
            if config.translation.chunk_delay > 0:
                time.sleep(config.translation.chunk_delay)
        
        # Stage 2: Reflection and improvement
        debug_print(f"", 'INFO', 'TRANS')
        debug_print(f"{'='*60}", 'INFO', 'TRANS')
        debug_print(f"[STAGE 2 START] Beginning refinement phase", 'INFO', 'TRANS')
        debug_print(f"{'='*60}", 'INFO', 'TRANS')

        final_translations: List[str] = []

        for i, (chunk, draft) in enumerate(zip(chunks, draft_translations)):
            chunk_num = i + 1
            previous_final = final_translations[-1] if final_translations else ""
            context_hash = self._get_context_hash(previous_final, custom_instructions)

            debug_print(f"", 'INFO', 'TRANS')
            debug_print(f"{'='*60}", 'INFO', 'TRANS')
            debug_print(f"[STAGE 2] Chunk {chunk_num}/{total_chunks}", 'INFO', 'TRANS')
            debug_print(f"  Original: {len(chunk)} chars", 'DEBUG', 'TRANS')
            debug_print(f"  Draft: {len(draft)} chars", 'DEBUG', 'TRANS')

            # Check cache for stage 2
            cached = self.cache.get(
                chunk, source_lang, target_lang,
                f"{self.model_name}_stage2", context_hash
            )

            if cached and not cached['translated_text'].startswith('[TRANSLATION_FAILED'):
                final = cached['translated_text']
                if is_likely_translated(chunk, final, source_lang, target_lang):
                    debug_print(f"[CACHE HIT S2] Using cached refinement ({len(final)} chars)", 'INFO', 'CACHE')
                    debug_print(f"  Cached text: {final[:100]}...", 'DEBUG', 'CACHE')
                    final_translations.append(final)

                    yield TranslationProgress(
                        progress=((chunk_num + total_chunks) / (total_chunks * 2)) * 100,
                        stage='reflection_improvement',
                        original_text='\n\n'.join(chunks),
                        machine_translation='\n\n'.join(draft_translations),
                        translated_text='\n\n'.join(final_translations),
                        current_chunk=chunk_num + total_chunks,
                        total_chunks=total_chunks * 2
                    )
                    continue

            # Skip stage 2 if stage 1 failed
            if draft.startswith('[TRANSLATION_FAILED'):
                debug_print(f"[SKIP S2] Stage 1 failed, skipping refinement", 'WARNING', 'TRANS')
                final_translations.append(draft)
            else:
                debug_print(f"[CACHE MISS S2] Requesting refinement", 'INFO', 'CACHE')

                # Improve translation
                final = self._translate_chunk_stage2(
                    chunk, draft, source_lang, target_lang, genre, custom_instructions
                )

                # Cache successful translation
                if not final.startswith('[TRANSLATION_FAILED'):
                    debug_print(f"[CACHE SAVE S2] Storing refinement ({len(final)} chars)", 'DEBUG', 'CACHE')
                    self.cache.set(
                        chunk, final, draft,
                        source_lang, target_lang,
                        f"{self.model_name}_stage2", context_hash
                    )

                final_translations.append(final)

            progress_pct = ((chunk_num + total_chunks) / (total_chunks * 2)) * 100
            debug_print(f"[PROGRESS] {progress_pct:.1f}% complete", 'INFO', 'TRANS')

            yield TranslationProgress(
                progress=progress_pct,
                stage='reflection_improvement',
                original_text='\n\n'.join(chunks),
                machine_translation='\n\n'.join(draft_translations),
                translated_text='\n\n'.join(final_translations),
                current_chunk=chunk_num + total_chunks,
                total_chunks=total_chunks * 2
            )

            if config.translation.chunk_delay > 0:
                time.sleep(config.translation.chunk_delay)
        
        # Final result
        self.logger.info(f"Translation complete: {total_chunks} chunks processed")

        final_text = '\n\n'.join(final_translations)
        debug_print(f"", 'INFO', 'TRANS')
        debug_print(f"{'='*60}", 'INFO', 'TRANS')
        debug_print(f"[TRANSLATION COMPLETE]", 'INFO', 'TRANS')
        debug_print(f"  Chunks processed: {total_chunks}", 'INFO', 'TRANS')
        debug_print(f"  Original length: {len(text)} chars", 'INFO', 'TRANS')
        debug_print(f"  Final length: {len(final_text)} chars", 'INFO', 'TRANS')
        debug_print(f"{'='*60}", 'INFO', 'TRANS')

        yield TranslationProgress(
            progress=100,
            stage='completed',
            original_text='\n\n'.join(chunks),
            machine_translation='\n\n'.join(draft_translations),
            translated_text=final_text,
            current_chunk=total_chunks * 2,
            total_chunks=total_chunks * 2
        )
