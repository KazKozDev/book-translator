"""The chunk-level translation cache.

Keyed on the chunk text, the language pair and a model string that the pipeline
loads with everything else that would change the answer — the stage, the
refinement version, the glossary fingerprint — so a run under different
instructions cannot replay drafts produced under the old ones.
"""

import hashlib
import sqlite3
from typing import Dict, Optional


class TranslationCache:
    # No default: where the cache file lives is the application's decision, and
    # translator.py passes CACHE_DB_PATH. A default here would be a second
    # answer to that question, and the wrong one would be silent.
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_cache_db()
    
    def _init_cache_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS translation_cache (
                    hash_key TEXT PRIMARY KEY,
                    source_lang TEXT,
                    target_lang TEXT,
                    original_text TEXT,
                    translated_text TEXT,
                    machine_translation TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP
                )
            ''')

    def _generate_hash(self, text: str, source_lang: str, target_lang: str, model: str = "") -> str:
        key = f"{text}:{source_lang}:{target_lang}:{model}".encode('utf-8')
        return hashlib.sha256(key).hexdigest()
    
    def get_cached_translation(self, text: str, source_lang: str, target_lang: str, model: str = "") -> Optional[Dict[str, str]]:
        hash_key = self._generate_hash(text, source_lang, target_lang, model)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute('''
                SELECT translated_text, machine_translation
                FROM translation_cache
                WHERE hash_key = ?
            ''', (hash_key,))
            
            result = cur.fetchone()
            if result:
                conn.execute('''
                    UPDATE translation_cache
                    SET last_used = CURRENT_TIMESTAMP
                    WHERE hash_key = ?
                ''', (hash_key,))
                return {
                    'translated_text': result[0],
                    'machine_translation': result[1]
                }
        
        return None
    
    def cache_translation(self, text: str, translated_text: str, machine_translation: str, 
                         source_lang: str, target_lang: str, model: str = ""):
        hash_key = self._generate_hash(text, source_lang, target_lang, model)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO translation_cache
                (hash_key, source_lang, target_lang, original_text, translated_text, 
                 machine_translation, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (hash_key, source_lang, target_lang, text, translated_text, machine_translation))
    
    def cleanup_old_entries(self, days: int = 30):
        # Parameterised rather than interpolated. `days` is ours today, but a
        # retention period is exactly the kind of value that later arrives from
        # a settings screen, and by then the f-string is invisible.
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM translation_cache WHERE last_used < datetime('now', ?)",
                (f'-{int(days)} days',),
            )
