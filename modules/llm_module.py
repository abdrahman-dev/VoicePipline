"""LLM Module: LLaMA 3.2 Integration via Ollama with Memory Management.

Handles:
1. Connection to Ollama server
2. Chat with sliding window history (last 10 messages)
3. Automatic summarization every 10 messages
4. SQLite session/message storage
"""

import logging
import json
import sqlite3
from typing import Optional, Tuple, List
from datetime import datetime
import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)

_SETTINGS = get_settings()
_LLM = _SETTINGS.llm


class LLMModuleError(RuntimeError):
    """Raised when LLM inference or database operations fail."""


class OllamaConnection:
    """Handles Ollama server communication."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        chat_timeout_seconds: Optional[int] = None,
        availability_timeout_seconds: Optional[int] = None,
    ):
        self.host = host or _LLM.ollama_host
        self.model = model or _LLM.ollama_model
        self.chat_timeout_seconds = chat_timeout_seconds or _LLM.request_timeout_seconds
        self.availability_timeout_seconds = (
            availability_timeout_seconds or _LLM.ollama_availability_timeout_seconds
        )
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=self.availability_timeout_seconds,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"[LLM] Ollama not available: {e}")
            return False
    
    def chat(self, messages: List[dict], timeout: Optional[int] = None) -> Optional[str]:
        """Send chat request to Ollama.
        
        Args:
            messages: List of dicts with 'role' and 'content'
            timeout: Max seconds to wait for response
        
        Returns:
            Model response text or None if failed
        """
        timeout = timeout if timeout is not None else self.chat_timeout_seconds
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }
            
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code != 200:
                raise LLMModuleError(
                    f"[llm_module.chat] Ollama returned status {response.status_code}. "
                    f"Response: {response.text}"
                )
            
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        
        except requests.Timeout:
            raise LLMModuleError(
                "[llm_module.chat] Ollama response timeout. "
                f"Fix: increase timeout or check Ollama performance. "
                f"Timeout was {timeout}s."
            )
        except requests.ConnectionError:
            raise LLMModuleError(
                "[llm_module.chat] Cannot connect to Ollama. "
                "Fix: ensure Ollama is running (ollama serve). "
                f"Expected at {self.host}"
            )
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.chat] Ollama request failed. "
                f"Reason: {e}. "
                "Fix: check Ollama logs and network."
            ) from e


class SessionManager:
    """Manages student sessions and conversation history."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _LLM.db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("[LLM] Database initialized")
        
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module._init_database] Failed to initialize database. "
                f"Reason: {e}. "
                "Fix: check file permissions and disk space."
            ) from e
    
    def create_session(
        self,
        student_name: str,
        language: str = _SETTINGS.general.default_session_language,
    ) -> str:
        """Create a new session for a student.
        
        Args:
            student_name: Student's name
            language: "ar" for Arabic, "en" for English
        
        Returns:
            Session ID
        """
        if not student_name or not isinstance(student_name, str):
            raise LLMModuleError(
                "[llm_module.create_session] Invalid student_name. "
                "Fix: provide non-empty string."
            )
        
        if language not in ("ar", "en"):
            raise LLMModuleError(
                "[llm_module.create_session] Invalid language. "
                "Fix: use 'ar' or 'en'."
            )
        
        session_id = f"{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, student_name, language)
                VALUES (?, ?, ?)
            """, (session_id, student_name, language))
            conn.commit()
            conn.close()
            
            logger.info(f"[LLM] Session created: {session_id}")
            return session_id
        
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.create_session] Database error. "
                f"Reason: {e}."
            ) from e
    
    def add_message(self, session_id: str, role: str, content: str) -> int:
        """Add message to session history.
        
        Args:
            session_id: Session ID
            role: "user" or "assistant"
            content: Message text
        
        Returns:
            Message ID
        """
        if role not in ("user", "assistant"):
            raise LLMModuleError(
                "[llm_module.add_message] Invalid role. "
                "Fix: use 'user' or 'assistant'."
            )
        
        if not content or not isinstance(content, str):
            raise LLMModuleError(
                "[llm_module.add_message] Invalid content. "
                "Fix: provide non-empty string."
            )
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (session_id, role, content)
                VALUES (?, ?, ?)
            """, (session_id, role, content))
            conn.commit()
            message_id = cursor.lastrowid
            conn.close()
            
            logger.debug(f"[LLM] Message saved: {message_id} ({role})")
            return message_id
        
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.add_message] Failed to save message. "
                f"Reason: {e}."
            ) from e
    
    def get_sliding_window(self, session_id: str, window_size: int = 10) -> List[dict]:
        """Get last N messages (sliding window).
        
        Args:
            session_id: Session ID
            window_size: Number of messages to retrieve (default 10)
        
        Returns:
            List of message dicts with 'role' and 'content'
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, window_size))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Reverse to chronological order
            messages = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
            logger.debug(f"[LLM] Retrieved {len(messages)} messages (window)")
            return messages
        
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.get_sliding_window] Failed to retrieve messages. "
                f"Reason: {e}."
            ) from e
    
    def get_full_history(self, session_id: str) -> List[dict]:
        """Get all messages in session (for summarization).
        
        Args:
            session_id: Session ID
        
        Returns:
            List of all message dicts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            messages = [{"role": row[0], "content": row[1]} for row in rows]
            return messages
        
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.get_full_history] Failed to retrieve history. "
                f"Reason: {e}."
            ) from e
    
    def get_message_count(self, session_id: str) -> int:
        """Get total message count in session."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,)
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.get_message_count] Failed to count messages. "
                f"Reason: {e}."
            ) from e
    
    def save_summary(self, session_id: str, summary_text: str, message_count: int):
        """Save conversation summary.
        
        Args:
            session_id: Session ID
            summary_text: Summary text
            message_count: Number of messages summarized
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO summaries (session_id, summary_text, message_count)
                VALUES (?, ?, ?)
            """, (session_id, summary_text, message_count))
            conn.commit()
            conn.close()
            logger.info(f"[LLM] Summary saved ({message_count} messages)")
        
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.save_summary] Failed to save summary. "
                f"Reason: {e}."
            ) from e
    
    def get_session_language(self, session_id: str) -> str:
        """Get session language."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT language FROM sessions WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else _SETTINGS.general.default_session_language
        except Exception:
            return _SETTINGS.general.default_session_language


class MemoryManager:
    """Manages conversation memory with sliding window and summarization."""
    
    def __init__(
        self,
        session_manager: SessionManager,
        ollama: OllamaConnection,
        llm_settings=None,
        window_size: Optional[int] = None,
    ):
        self.session_manager = session_manager
        self.ollama = ollama
        self._llm_settings = llm_settings or _LLM
        self.window_size = int(window_size if window_size is not None else self._llm_settings.sliding_window_size)
    
    def should_summarize(self, session_id: str) -> bool:
        """Check if conversation reached summarization threshold."""
        count = self.session_manager.get_message_count(session_id)
        return count >= self.window_size
    
    def summarize_conversation(self, session_id: str) -> Optional[str]:
        """Summarize entire conversation history.
        
        Args:
            session_id: Session ID
        
        Returns:
            Summary text or None if failed
        """
        history = self.session_manager.get_full_history(session_id)
        language = self.session_manager.get_session_language(session_id)
        
        if not history:
            return None
        
        # Build conversation text
        conv_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ])
        
        # Prepare summarization prompt
        if language == "ar":
            summary_prompt = f"""يرجى تلخيص المحادثة التعليمية التالية بشكل موجز وواضح:

{conv_text}

الملخص:"""
        else:
            summary_prompt = f"""Please summarize the following educational conversation concisely:

{conv_text}

Summary:"""
        
        try:
            logger.info("[LLM] Starting summarization...")
            summary = self.ollama.chat([
                {"role": "user", "content": summary_prompt}
            ], timeout=self._llm_settings.summarization_timeout_seconds)
            
            if summary:
                self.session_manager.save_summary(
                    session_id,
                    summary,
                    len(history)
                )
                logger.info("[LLM] Summarization completed")
            
            return summary
        
        except Exception as e:
            logger.error(f"[LLM] Summarization failed: {e}")
            return None


class LLMModule:
    """Main LLM module combining Ollama, Memory, and Session management."""
    
    def __init__(
        self,
        settings=_LLM,
        backend: Optional[OllamaConnection] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        # backend/session_manager are injectable for unit tests.
        self._llm_settings = settings

        if backend is None and settings.provider != "ollama":
            raise LLMModuleError(
                f"[llm_module] Unsupported LLM provider '{settings.provider}'. "
                "Fix: pass a custom backend or implement another provider."
            )

        self.ollama = backend or OllamaConnection(
            host=settings.ollama_host,
            model=settings.ollama_model,
            chat_timeout_seconds=settings.request_timeout_seconds,
            availability_timeout_seconds=settings.ollama_availability_timeout_seconds,
        )
        self.session_manager = session_manager or SessionManager(settings.db_path)
        self.memory_manager = MemoryManager(
            self.session_manager,
            self.ollama,
            llm_settings=settings,
            window_size=settings.sliding_window_size,
        )
    
    def is_ready(self) -> bool:
        """Check if Ollama server is available."""
        return self.ollama.is_available()
    
    def chat(self, session_id: str, user_message: str) -> Optional[str]:
        """Process user message and generate response.
        
        Args:
            session_id: Session ID
            user_message: User's input text
        
        Returns:
            Assistant's response or None if failed
        
        Raises:
            LLMModuleError: On critical failures
        """
        if not user_message or not isinstance(user_message, str):
            raise LLMModuleError(
                "[llm_module.chat] Invalid user_message. "
                "Fix: provide non-empty string."
            )
        
        try:
            # Get language and system prompt
            language = self.session_manager.get_session_language(session_id)
            system_prompt = (
                self._llm_settings.system_prompt_arabic
                if language == "ar"
                else self._llm_settings.system_prompt_english
            )
            
            # Save user message
            self.session_manager.add_message(session_id, "user", user_message)
            logger.info(f"[LLM] User message saved for {session_id}")
            
            # Get sliding window history
            history = self.session_manager.get_sliding_window(session_id, window_size=self.memory_manager.window_size)
            
            # Build messages for Ollama
            messages = [
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_message}
            ]
            
            # Get response from LLaMA
            logger.info("[LLM] Sending to Ollama...")
            response = self.ollama.chat(messages, timeout=self._llm_settings.request_timeout_seconds)
            
            if response:
                # Save assistant response
                self.session_manager.add_message(session_id, "assistant", response)
                logger.info("[LLM] Response saved")
                
                # Check if summarization needed
                if self.memory_manager.should_summarize(session_id):
                    logger.info("[LLM] Triggering summarization...")
                    self.memory_manager.summarize_conversation(session_id)
                
                return response
            else:
                raise LLMModuleError("[llm_module.chat] Empty response from Ollama")
        
        except LLMModuleError:
            raise
        except Exception as e:
            raise LLMModuleError(
                f"[llm_module.chat] Unexpected error. "
                f"Reason: {e}."
            ) from e


"""
This module intentionally contains no inline test runners.
Use `d:\\AI_Robot\\VoicePipline\\tests` for unit tests and integration tests.
"""