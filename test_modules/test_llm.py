"""
Test suite for llm_module.py

Run: pytest test_modules/test_llm.py -v
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests

from modules.llm_module import (
    LLMModule,
    LLMModuleError,
    MemoryManager,
    OllamaConnection,
    SessionManager,
)


class TestOllamaConnection(unittest.TestCase):
    def test_is_available_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch('modules.llm_module.requests.get', return_value=mock_resp):
            conn = OllamaConnection(host='http://localhost:11434', model='m')
            self.assertTrue(conn.is_available())

    def test_is_available_failure(self):
        with patch('modules.llm_module.requests.get', side_effect=Exception('down')):
            conn = OllamaConnection(host='http://localhost:11434', model='m')
            self.assertFalse(conn.is_available())

    def test_chat_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'message': {'content': '  hello  '}}
        with patch('modules.llm_module.requests.post', return_value=mock_resp):
            conn = OllamaConnection(host='http://h', model='m', chat_timeout_seconds=5)
            out = conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertEqual(out, 'hello')

    def test_chat_bad_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = 'err'
        with patch('modules.llm_module.requests.post', return_value=mock_resp):
            conn = OllamaConnection(host='http://h', model='m', chat_timeout_seconds=5)
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('500', str(ctx.exception))

    def test_chat_timeout(self):
        with patch(
            'modules.llm_module.requests.post',
            side_effect=requests.Timeout(),
        ):
            conn = OllamaConnection(host='http://h', model='m', chat_timeout_seconds=5)
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('timeout', str(ctx.exception).lower())

    def test_chat_connection_error(self):
        with patch(
            'modules.llm_module.requests.post',
            side_effect=requests.ConnectionError(),
        ):
            conn = OllamaConnection(host='http://h', model='m', chat_timeout_seconds=5)
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('connect', str(ctx.exception).lower())


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.sm = SessionManager(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_session_invalid_name(self):
        with self.assertRaises(LLMModuleError):
            self.sm.create_session('')
        with self.assertRaises(LLMModuleError):
            self.sm.create_session(123)  # type: ignore[arg-type]

    def test_create_session_invalid_language(self):
        with self.assertRaises(LLMModuleError):
            self.sm.create_session('Ali', language='fr')

    def test_create_session_and_messages(self):
        sid = self.sm.create_session('Ali', language='en')
        self.assertIn('Ali_', sid)
        mid = self.sm.add_message(sid, 'user', 'Hello')
        self.assertIsInstance(mid, int)
        self.assertEqual(self.sm.get_message_count(sid), 1)
        window = self.sm.get_sliding_window(sid, window_size=10)
        self.assertEqual(len(window), 1)
        self.assertEqual(window[0]['role'], 'user')
        self.assertEqual(window[0]['content'], 'Hello')

    def test_add_message_invalid_role(self):
        sid = self.sm.create_session('Bob', language='ar')
        with self.assertRaises(LLMModuleError):
            self.sm.add_message(sid, 'system', 'x')

    def test_add_message_invalid_content(self):
        sid = self.sm.create_session('Bob', language='ar')
        with self.assertRaises(LLMModuleError):
            self.sm.add_message(sid, 'user', '')

    def test_get_sliding_window_contains_messages(self):
        """Sliding window returns both roles; SQLite ties on second-resolution timestamps."""
        sid = self.sm.create_session('Carol', language='en')
        self.sm.add_message(sid, 'user', 'a')
        self.sm.add_message(sid, 'assistant', 'b')
        window = self.sm.get_sliding_window(sid, window_size=10)
        self.assertEqual(len(window), 2)
        pairs = {(m['role'], m['content']) for m in window}
        self.assertEqual(pairs, {('user', 'a'), ('assistant', 'b')})


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.sm = SessionManager(db_path=self.db_path)
        self.mock_ollama = MagicMock()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_should_summarize_false_when_below_window(self):
        from config.settings import get_settings

        settings = get_settings().llm
        mm = MemoryManager(self.sm, self.mock_ollama, llm_settings=settings, window_size=10)
        sid = self.sm.create_session('D', language='en')
        self.sm.add_message(sid, 'user', 'one')
        self.assertFalse(mm.should_summarize(sid))

    def test_summarize_conversation_empty_history(self):
        from config.settings import get_settings

        settings = get_settings().llm
        mm = MemoryManager(self.sm, self.mock_ollama, llm_settings=settings, window_size=10)
        sid = self.sm.create_session('E', language='en')
        self.assertIsNone(mm.summarize_conversation(sid))


class TestLLMModule(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.sm = SessionManager(db_path=self.db_path)
        self.mock_ollama = MagicMock()
        self.mock_ollama.is_available.return_value = True
        self.mock_ollama.chat.return_value = 'Assistant reply'

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_chat_invalid_user_message(self):
        from config.settings import get_settings

        settings = get_settings().llm
        llm = LLMModule(settings=settings, backend=self.mock_ollama, session_manager=self.sm)
        sid = self.sm.create_session('F', language='en')
        with self.assertRaises(LLMModuleError):
            llm.chat(sid, '')
        with self.assertRaises(LLMModuleError):
            llm.chat(sid, 42)  # type: ignore[arg-type]

    def test_chat_success_saves_messages(self):
        from config.settings import get_settings

        settings = get_settings().llm
        llm = LLMModule(settings=settings, backend=self.mock_ollama, session_manager=self.sm)
        sid = self.sm.create_session('G', language='en')
        out = llm.chat(sid, 'What is 2+2?')
        self.assertEqual(out, 'Assistant reply')
        self.mock_ollama.chat.assert_called()
        count = self.sm.get_message_count(sid)
        self.assertEqual(count, 2)

    def test_chat_empty_ollama_response_raises(self):
        from config.settings import get_settings

        settings = get_settings().llm
        self.mock_ollama.chat.return_value = None
        llm = LLMModule(settings=settings, backend=self.mock_ollama, session_manager=self.sm)
        sid = self.sm.create_session('H', language='en')
        with self.assertRaises(LLMModuleError) as ctx:
            llm.chat(sid, 'Hi')
        self.assertIn('Empty response', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
