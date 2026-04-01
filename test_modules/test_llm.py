"""
Test suite for llm_module.py with OpenRouter API

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
    openrouterConnection,
    SessionManager,
)


class TestopenrouterConnection(unittest.TestCase):
    """Test openrouter API connection (replaces local ollama server)"""
    
    def test_is_available_success(self):
        """Test successful API availability check"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch('modules.llm_module.requests.get', return_value=mock_resp):
            conn = openrouterConnection(api_key='sk-test-key', model='qwen/qwen-plus:free')
            self.assertTrue(conn.is_available())

    def test_is_available_failure(self):
        """Test API availability check failure"""
        with patch('modules.llm_module.requests.get', side_effect=Exception('down')):
            conn = openrouterConnection(api_key='sk-test-key', model='qwen/qwen-plus:free')
            self.assertFalse(conn.is_available())

    def test_missing_api_key(self):
        """Test that missing API key raises error immediately"""
        with self.assertRaises(LLMModuleError) as ctx:
            openrouterConnection(api_key=None, model='m')
        self.assertIn('Missing openrouter API key', str(ctx.exception))

    def test_chat_success_openai_format(self):
        """Test successful chat with OpenAI-compatible response format"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # OpenRouter returns OpenAI format: choices[0].message.content
        mock_resp.json.return_value = {
            'choices': [{'message': {'content': '  hello  '}}]
        }
        with patch('modules.llm_module.requests.post', return_value=mock_resp):
            conn = openrouterConnection(
                api_key='sk-test-key',
                model='qwen/qwen-plus:free',
                chat_timeout_seconds=5
            )
            out = conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertEqual(out, 'hello')

    def test_chat_bad_status_401_unauthorized(self):
        """Test error handling for 401 Unauthorized (invalid API key)"""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"error": "Invalid API key"}'
        with patch('modules.llm_module.requests.post', return_value=mock_resp):
            conn = openrouterConnection(
                api_key='invalid-key',
                model='qwen/qwen-plus:free',
                chat_timeout_seconds=5
            )
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('401', str(ctx.exception))

    def test_chat_bad_status_500_server_error(self):
        """Test error handling for 500 Server Error"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = 'Internal Server Error'
        with patch('modules.llm_module.requests.post', return_value=mock_resp):
            conn = openrouterConnection(
                api_key='sk-test-key',
                model='m',
                chat_timeout_seconds=5
            )
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('500', str(ctx.exception))

    def test_chat_timeout(self):
        """Test timeout error handling"""
        with patch(
            'modules.llm_module.requests.post',
            side_effect=requests.Timeout(),
        ):
            conn = openrouterConnection(
                api_key='sk-test-key',
                model='m',
                chat_timeout_seconds=5
            )
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('timeout', str(ctx.exception).lower())

    def test_chat_connection_error(self):
        """Test connection error handling (network issues)"""
        with patch(
            'modules.llm_module.requests.post',
            side_effect=requests.ConnectionError(),
        ):
            conn = openrouterConnection(
                api_key='sk-test-key',
                model='m',
                chat_timeout_seconds=5
            )
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('connect', str(ctx.exception).lower())

    def test_chat_empty_response(self):
        """Test error handling for empty content in API response"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'choices': [{'message': {'content': ''}}]}
        with patch('modules.llm_module.requests.post', return_value=mock_resp):
            conn = openrouterConnection(
                api_key='sk-test-key',
                model='m',
                chat_timeout_seconds=5
            )
            with self.assertRaises(LLMModuleError) as ctx:
                conn.chat([{'role': 'user', 'content': 'hi'}])
            self.assertIn('Empty response', str(ctx.exception))

    def test_chat_uses_bearer_token_auth(self):
        """Test that API key is properly sent in Authorization header"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'choices': [{'message': {'content': 'response'}}]
        }
        
        with patch('modules.llm_module.requests.post', return_value=mock_resp) as mock_post:
            conn = openrouterConnection(
                api_key='sk-test-123',
                model='qwen/qwen-plus:free',
                chat_timeout_seconds=5
            )
            conn.chat([{'role': 'user', 'content': 'test'}])
            
            # Verify the headers contain Bearer token
            call_kwargs = mock_post.call_args[1]
            self.assertIn('headers', call_kwargs)
            self.assertIn('Authorization', call_kwargs['headers'])
            self.assertEqual(
                call_kwargs['headers']['Authorization'],
                'Bearer sk-test-123'
            )


class TestSessionManager(unittest.TestCase):
    """Test session and message management"""
    
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.sm = SessionManager(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_session_invalid_name(self):
        """Test validation of student name"""
        with self.assertRaises(LLMModuleError):
            self.sm.create_session('')
        with self.assertRaises(LLMModuleError):
            self.sm.create_session(123)  # type: ignore[arg-type]

    def test_create_session_invalid_language(self):
        """Test validation of language code"""
        with self.assertRaises(LLMModuleError):
            self.sm.create_session('Ali', language='fr')

    def test_create_session_and_messages(self):
        """Test complete flow: create session, add messages, retrieve"""
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
        """Test validation of message role"""
        sid = self.sm.create_session('Bob', language='ar')
        with self.assertRaises(LLMModuleError):
            self.sm.add_message(sid, 'system', 'x')

    def test_add_message_invalid_content(self):
        """Test validation of message content"""
        sid = self.sm.create_session('Bob', language='ar')
        with self.assertRaises(LLMModuleError):
            self.sm.add_message(sid, 'user', '')

    def test_get_sliding_window_contains_messages(self):
        """Test sliding window returns correct message history"""
        sid = self.sm.create_session('Carol', language='en')
        self.sm.add_message(sid, 'user', 'a')
        self.sm.add_message(sid, 'assistant', 'b')
        window = self.sm.get_sliding_window(sid, window_size=10)
        self.assertEqual(len(window), 2)
        pairs = {(m['role'], m['content']) for m in window}
        self.assertEqual(pairs, {('user', 'a'), ('assistant', 'b')})

    def test_get_session_language(self):
        """Test retrieval of session language"""
        sid = self.sm.create_session('Dave', language='ar')
        lang = self.sm.get_session_language(sid)
        self.assertEqual(lang, 'ar')


class TestMemoryManager(unittest.TestCase):
    """Test memory management and summarization"""
    
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.sm = SessionManager(db_path=self.db_path)
        self.mock_openrouter = MagicMock()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_should_summarize_false_when_below_window(self):
        """Test that summarization doesn't trigger before threshold"""
        from config.settings import get_settings

        settings = get_settings().llm
        mm = MemoryManager(
            self.sm,
            self.mock_openrouter,
            llm_settings=settings,
            window_size=10
        )
        sid = self.sm.create_session('D', language='en')
        self.sm.add_message(sid, 'user', 'one')
        self.assertFalse(mm.should_summarize(sid))

    def test_should_summarize_true_at_threshold(self):
        """Test that summarization triggers at message threshold"""
        from config.settings import get_settings

        settings = get_settings().llm
        mm = MemoryManager(
            self.sm,
            self.mock_openrouter,
            llm_settings=settings,
            window_size=3
        )
        sid = self.sm.create_session('D2', language='en')
        self.sm.add_message(sid, 'user', 'one')
        self.sm.add_message(sid, 'assistant', 'two')
        self.sm.add_message(sid, 'user', 'three')
        self.assertTrue(mm.should_summarize(sid))

    def test_summarize_conversation_empty_history(self):
        """Test summarization with empty conversation"""
        from config.settings import get_settings

        settings = get_settings().llm
        mm = MemoryManager(
            self.sm,
            self.mock_openrouter,
            llm_settings=settings,
            window_size=10
        )
        sid = self.sm.create_session('E', language='en')
        self.assertIsNone(mm.summarize_conversation(sid))

    def test_summarize_conversation_success(self):
        """Test successful conversation summarization"""
        from config.settings import get_settings

        settings = get_settings().llm
        self.mock_openrouter.chat.return_value = 'This is a summary'
        
        mm = MemoryManager(
            self.sm,
            self.mock_openrouter,
            llm_settings=settings,
            window_size=10
        )
        sid = self.sm.create_session('E2', language='en')
        self.sm.add_message(sid, 'user', 'hello')
        self.sm.add_message(sid, 'assistant', 'hi there')
        
        result = mm.summarize_conversation(sid)
        self.assertEqual(result, 'This is a summary')


class TestLLMModule(unittest.TestCase):
    """Test main LLM module"""
    
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.sm = SessionManager(db_path=self.db_path)
        self.mock_openrouter = MagicMock()
        self.mock_openrouter.is_available.return_value = True
        self.mock_openrouter.chat.return_value = 'Assistant reply'

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_chat_invalid_user_message(self):
        """Test validation of user message"""
        from config.settings import get_settings

        settings = get_settings().llm
        llm = LLMModule(
            settings=settings,
            backend=self.mock_openrouter,
            session_manager=self.sm
        )
        sid = self.sm.create_session('F', language='en')
        
        with self.assertRaises(LLMModuleError):
            llm.chat(sid, '')
        
        with self.assertRaises(LLMModuleError):
            llm.chat(sid, 42)  # type: ignore[arg-type]

    def test_chat_success_saves_messages(self):
        """Test successful chat saves user and assistant messages"""
        from config.settings import get_settings

        settings = get_settings().llm
        llm = LLMModule(
            settings=settings,
            backend=self.mock_openrouter,
            session_manager=self.sm
        )
        sid = self.sm.create_session('G', language='en')
        out = llm.chat(sid, 'What is 2+2?')
        
        self.assertEqual(out, 'Assistant reply')
        self.mock_openrouter.chat.assert_called()
        count = self.sm.get_message_count(sid)
        self.assertEqual(count, 2)  # user + assistant

    def test_chat_empty_openrouter_response_raises(self):
        """Test error when API returns empty response"""
        from config.settings import get_settings

        settings = get_settings().llm
        self.mock_openrouter.chat.return_value = None
        
        llm = LLMModule(
            settings=settings,
            backend=self.mock_openrouter,
            session_manager=self.sm
        )
        sid = self.sm.create_session('H', language='en')
        
        with self.assertRaises(LLMModuleError) as ctx:
            llm.chat(sid, 'Hi')
        self.assertIn('Empty response', str(ctx.exception))

    def test_is_ready_checks_api_availability(self):
        """Test is_ready() checks API availability"""
        from config.settings import get_settings

        settings = get_settings().llm
        llm = LLMModule(
            settings=settings,
            backend=self.mock_openrouter,
            session_manager=self.sm
        )
        
        self.mock_openrouter.is_available.return_value = True
        self.assertTrue(llm.is_ready())
        
        self.mock_openrouter.is_available.return_value = False
        self.assertFalse(llm.is_ready())

    def test_chat_with_arabic_language(self):
        """Test chat with Arabic language session"""
        from config.settings import get_settings

        settings = get_settings().llm
        llm = LLMModule(
            settings=settings,
            backend=self.mock_openrouter,
            session_manager=self.sm
        )
        sid = self.sm.create_session('أحمد', language='ar')
        out = llm.chat(sid, 'مرحبا')
        
        self.assertEqual(out, 'Assistant reply')
        
        # Verify system prompt was Arabic
        call_args = self.mock_openrouter.chat.call_args[0][0]
        system_msg = call_args[0]
        self.assertEqual(system_msg['role'], 'system')
        self.assertIn('أنت', system_msg['content'])  # Check for Arabic


if __name__ == '__main__':
    unittest.main()