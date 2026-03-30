"""
Test suite for tts_module.py

Run: pytest test_modules/test_tts.py -v
"""

import sys
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.tts_module import TTSModule, TTSModuleError, detect_language


def _fake_tts_init_dependencies(self):
    """Avoid pygame / edge-tts / mixer in unit tests."""
    mock_pygame = MagicMock()
    mock_pygame.mixer.music.get_busy.return_value = False
    self._pygame = mock_pygame
    mock_edge = MagicMock()
    comm = MagicMock()
    comm.save = AsyncMock()
    mock_edge.Communicate.return_value = comm
    self._edge_tts = mock_edge
    self._pygame_ready = True


class TestDetectLanguage(unittest.TestCase):
    def test_english_only(self):
        self.assertEqual(detect_language('Hello world'), 'en')

    def test_arabic(self):
        self.assertEqual(detect_language('مرحبا'), 'ar')

    def test_mixed_prefers_arabic_if_arabic_char_present(self):
        self.assertEqual(detect_language('Hello مرحبا'), 'ar')


class TestTTSModuleInit(unittest.TestCase):
    @patch.object(TTSModule, '_init_dependencies', _fake_tts_init_dependencies)
    def test_unsupported_engine_raises(self):
        settings = SimpleNamespace(
            engine='other_engine',
            voice_map={'en': 'en-US-GuyNeural'},
            audio_temp_dir=tempfile.gettempdir(),
            audio_filename_template='tts_{turn_id}.mp3',
            pygame_poll_interval_seconds=0.05,
        )
        with self.assertRaises(TTSModuleError) as ctx:
            TTSModule(settings=settings)
        self.assertIn('Unsupported TTS engine', str(ctx.exception))

    @patch.object(TTSModule, '_init_dependencies', _fake_tts_init_dependencies)
    def test_get_voice_defaults(self):
        settings = SimpleNamespace(
            engine='edge_tts',
            voice_map={'en': 'en-US-GuyNeural', 'ar': 'ar-EG-SalmaNeural'},
            audio_temp_dir=tempfile.gettempdir(),
            audio_filename_template='tts_{turn_id}.mp3',
            pygame_poll_interval_seconds=0.05,
        )
        tts = TTSModule(settings=settings)
        self.assertEqual(tts._get_voice('en'), 'en-US-GuyNeural')
        self.assertEqual(tts._get_voice('ar'), 'ar-EG-SalmaNeural')

    @patch.object(TTSModule, '_init_dependencies', _fake_tts_init_dependencies)
    def test_speak_empty_raises(self):
        settings = SimpleNamespace(
            engine='edge_tts',
            voice_map={'en': 'en-US-GuyNeural'},
            audio_temp_dir=tempfile.gettempdir(),
            audio_filename_template='tts_{turn_id}.mp3',
            pygame_poll_interval_seconds=0.05,
        )
        tts = TTSModule(settings=settings)
        with self.assertRaises(TTSModuleError):
            tts.speak('')
        with self.assertRaises(TTSModuleError):
            tts.speak('   ')

    @patch.object(TTSModule, '_init_dependencies', _fake_tts_init_dependencies)
    @patch.object(TTSModule, '_play_audio')
    @patch('modules.tts_module.asyncio.run')
    def test_speak_starts_worker(
        self,
        mock_asyncio_run,
        mock_play_audio,
    ):
        settings = SimpleNamespace(
            engine='edge_tts',
            voice_map={'en': 'en-US-GuyNeural'},
            audio_temp_dir=tempfile.gettempdir(),
            audio_filename_template='tts_{turn_id}.mp3',
            pygame_poll_interval_seconds=0.05,
        )
        tts = TTSModule(settings=settings)

        def run_worker(coro):
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_asyncio_run.side_effect = run_worker

        real_thread_class = __import__('threading').Thread

        def immediate_start(self):
            if self._target:
                self._target(*self._args, **self._kwargs)

        with patch('modules.tts_module.threading.Thread', autospec=True) as MockThread:
            MockThread.side_effect = real_thread_class
            with patch.object(real_thread_class, 'start', immediate_start):
                tts.speak('Hello')

        mock_asyncio_run.assert_called()
        mock_play_audio.assert_called_once()

    @patch.object(TTSModule, '_init_dependencies', _fake_tts_init_dependencies)
    def test_is_playing_false_when_mixer_not_busy(self):
        settings = SimpleNamespace(
            engine='edge_tts',
            voice_map={'en': 'en-US-GuyNeural'},
            audio_temp_dir=tempfile.gettempdir(),
            audio_filename_template='tts_{turn_id}.mp3',
            pygame_poll_interval_seconds=0.05,
        )
        tts = TTSModule(settings=settings)
        tts._pygame.mixer.music.get_busy.return_value = False
        self.assertFalse(tts.is_playing())

    @patch.object(TTSModule, '_init_dependencies', _fake_tts_init_dependencies)
    def test_stop_best_effort(self):
        settings = SimpleNamespace(
            engine='edge_tts',
            voice_map={'en': 'en-US-GuyNeural'},
            audio_temp_dir=tempfile.gettempdir(),
            audio_filename_template='tts_{turn_id}.mp3',
            pygame_poll_interval_seconds=0.05,
        )
        tts = TTSModule(settings=settings)
        tts.stop()


if __name__ == '__main__':
    unittest.main()
