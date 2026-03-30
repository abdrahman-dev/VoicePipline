"""
Test suite for asr_module.py

Run with:
    pytest tests/test_asr.py -v
or standalone:
    python tests/test_asr.py
"""

import os
import sys
import unittest
import logging
from unittest.mock import Mock, patch, MagicMock
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..modules.asr_module import (
    record_audio,
    transcribe,
    ASRModuleError,
    TARGET_SAMPLE_RATE,
)

logging.basicConfig(level=logging.INFO)


class TestRecordAudio(unittest.TestCase):
    """Test audio recording functionality."""

    def test_invalid_duration_negative(self):
        """Duration must be positive."""
        with self.assertRaises(ASRModuleError) as ctx:
            record_audio(duration=-5)
        self.assertIn("Invalid duration", str(ctx.exception))

    def test_invalid_duration_zero(self):
        """Duration must be positive."""
        with self.assertRaises(ASRModuleError) as ctx:
            record_audio(duration=0)
        self.assertIn("Invalid duration", str(ctx.exception))

    @patch('sounddevice.rec')
    @patch('sounddevice.wait')
    def test_record_audio_success(self, mock_wait, mock_rec):
        """Successful recording returns bytes."""
        # Mock sounddevice
        mock_audio = np.array([[100, 200, 300, 400]], dtype='int16').T
        mock_rec.return_value = mock_audio
        mock_wait.return_value = None

        result = record_audio(duration=1, samplerate=16000)

        # Check result is bytes
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        
        # Verify sounddevice was called correctly
        mock_rec.assert_called_once()
        mock_wait.assert_called_once()

    @patch('sounddevice.rec')
    def test_record_audio_microphone_error(self, mock_rec):
        """Handle microphone/device errors gracefully."""
        mock_rec.side_effect = RuntimeError("Microphone not found")

        with self.assertRaises(ASRModuleError) as ctx:
            record_audio(duration=1)
        self.assertIn("Recording failed", str(ctx.exception))


class TestTranscribe(unittest.TestCase):
    """Test audio transcription functionality."""

    def test_transcribe_empty_audio(self):
        """Empty audio should raise error."""
        with self.assertRaises(ASRModuleError) as ctx:
            transcribe(b'')
        self.assertIn("No audio data", str(ctx.exception))

    def test_transcribe_none_audio(self):
        """None audio should raise error."""
        with self.assertRaises(ASRModuleError) as ctx:
            transcribe(None)
        self.assertIn("No audio data", str(ctx.exception))

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_transcribe_arabic_success(self, mock_recognize):
        """Successfully transcribe Arabic."""
        mock_recognize.return_value = "السلام عليكم"
        
        audio_bytes = b'\x00\x01\x02\x03' * 1000  # Dummy audio
        text, lang = transcribe(audio_bytes, language="ar")
        
        self.assertEqual(text, "السلام عليكم")
        self.assertEqual(lang, "ar")

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_transcribe_english_success(self, mock_recognize):
        """Successfully transcribe English."""
        mock_recognize.return_value = "Hello world"
        
        audio_bytes = b'\x00\x01\x02\x03' * 1000
        text, lang = transcribe(audio_bytes, language="en")
        
        self.assertEqual(text, "Hello world")
        self.assertEqual(lang, "en")

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_transcribe_auto_detect_arabic_first(self, mock_recognize):
        """Auto-detect tries Arabic first, then English."""
        # First call (Arabic) succeeds
        mock_recognize.return_value = "مرحبا"
        
        audio_bytes = b'\x00\x01\x02\x03' * 1000
        text, lang = transcribe(audio_bytes, language="auto")
        
        self.assertEqual(text, "مرحبا")
        self.assertEqual(lang, "ar")

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_transcribe_auto_detect_fallback_to_english(self, mock_recognize):
        """Auto-detect fallbacks to English if Arabic fails."""
        import speech_recognition as sr
        
        # First call (Arabic) fails, second (English) succeeds
        mock_recognize.side_effect = [
            sr.UnknownValueError(),  # Arabic fails
            "Hello"                   # English succeeds
        ]
        
        audio_bytes = b'\x00\x01\x02\x03' * 1000
        text, lang = transcribe(audio_bytes, language="auto")
        
        self.assertEqual(text, "Hello")
        self.assertEqual(lang, "en")

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_transcribe_no_speech_detected(self, mock_recognize):
        """Handle case where no speech is detected in any language."""
        import speech_recognition as sr
        
        mock_recognize.side_effect = sr.UnknownValueError()
        
        audio_bytes = b'\x00\x01\x02\x03' * 1000
        text, lang = transcribe(audio_bytes, language="auto")
        
        self.assertIsNone(text)
        self.assertIsNone(lang)

    @patch('speech_recognition.Recognizer.recognize_google')
    def test_transcribe_network_error(self, mock_recognize):
        """Handle network errors gracefully."""
        import speech_recognition as sr
        
        mock_recognize.side_effect = sr.RequestError("No internet")
        
        audio_bytes = b'\x00\x01\x02\x03' * 1000
        
        with self.assertRaises(ASRModuleError) as ctx:
            transcribe(audio_bytes)
        self.assertIn("Google Speech API", str(ctx.exception))

    def test_transcribe_unsupported_language(self):
        """Reject unsupported languages."""
        audio_bytes = b'\x00\x01\x02\x03' * 1000
        
        with self.assertRaises(ASRModuleError) as ctx:
            transcribe(audio_bytes, language="fr")
        self.assertIn("Unsupported language", str(ctx.exception))


class TestIntegration(unittest.TestCase):
    """Integration tests (requires actual hardware/network)."""

    @patch('sounddevice.rec')
    @patch('sounddevice.wait')
    @patch('speech_recognition.Recognizer.recognize_google')
    def test_full_pipeline(self, mock_recognize, mock_wait, mock_rec):
        """Full pipeline: record → transcribe."""
        # Setup mocks
        mock_audio = np.array([[100, 200, 300, 400]], dtype='int16').T
        mock_rec.return_value = mock_audio
        mock_wait.return_value = None
        mock_recognize.return_value = "مرحبا"
        
        # Record
        audio = record_audio(duration=1)
        self.assertIsInstance(audio, bytes)
        
        # Transcribe
        text, lang = transcribe(audio)
        self.assertEqual(text, "مرحبا")
        self.assertEqual(lang, "ar")


# Manual testing (for debugging)
def manual_test():
    """Run manual test with real microphone (no mocks)."""
    print("\n=== Manual ASR Test ===")
    print("Recording for 5 seconds...")
    
    try:
        audio = record_audio(duration=5)
        print(f"✓ Recorded {len(audio)} bytes")
        
        text, lang = transcribe(audio)
        if text:
            print(f"✓ Transcribed ({lang}): {text}")
        else:
            print("✗ No speech detected")
    
    except ASRModuleError as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # python test_asr.py --manual
        manual_test()
    else:
        # python test_asr.py
        # or pytest test_asr.py
        unittest.main()