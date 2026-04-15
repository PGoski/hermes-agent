"""Tests for the Chatterbox TTS provider.

All tests use mocks — no real model loading, no network calls, no GPU
required.  Safe to run in CI environments without PyTorch or chatterbox-tts.
"""

import os
import struct
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import the provider module.  We import individual functions so that tests
# remain independent of the rest of the Hermes codebase.
# ---------------------------------------------------------------------------

from tools.chatterbox_tts_provider import (
    _check_chatterbox_available,
    _concatenate_wav_bytes,
    _resolve_device,
    _split_text_for_chatterbox,
    generate_chatterbox_tts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(n_samples: int = 2400, sample_rate: int = 24000) -> bytes:
    """Build a minimal valid WAV file (16-bit PCM mono)."""
    pcm = b"\x00\x00" * n_samples
    data_size = len(pcm)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16
    )
    riff_size = 4 + len(fmt_chunk) + 8 + data_size
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + fmt_chunk
        + b"data"
        + struct.pack("<I", data_size)
        + pcm
    )


# ===========================================================================
# Text chunking
# ===========================================================================


class TestTextChunking:
    """Tests for _split_text_for_chatterbox."""

    def test_short_text_single_chunk(self):
        chunks = _split_text_for_chatterbox("Hello world.", limit=240)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_splits_at_sentence_boundary(self):
        text = (
            "First sentence. Second sentence. "
            "Third sentence is a bit longer to push the limit."
        )
        chunks = _split_text_for_chatterbox(text, limit=40)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 40

    def test_single_long_sentence_splits_on_words(self):
        text = "A " * 200  # 400 chars, no sentence breaks
        chunks = _split_text_for_chatterbox(text.strip(), limit=100)
        assert all(len(c) <= 100 for c in chunks)
        reconstructed = " ".join(chunks)
        assert reconstructed.count("A") == text.strip().count("A")

    def test_empty_text_returns_single_empty(self):
        chunks = _split_text_for_chatterbox("", limit=240)
        assert chunks == [""]

    def test_whitespace_only_returns_single_empty(self):
        chunks = _split_text_for_chatterbox("   \n\t  ", limit=240)
        assert chunks == [""]

    def test_exact_limit_no_split(self):
        text = "X" * 240
        chunks = _split_text_for_chatterbox(text, limit=240)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_preserves_punctuation(self):
        text = "Hello! How are you? I'm fine."
        chunks = _split_text_for_chatterbox(text, limit=15)
        combined = " ".join(chunks)
        assert "!" in combined
        assert "?" in combined
        assert "." in combined

    def test_unicode_and_emoji(self):
        text = "Héllo wörld! 🔥 This has üñíçödé. And emoji too! 😏👀"
        chunks = _split_text_for_chatterbox(text, limit=30)
        assert all(len(c) <= 30 for c in chunks)
        combined = " ".join(chunks)
        assert "🔥" in combined
        assert "😏" in combined

    def test_oversized_word_split_at_limit(self):
        text = "X" * 500  # Single "word" exceeding limit
        chunks = _split_text_for_chatterbox(text, limit=240)
        assert all(len(c) <= 240 for c in chunks)
        assert "".join(chunks) == text


# ===========================================================================
# WAV concatenation
# ===========================================================================


class TestWavConcatenation:
    """Tests for _concatenate_wav_bytes."""

    def test_empty_list_returns_empty(self):
        result = _concatenate_wav_bytes([])
        assert result == b""

    def test_single_segment_passthrough(self):
        wav = _make_wav_bytes(100)
        result = _concatenate_wav_bytes([wav])
        assert result == wav

    def test_two_segments_larger_than_either(self):
        wav1 = _make_wav_bytes(100)
        wav2 = _make_wav_bytes(200)
        result = _concatenate_wav_bytes([wav1, wav2])
        assert result[:4] == b"RIFF"
        assert len(result) > len(wav1)
        assert len(result) > len(wav2)

    def test_concatenated_data_size_correct(self):
        wav1 = _make_wav_bytes(1000)
        wav2 = _make_wav_bytes(500)
        result = _concatenate_wav_bytes([wav1, wav2])
        assert result[:4] == b"RIFF"
        data_pos = result.index(b"data")
        data_size = struct.unpack_from("<I", result, data_pos + 4)[0]
        # (1000 + 500) samples * 2 bytes each = 3000
        assert data_size == 3000

    def test_tiny_segment_skipped(self):
        wav = _make_wav_bytes(100)
        result = _concatenate_wav_bytes([wav, b"tiny", wav])
        assert result[:4] == b"RIFF"


# ===========================================================================
# Device resolution
# ===========================================================================


class TestDeviceResolution:
    """Tests for _resolve_device."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_cuda_preferred(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            assert _resolve_device() == "cuda"

    def test_mps_when_no_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            assert _resolve_device() == "mps"

    def test_cpu_fallback(self):
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            assert _resolve_device() == "cpu"


# ===========================================================================
# Local mode (mocked model)
# ===========================================================================


class TestGenerateLocal:
    """Tests for the local generation path with a mocked model."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    @staticmethod
    def _make_mock_model():
        import torch

        model = mock.MagicMock()
        model.generate.return_value = torch.zeros(1, 24000)  # 1s silence
        return model

    def test_generates_wav_file(self, tmp_path):
        out = str(tmp_path / "test.wav")
        config = {"chatterbox": {"mode": "local", "model": "original"}}
        model = self._make_mock_model()

        with mock.patch(
            "tools.chatterbox_tts_provider._get_or_load_model",
            return_value=model,
        ), mock.patch(
            "tools.chatterbox_tts_provider._check_chatterbox_available",
            return_value=True,
        ):
            result = generate_chatterbox_tts("Hello", out, config)

        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_passes_ref_audio(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(_make_wav_bytes(100))
        out = str(tmp_path / "test.wav")
        config = {"chatterbox": {"mode": "local", "ref_audio": str(ref)}}
        model = self._make_mock_model()

        with mock.patch(
            "tools.chatterbox_tts_provider._get_or_load_model",
            return_value=model,
        ), mock.patch(
            "tools.chatterbox_tts_provider._check_chatterbox_available",
            return_value=True,
        ):
            generate_chatterbox_tts("Hello", out, config)

        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["audio_prompt_path"] == str(ref)

    def test_raises_when_not_installed(self, tmp_path):
        out = str(tmp_path / "test.wav")
        config = {"chatterbox": {"mode": "local"}}

        with mock.patch(
            "tools.chatterbox_tts_provider._check_chatterbox_available",
            return_value=False,
        ):
            with pytest.raises(ValueError, match="chatterbox-tts.*not installed"):
                generate_chatterbox_tts("Hello", out, config)

    def test_chunks_long_text(self, tmp_path):
        out = str(tmp_path / "test.wav")
        long_text = "This is a sentence. " * 30  # ~600 chars
        config = {"chatterbox": {"mode": "local"}}
        model = self._make_mock_model()

        with mock.patch(
            "tools.chatterbox_tts_provider._get_or_load_model",
            return_value=model,
        ), mock.patch(
            "tools.chatterbox_tts_provider._check_chatterbox_available",
            return_value=True,
        ):
            generate_chatterbox_tts(long_text, out, config)

        assert model.generate.call_count > 1

    def test_empty_text_raises(self, tmp_path):
        out = str(tmp_path / "test.wav")
        config = {"chatterbox": {"mode": "local"}}

        with pytest.raises(ValueError, match="Text is required"):
            generate_chatterbox_tts("", out, config)


# ===========================================================================
# Server mode (mocked HTTP)
# ===========================================================================


class TestGenerateServer:
    """Tests for the server generation path with mocked HTTP."""

    def test_openai_compatible_endpoint(self, tmp_path):
        out = str(tmp_path / "test.wav")
        config = {"chatterbox": {"mode": "server", "url": "http://localhost:7860"}}

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _make_wav_bytes(2400)
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("requests.post", return_value=mock_resp) as mock_post:
            result = generate_chatterbox_tts("Hello", out, config)

        assert os.path.exists(result)
        call_url = mock_post.call_args[0][0]
        assert "/v1/audio/speech" in call_url

    def test_server_unreachable_raises_runtime_error(self, tmp_path):
        import requests

        out = str(tmp_path / "test.wav")
        config = {"chatterbox": {"mode": "server", "url": "http://localhost:99999"}}

        with mock.patch(
            "requests.post", side_effect=requests.ConnectionError("refused")
        ):
            with pytest.raises(RuntimeError, match="unreachable"):
                generate_chatterbox_tts("Hello", out, config)

    def test_empty_text_raises(self, tmp_path):
        out = str(tmp_path / "test.wav")
        config = {"chatterbox": {"mode": "server"}}

        with pytest.raises(ValueError, match="Text is required"):
            generate_chatterbox_tts("   ", out, config)
