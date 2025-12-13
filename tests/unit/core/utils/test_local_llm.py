"""
Tests for Local LLM
"""
import pytest
from unittest.mock import Mock, patch
from core.utils.local_llm import LocalLLM

class TestLocalLLM:
    @pytest.fixture
    def llm(self):
        # Mock model loading to avoid errors
        with patch.object(LocalLLM, '_load_model'):
            llm = LocalLLM(model_name="test-model")
            return llm

    def test_init(self, llm):
        assert llm is not None

    def test_synthesize_facts(self, llm):
        # synthesize_facts(context, evidence)
        with patch.object(llm, '_generate', return_value="Fact."):
            res = llm.synthesize_facts("context", "evidence")
            assert isinstance(res, str)
        
    def test_fallback_synthesis(self, llm):
        with patch('google.generativeai.GenerativeModel') as mock_gemini:
            mock_gemini.return_value.generate_content.return_value.text = "Fallback fact."
            res = llm._fallback_synthesis("context", "evidence")
            assert isinstance(res, str)
