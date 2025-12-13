"""
Tests for core/agents/oracle.py (NeuralOracle)
"""
import pytest
from unittest.mock import Mock, patch
from core.agents.oracle import NeuralOracle

class TestNeuralOracle:
    """Tests for NeuralOracle class."""
    
    @pytest.fixture
    def oracle(self):
        """Create oracle instance with mocked LLMs."""
        with patch('core.agents.oracle.LocalLLM') as MockLocal, \
             patch('core.agents.oracle.genai') as MockGenAI, \
             patch('os.getenv', return_value='fake_key'):
             
             instance = NeuralOracle(use_hybrid=True)
             instance.local_llm.synthesize_facts.return_value = "Factual Draft"
             # Mock Gemini model created inside init
             instance.gemini_model = Mock()
             instance.gemini_model.generate_content.return_value = Mock(text="Refined Answer")
             instance.is_gemini_available = True
             
             return instance

    def test_init(self, oracle):
        """Test initialization."""
        assert oracle is not None
        assert oracle.local_llm is not None

    def test_synthesize_hybrid(self, oracle):
        """Test hybrid synthesis flow."""
        result = oracle.synthesize("query", [], mode="hybrid")
        assert result == "Refined Answer"
        oracle.local_llm.synthesize_facts.assert_called()
        oracle.gemini_model.generate_content.assert_called()

    def test_synthesize_local(self, oracle):
        """Test local mode synthesis."""
        result = oracle.synthesize("query", [], mode="local")
        assert result == "Factual Draft"
        # Gemini should not be called in local mode
        # Note: In hybrid flow it calls gemini_refine which calls generate_content
        # In local mode, it calls synthesize_facts and returns
        oracle.gemini_model.generate_content.assert_not_called()

    def test_get_capabilities(self, oracle):
        """Test capabilities reporting."""
        caps = oracle.get_capabilities()
        assert caps['mode'] == 'hybrid'
        assert caps['is_gemini_available'] is True
