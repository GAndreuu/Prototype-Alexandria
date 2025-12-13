"""
Tests for core/learning/profiles.py
"""
import pytest
from core.learning.profiles import ReasoningProfile, ALL_PROFILES, get_scout_profile, get_judge_profile, get_weaver_profile

class TestReasoningProfiles:
    """Tests for Reasoning Profiles."""
    
    def test_scout_profile(self):
        """Test the Scout profile properties."""
        profile = get_scout_profile()
        assert isinstance(profile, ReasoningProfile)
        assert profile.name == "The Scout"
        assert profile.novelty_bonus > 1.0  # High novelty bonus
        assert profile.risk_weight < 1.0    # Low risk aversion
    
    def test_judge_profile(self):
        """Test the Judge profile properties."""
        profile = get_judge_profile()
        assert profile.name == "The Judge"
        assert profile.risk_weight > 1.0    # High risk aversion
        assert profile.temperature < 0.5    # Low randomness
        
    def test_weaver_profile(self):
        """Test the Weaver profile properties."""
        profile = get_weaver_profile()
        assert profile.name == "The Weaver"
        assert profile.ambiguity_weight > 0
        
    def test_all_profiles_available(self):
        """Test all defined profiles map."""
        keys = ALL_PROFILES.keys()
        expected = ['scout', 'judge', 'weaver']
        for k in expected:
            assert k in keys
            factory = ALL_PROFILES[k]
            profile = factory()
            assert isinstance(profile, ReasoningProfile)
            
    def test_profile_structure(self):
        """Test profile dataclass fields presence."""
        p = get_scout_profile()
        fields = [
            'risk_weight', 'ambiguity_weight', 'novelty_bonus', 
            'planning_horizon', 'temperature', 'learning_rate_mod'
        ]
        for f in fields:
            assert hasattr(p, f)
