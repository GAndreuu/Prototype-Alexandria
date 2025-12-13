"""
Tests for Harvester
"""
import pytest
from unittest.mock import Mock, patch
from core.utils.harvester import ArxivHarvester

class TestArxivHarvester:
    @pytest.fixture
    def harvester(self):
        return ArxivHarvester()

    def test_init(self, harvester):
        assert harvester is not None

    def test_harvest(self, harvester):
        # Mock internal methods to avoid real API calls
        # harvest internally calls search_arxiv and download
        with patch.object(harvester, 'search_arxiv', return_value=[]):
            with patch.object(harvester, 'download_paper', return_value=None):
                # harvest(query, max_papers, ingest) - ingest=False to avoid topology import
                results = harvester.harvest("query", max_papers=1, ingest=False)
                assert isinstance(results, list)
