"""
Tests for Logger
"""
import pytest
from loguru import logger as loguru_logger
from core.utils.logger import setup_logger

class TestLogger:
    def test_setup_logger_runs(self):
        # setup_logger configures global loguru logger, returns None
        result = setup_logger()
        # It's expected to return None
        assert result is None
        
    def test_info_logging(self):
        # Use loguru logger directly (imported globally)
        loguru_logger.info("Test info")
        
    def test_warning_logging(self):
        loguru_logger.warning("Test warn")
