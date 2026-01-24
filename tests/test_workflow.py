"""
Tests for MassFlow workflow module.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY
from MassFlow import workflow
from MassFlow.config import MassFlowConfig, InputConfig, ProcessingConfig
from pathlib import Path

@pytest.fixture
def mock_config():
    return MassFlowConfig(
        input=InputConfig(file_path=Path("test.mgf")),
        processing=ProcessingConfig(),
        output_directory=Path("out")
    )

def test_load_data_mgf(mock_config):
    """Test load_data handles mgf format correctly."""
    mock_config.input.format = "mgf"
    with patch("MassFlow.workflow.load_from_mgf") as mock_load:
        workflow.load_data(mock_config)
        mock_load.assert_called_once_with("test.mgf")

def test_load_data_msp(mock_config):
    """Test load_data handles msp format correctly."""
    mock_config.input.format = "msp"
    mock_config.input.file_path = Path("test.msp")
    with patch("MassFlow.workflow.load_from_msp") as mock_load:
        workflow.load_data(mock_config)
        mock_load.assert_called_once_with("test.msp")

def test_load_data_invalid_format(mock_config):
    """Test load_data raises error for unsupported format."""
    # We need to bypass pydantic validation for this test or use a mock object that quacks like config
    # Since config is typed, we can just assign an invalid string if we force it or mock it
    # But InputConfig limits literals. 
    # Let's mock the config object to return an invalid format string
    mock_conf = MagicMock()
    mock_conf.input.file_path = "test.txt"
    mock_conf.input.format.lower.return_value = "txt"
    
    with pytest.raises(ValueError, match="Unsupported format"):
        workflow.load_data(mock_conf)

@patch("MassFlow.workflow.get_engine")
@patch("MassFlow.workflow.get_session_factory")
@patch("MassFlow.workflow.load_data")
@patch("MassFlow.workflow.metadata_processing")
@patch("MassFlow.workflow.peak_processing")
def test_run_workflow(mock_peak, mock_meta, mock_load, mock_session_fac, mock_engine, mock_config):
    """Test the run_workflow execution flow."""
    # Setup mocks
    mock_spectrum = MagicMock()
    mock_load.return_value = [mock_spectrum]
    mock_meta.return_value = mock_spectrum
    mock_peak.return_value = mock_spectrum
    
    mock_session = MagicMock()
    mock_session_fac.return_value = MagicMock(return_value=mock_session)
    
    # Configure database in config to trigger DB logic
    from MassFlow.config import DatabaseConfig
    mock_config.database = DatabaseConfig(url="sqlite:///:memory:")
    
    workflow.run_workflow(mock_config)
    
    # Verify calls
    mock_engine.assert_called_once()
    mock_session_fac.assert_called_once()
    mock_load.assert_called_once_with(mock_config)
    mock_meta.assert_called_with(mock_spectrum)
    mock_peak.assert_called_with(mock_spectrum, min_intensity=0.0, normalize=True)
    
    # Verify DB interaction (Job creation and update)
    assert mock_session.add.called
    assert mock_session.commit.call_count >= 2 # Once for create, once for complete

@patch("MassFlow.workflow.load_data")
def test_run_workflow_error_handling(mock_load, mock_config):
    """Test that workflow handles exceptions gracefully."""
    mock_load.side_effect = Exception("Test Error")
    
    # With DB config
    from MassFlow.config import DatabaseConfig
    mock_config.database = DatabaseConfig(url="sqlite:///:memory:")
    
    with patch("MassFlow.workflow.get_engine"), \
         patch("MassFlow.workflow.get_session_factory") as mock_session_fac:
        
        mock_session = MagicMock()
        mock_session_fac.return_value = MagicMock(return_value=mock_session)
        
        with pytest.raises(Exception, match="Test Error"):
            workflow.run_workflow(mock_config)
            
        # Check that job status was updated to FAILED
        # We can't easily check the exact property set on the job object without more complex mocking,
        # but we can check that commit was called after the error (catch block)
        assert mock_session.commit.call_count >= 2
