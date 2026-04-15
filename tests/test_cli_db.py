import argparse
from unittest.mock import MagicMock, patch

from MassFlow.cli import run_db_build, run_db_inspect, run_db_merge


@patch("MassFlow.database.SpectralDatabase")
def test_run_db_inspect(mock_db_class, capsys):
    mock_db = mock_db_class.return_value
    mock_db.get_total_spectra_count.return_value = 100
    mock_db.get_category_counts.return_value = {"ref": 50, "test": 50}
    mock_db.get_precursor_mz_range.return_value = (100.0, 200.0)

    args = argparse.Namespace(file="dummy.db")
    result = run_db_inspect(args)

    assert result == 0
    captured = capsys.readouterr()
    assert "Total Spectra: 100" in captured.out
    assert "100.0000 to 200.0000" in captured.out
    assert "- ref: 50 spectra" in captured.out


@patch("MassFlow.database.SpectralDatabase")
def test_run_db_inspect_error(mock_db_class, caplog):
    mock_db_class.side_effect = Exception("DB error")

    args = argparse.Namespace(file="dummy.db")
    result = run_db_inspect(args)

    assert result == 1
    assert "Database inspection failed: DB error" in caplog.text


@patch("MassFlow.database.SpectralDatabase")
def test_run_db_merge(mock_db_class):
    mock_in_db = MagicMock()
    mock_out_db = MagicMock()

    # Return out_db for the first call (output), then in_db for inputs
    mock_db_class.side_effect = [mock_out_db, mock_in_db, mock_in_db]

    mock_in_db.get_spectra.return_value = iter([MagicMock()])
    mock_out_db.add_spectra.return_value = 1

    args = argparse.Namespace(inputs=["in1.db", "in2.db"], output="out.db")
    result = run_db_merge(args)

    assert result == 0
    assert mock_out_db.add_spectra.call_count == 2
    assert mock_in_db.close.call_count == 2
    mock_out_db.close.assert_called_once()


@patch("MassFlow.database.SpectralDatabase")
def test_run_db_merge_empty(mock_db_class, caplog):
    mock_in_db = MagicMock()
    mock_out_db = MagicMock()

    mock_db_class.side_effect = [mock_out_db, mock_in_db]

    mock_out_db.add_spectra.return_value = 0

    args = argparse.Namespace(inputs=["in1.db"], output="out.db")
    result = run_db_merge(args)

    assert result == 1
    assert "No valid spectra were merged" in caplog.text


@patch("MassFlow.cli.MassFlowConfig.from_yaml")
@patch("MassFlow.database.SpectralDatabase")
@patch("MassFlow.io.load_spectra")
@patch("MassFlow.processing.process_spectra")
def test_run_db_build(mock_process, mock_load, mock_db_class, mock_config):
    mock_db = mock_db_class.return_value
    mock_db.add_spectra.return_value = 10

    mock_process.return_value = iter([MagicMock()])
    mock_load.return_value = iter([MagicMock()])

    args = argparse.Namespace(
        input="dummy.mgf", output="dummy.db", config="dummy.yaml", category="test"
    )
    result = run_db_build(args)

    assert result == 0
    mock_db.add_spectra.assert_called_once()


@patch("MassFlow.cli.MassFlowConfig.from_yaml")
@patch("MassFlow.database.SpectralDatabase")
@patch("MassFlow.io.load_spectra")
@patch("MassFlow.processing.process_spectra")
def test_run_db_build_empty(
    mock_process, mock_load, mock_db_class, mock_config, caplog
):
    mock_db = mock_db_class.return_value
    mock_db.add_spectra.return_value = 0

    args = argparse.Namespace(
        input="dummy.mgf", output="dummy.db", config="dummy.yaml", category="test"
    )
    result = run_db_build(args)

    assert result == 1
    assert "No valid spectra were extracted" in caplog.text
