import argparse
from unittest.mock import MagicMock, patch


from MassFlow.cli import run_db_inspect, run_db_merge


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
