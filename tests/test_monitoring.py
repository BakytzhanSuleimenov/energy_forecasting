import json

from src.pipelines.monitoring import main
from tests.test_predict_cli import prepare_artifacts


def test_monitoring_cli_persists_reports(tmp_path, monkeypatch, capsys):
    reference_df, artifacts_dir = prepare_artifacts(tmp_path)
    reference_csv = tmp_path / "reference.csv"
    current_csv = tmp_path / "current.csv"
    output_dir = tmp_path / "monitoring"

    reference_df.to_csv(reference_csv, index=False)
    current_df = reference_df.copy()
    current_df["temperature"] = current_df["temperature"] + 25
    current_df.to_csv(current_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "monitoring.py",
            "--artifacts-dir",
            str(artifacts_dir),
            "--reference-csv",
            str(reference_csv),
            "--current-csv",
            str(current_csv),
            "--output-dir",
            str(output_dir),
        ],
    )
    main()

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    drift_report = json.loads((output_dir / "feature_drift.json").read_text())
    error_report = json.loads((output_dir / "error_report.json").read_text())

    assert summary["model_name"] == "random_forest"
    assert "temperature" in summary["drifted_features"]
    assert isinstance(drift_report, list)
    assert error_report["reference"]["window_count"] > 0
    assert (output_dir / "summary.json").exists()
