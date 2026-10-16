from src.cnn_model import classify_cnn_status


def test_cnn_status_safe_below_warning_threshold():
    assert classify_cnn_status(0.349) == "SAFE"


def test_cnn_status_warning_at_threshold():
    assert classify_cnn_status(0.35) == "WARNING"
    assert classify_cnn_status(0.649) == "WARNING"


def test_cnn_status_critical_at_threshold():
    assert classify_cnn_status(0.65) == "CRITICAL"


def test_missing_probability_is_unavailable():
    assert classify_cnn_status(None) == "UNAVAILABLE"
