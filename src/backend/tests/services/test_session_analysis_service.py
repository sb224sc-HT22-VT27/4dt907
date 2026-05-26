from app.services import session_analysis_service


def _make_frame(z_by_name):
    return [
        {"name": name, "x": 0.1, "y": 0.2, "z": z_by_name.get(name, 0.0)}
        for name in session_analysis_service._MODEL_JOINT_NAMES
    ]


def test_analyze_session_reuses_mediapipe_z(monkeypatch):
    monkeypatch.setattr(
        "app.services.session_analysis_service.start_stop_model_service.predict_batch",
        lambda _features, _variant="champion": [1],
    )
    monkeypatch.setattr(
        "app.services.session_analysis_service.goodbad_model_service.predict_session",
        lambda _frames, _variant="champion": 0.91,
    )
    monkeypatch.setattr(
        "app.services.session_analysis_service.scoring_model_service.predict_session",
        lambda _frames, _variant="champion": 2,
    )

    z_values = {
        name: float(i) / 10.0
        for i, name in enumerate(session_analysis_service._MODEL_JOINT_NAMES)
    }
    frames = [_make_frame(z_values)]

    results, timings = session_analysis_service.analyze_session(
        frames, norm_frames=frames
    )

    assert results[0].predicted_z == z_values
    assert results[0].good_bad_score == 0.91
    assert results[0].squat_score == 2
    assert "z_prediction_ms" in timings
    assert "scoring_ms" in timings


def test_analyze_session_fills_missing_joint_z_with_zero(monkeypatch):
    monkeypatch.setattr(
        "app.services.session_analysis_service.start_stop_model_service.predict_batch",
        lambda _features, _variant="champion": [0],
    )
    monkeypatch.setattr(
        "app.services.session_analysis_service.scoring_model_service.predict_session",
        lambda _frames, _variant="champion": 1,
    )
    monkeypatch.setattr(
        "app.services.session_analysis_service.goodbad_model_service.predict_session",
        lambda _frames, _variant="champion": None,
    )

    frames = [[{"name": "left_hip", "x": 0.1, "y": 0.2, "z": -0.4}]]

    results, _timings = session_analysis_service.analyze_session(frames)

    assert results[0].predicted_z["left_hip"] == -0.4
    assert results[0].predicted_z["right_hip"] == 0.0
    assert results[0].predicted_z["nose"] == 0.0
    assert results[0].squat_score == 1
