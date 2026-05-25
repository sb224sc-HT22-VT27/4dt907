from app.services import session_analysis_service


def _make_frame():
    return [
        {"name": name, "x": 0.1, "y": 0.2, "z": 0.3}
        for name in session_analysis_service._MODEL_JOINT_NAMES
    ]


def test_analyze_session_sets_goodbad_and_squat_score(monkeypatch):
    frames = [_make_frame(), _make_frame(), _make_frame()]

    monkeypatch.setattr(
        session_analysis_service.start_stop_model_service,
        "predict_batch",
        lambda *_args, **_kwargs: [1, 1, 0],
    )
    monkeypatch.setattr(
        session_analysis_service,
        "_predict_z_all_frames",
        lambda _frames: [{"nose": 0.3}, {"nose": 0.3}, {"nose": 0.3}],
    )
    monkeypatch.setattr(
        session_analysis_service.goodbad_model_service,
        "predict_session",
        lambda *_args, **_kwargs: 0.75,
    )
    monkeypatch.setattr(
        session_analysis_service.scoring_model_service,
        "predict_session",
        lambda *_args, **_kwargs: 1.5,
    )

    results, timings = session_analysis_service.analyze_session(frames, norm_frames=frames)

    assert results[0].good_bad_score == 0.75
    assert results[1].good_bad_score == 0.75
    assert results[0].squat_score == 1.5
    assert results[1].squat_score == 1.5
    assert results[2].good_bad_score is None
    assert results[2].squat_score is None
    assert "goodbad_ms" in timings
    assert "scoring_ms" in timings
