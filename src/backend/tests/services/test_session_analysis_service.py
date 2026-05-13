from app.services import session_analysis_service


def _frame(seed: float):
    return [
        {"name": name, "x": seed + i, "y": seed + i + 0.1, "z": seed + i + 0.2}
        for i, name in enumerate(session_analysis_service._MODEL_JOINT_NAMES)
    ]


def test_build_features_uses_xy_only():
    frame = _frame(1.0)

    feats = session_analysis_service._build_features(frame)

    assert len(feats) == 26
    assert feats[:6] == [1.0, 1.1, 2.0, 2.1, 3.0, 3.1]


def test_analyze_session_cuts_start_stop_input_and_scores_only_exercise_segment(monkeypatch):
    captured = {}

    def _fake_predict_batch(features_batch, variant="champion"):
        captured["features_batch"] = features_batch
        return [0, 1, 1, 0]

    def _fake_goodbad_predict_session(exercise_frames, variant="champion"):
        captured["exercise_len"] = len(exercise_frames)
        return 0.88

    monkeypatch.setattr(
        session_analysis_service.start_stop_model_service,
        "predict_batch",
        _fake_predict_batch,
    )
    monkeypatch.setattr(
        session_analysis_service,
        "_smooth_start_stop",
        lambda preds, gap_threshold=10: preds,
    )
    monkeypatch.setattr(
        session_analysis_service,
        "_predict_z_all_joints",
        lambda _kp3d: {},
    )
    monkeypatch.setattr(
        session_analysis_service,
        "_classify_with_z",
        lambda _kp3d, _z: ("Deep", 80.0, 81.0, 0.9),
    )
    monkeypatch.setattr(
        session_analysis_service.goodbad_model_service,
        "predict_session",
        _fake_goodbad_predict_session,
    )

    frames = [_frame(1.0), _frame(2.0), _frame(3.0), _frame(4.0)]
    results = session_analysis_service.analyze_session(frames, norm_frames=frames)

    assert len(captured["features_batch"]) == 4
    assert all(len(features) == 26 for features in captured["features_batch"])
    assert captured["exercise_len"] == 2
    assert [r.good_bad_score for r in results] == [None, 0.88, 0.88, None]
