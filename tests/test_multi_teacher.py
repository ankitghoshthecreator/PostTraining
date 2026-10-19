"""tests/test_multi_teacher.py — Multi-Teacher Ensemble and Disagreement Router tests."""

from eval.multi_teacher import MultiTeacherEnsemble, MultiTeacherResult


def test_multi_teacher_consensus_calculation():
    ensemble = MultiTeacherEnsemble(disagreement_threshold=0.30)
    prompt = "Summarize the paragraph."
    response = "The telescope detected signals from an ancient galaxy dating back billions of years."
    ref = "Ancient galaxy signals detected by deep space telescope."

    res = ensemble.evaluate(prompt, response, stage=2, reference=ref)
    assert isinstance(res, MultiTeacherResult)
    assert 0.0 <= res.consensus_score <= 1.0
    assert 0.0 <= res.disagreement_delta <= 1.0
    assert len(res.teacher_scores) == 3


def test_multi_teacher_disagreement_detection(tmp_path):
    log_file = tmp_path / "disagreements.jsonl"
    ensemble = MultiTeacherEnsemble(disagreement_threshold=0.05, review_log_path=log_file)

    # Output with potential divergent ratings across Rubric A, B, and Local
    prompt = "Fix the punctuation: yes i do agree"
    response = "yes, I do agree"  # Missing terminal period might be scored differently
    res = ensemble.evaluate(prompt, response, stage=4, reference="Yes, I do agree.")

    # High sensitivity threshold triggers audit logging
    if res.disagreement_delta >= 0.05:
        assert res.requires_human_review is True
        assert log_file.exists()
