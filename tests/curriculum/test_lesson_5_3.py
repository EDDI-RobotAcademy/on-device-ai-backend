"""실습 5-3 — AI의 모든 판단을 로그로 남겨라.

    pytest -m lesson_5_3 -s

"로그를 남기고 있습니다"는 답이 아니다.
**무엇을 남기고 있는가**가 답이다.

로그는 나중에 추가할 수 없다. 지나간 시간은 다시 안 온다.
"""

from __future__ import annotations

import pytest

from domain.operations.inference_log import (
    InferenceLogPolicy,
    InferenceRecord,
    LogCoverage,
    summarize,
)
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Severity
from tests.support import report

pytestmark = pytest.mark.lesson_5_3


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_한_판단이_남겨야_하는_여섯_가지(deployed) -> None:
    report.section("실습 5-3 · AI의 모든 판단을 로그로 남겨라")

    record = deployed.records[0]
    report.block(
        "추론 하나의 기록",
        "\n".join(
            [
                f"  언제       : {record.occurred_at}",
                f"  어느 디바이스 : {record.device_id}",
                f"  어느 버전  : v{record.deployment_version}",
                f"  무엇이라고 : {record.predicted_label}",
                f"  얼마나 확신 : {record.confidence:.3f}",
                f"  몇 ms      : {record.latency_ms:.4f}",
                f"  입력 지문  : {record.input_digest}",
                f"  정답       : {record.ground_truth or '(없음)'}",
            ]
        ),
    )
    report.note(
        "이 여섯 개가 없으면 실습 5-4 부터 5-11 까지 전부 못 한다. "
        "각각이 나중에 나올 질문 하나씩에 대응한다."
    )
    assert record.occurred_at and record.device_id and record.deployment_version


def test_현장에는_정답이_거의_없다(deployed) -> None:
    """**모듈 5 전체가 이 사실 위에 서 있다.**"""
    coverage = summarize(deployed.records)
    report.block("4일치 로그 요약", coverage.describe())

    assert coverage.total_count > 30_000
    assert coverage.distinct_devices == 3
    assert 0.05 < coverage.labeled_ratio < 0.25
    report.note(
        f"정답이 붙은 것은 {coverage.labeled_ratio:.1%} 다. "
        "나머지 88% 는 모델이 맞았는지 틀렸는지 **아무도 모른다.**"
    )
    report.note(
        "그래서 현장 정확도를 실시간으로 잴 수 없고, "
        "지연시간·예측분포·입력분포로 대신 본다 (실습 5-5, 5-6, 5-7)."
    )


def test_로그가_비어_있으면_아무것도_안_보고_있는_것이다() -> None:
    findings = InferenceLogPolicy().inspect(
        LogCoverage(
            total_count=0,
            with_timestamp=0,
            with_device=0,
            with_version=0,
            with_confidence=0,
            with_digest=0,
            labeled_count=0,
            distinct_devices=0,
            distinct_versions=0,
        )
    )
    assert "LOG_EMPTY" in codes(findings)
    report.note(findings[0].message)


def test_빠진_항목마다_못_하게_되는_일이_정해져_있다() -> None:
    partial = LogCoverage(
        total_count=1_000,
        with_timestamp=1_000,
        with_device=400,
        with_version=0,
        with_confidence=1_000,
        with_digest=0,
        labeled_count=0,
        distinct_devices=2,
        distinct_versions=1,
    )
    findings = InferenceLogPolicy().inspect(partial)
    report.block("반쯤 남긴 로그", "\n".join(f"  - {f.describe()}" for f in findings))

    found = codes(findings)
    assert "LOG_MISSING_DEVICE" in found
    assert "LOG_MISSING_VERSION" in found
    assert "LOG_MISSING_DIGEST" in found
    assert any(f.severity is Severity.CRITICAL for f in findings)
    report.note(
        "'디바이스가 없다' 는 '전부 이상한지 한 대만 이상한지 구분할 수 없다' 와 같은 말이다."
    )


def test_한_창에_두_버전이_섞이면_그_숫자는_아무의_것도_아니다() -> None:
    mixed = LogCoverage(
        total_count=1_000,
        with_timestamp=1_000,
        with_device=1_000,
        with_version=1_000,
        with_confidence=1_000,
        with_digest=1_000,
        labeled_count=120,
        distinct_devices=3,
        distinct_versions=2,
    )
    findings = InferenceLogPolicy().inspect(mixed)
    assert "LOG_MIXED_VERSIONS" in codes(findings)
    report.note(
        "배포 직후 구간이 대개 이렇게 된다. "
        "그 창의 p95 를 v2 의 성능이라고 부르면 틀린다."
    )


def test_남길_수_없는_것은_지문으로_남긴다(deployed) -> None:
    """원본 신호를 전부 남길 수는 없다."""
    digests = {r.input_digest for r in deployed.records[:1000]}
    assert len(digests) == 1000
    report.note(
        "입력 자체가 아니라 지문(sha1 16자)을 남긴다. "
        "이상한 예측을 발견했을 때 '그때 그 입력'을 되짚는 열쇠가 된다."
    )


def test_시각_없는_로그는_로그가_아니다() -> None:
    with pytest.raises(InvariantViolation) as caught:
        InferenceRecord(
            occurred_at="",
            device_id="DEV-01",
            deployment_version=1,
            predicted_label="NORMAL",
            confidence=0.9,
            latency_ms=0.03,
        )
    report.note(str(caught.value))

    with pytest.raises(InvariantViolation):
        InferenceRecord(
            occurred_at="2026-05-20 00:00:00",
            device_id="DEV-01",
            deployment_version=0,  # 어느 모델이 낸 답인지 모른다
            predicted_label="NORMAL",
            confidence=0.9,
            latency_ms=0.03,
        )


def test_정답이_붙은_것만_맞고_틀림을_말할_수_있다() -> None:
    unlabeled = InferenceRecord(
        occurred_at="2026-05-20 00:00:00",
        device_id="DEV-01",
        deployment_version=1,
        predicted_label="NORMAL",
        confidence=0.9,
        latency_ms=0.03,
    )
    assert unlabeled.is_correct is None
    report.note(
        "`is_correct` 가 False 가 아니라 **None** 이다. "
        "'틀렸다' 와 '모른다' 는 다른 것이고, 현장은 대부분 '모른다' 다."
    )

    from dataclasses import replace

    labeled = replace(unlabeled, ground_truth="FAULT")
    assert labeled.is_correct is False
