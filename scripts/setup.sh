#!/usr/bin/env bash
# 과정 시작 시 한 번만 실행한다.
#   bash scripts/setup.sh
#
# 필요한 라이브러리를 전부 설치하고, 설치 상태를 점검하고, 실습 데이터를 만든다.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "가상환경이 없다. 먼저 만든다:  python3 -m venv .venv"
    exit 1
fi

echo "==> 라이브러리 설치 (시간이 걸린다. PyTorch/TensorFlow 가 크다)"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo
echo "==> 설치 점검"
"$PYTHON" -m scripts.preflight

echo
echo "==> 실습 데이터 생성"
"$PYTHON" -m scripts.generate_sample_data

echo
echo "==> 전체 테스트"
"$PYTHON" -m pytest -q

echo
echo "준비 완료. docs/curriculum/1-1.md 부터 시작한다."
