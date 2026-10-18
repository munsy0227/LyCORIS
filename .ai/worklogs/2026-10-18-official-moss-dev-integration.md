# 공식 sd-scripts MOSS 기반 통합 검증

- 승인한 계획에 따라 sd-scripts 작업 브랜치에 공식 dev `3e84e58f64` 전체를 적용하고 MOSS trainer를 공식 모델·로더·캐시로 재구성했다.
- 상세 구현, 캐시 전환, ai-toolkit 5개 변경의 반영 판단, 검증 명령과 제한은 [sd-scripts 작업 기록](/home/munsy0227/sd-scripts/.ai/worklogs/2026-10-18-official-moss-dev-integration.md)에 있다.
- LyCORIS에서는 핵심 수학·프리셋·portable 형식을 변경하지 않았다. 사용자 가이드에 공식 trainer 연결과 Tensor 반환 차이를 추가했다.
- `test/test_moss_sd_scripts.py`를 추가했다. 실제 공식 클래스에서 252/269개 범위와 두 factory × 두 preset × 3개 알고리즘 조합 × CPU FP32/CUDA BF16을 확인했다: 2 tests, 24 subtests passed.
- 기존 MOSS·Kohya optimizer 선택: 17 passed, 3 optional model cases skipped, 10 subtests passed.
- sd-scripts의 CPU 선택 156 passed/10 CUDA skipped, 별도 CUDA 선택 17 passed, sample→학습 복귀 CPU/CUDA 2 passed. 실제 ComfyUI 49개 검사는 CPU 선택에 포함한다. 선택 간 중복은 총수로 합산하지 않는다.
- 5B/Qwen9B 전체 학습·화질·성능·Windows/MPS는 미검증이다.
- LyCORIS main `be69d8a`, sd-scripts main `2de3c83`은 유지했다. sd-scripts는 머지 커밋 대기 상태이며 커밋·푸시는 수행하지 않았다.
- 기존 `artifacts/`와 autostop 작업 기록은 변경하지 않았다.
