# 2026-10-08 sd-scripts 호환성 및 BF16 원인 확인

## 최종 범위

- 사용자 지정 대상은 `/home/munsy0227/tempsd/sd-scripts`이며, upstream main
  `690ea7f96c23182352ec63def76d431c6120bd2f`에서 검사했다.
- 사용자는 BF16 원인이 sd-scripts라면 수정하지 말라고 지시했다. 이에 따라
  임시 적용했던 `library/anima_models.py` 변경을 정확히 역적용했다.
  이후 대상 저장소의 `git status --short`와 `git diff --exit-code`가 깨끗했다.
- sd-scripts 패치에 의존하는 신규 LyCORIS 통합 테스트도 제거했다.
  두 저장소의 BF16 실행 코드는 최종적으로 변경하지 않았다.
- LyCORIS의 `requirements-kohya.txt`만 위 sd-scripts revision의 22개 직접
  의존성과 일치하도록 정리하고, VCS 설치 revision을 고정했다. 기존 sd-scripts
  환경에서는 해당 저장소의 requirements를 사용하고 LyCORIS를 별도 설치한다.
- 최초 검증 종료 시점에는 commit, push, PR 갱신 및 병합을 하지 않았다.
  이후 사용자가 커밋·푸시를 명시적으로 요청했다. 게시 범위는 LyCORIS
  `main`의 requirements 변경과 이 검증 기록 두 파일이며 대상 원격은
  `origin` (`munsy0227/LyCORIS`)이다. sd-scripts 수정과 PR 갱신/병합은 포함하지 않는다.
  기존 학습 가상환경과 `/home/munsy0227/sd-scripts`의 구형 fork 소스는 수정하지 않았다.

## BF16 오류의 근거

- Python 3.12.15 / torch 2.13.0+cu130 / Transformers 5.17.0 / RTX 4070에서
  실제 tempsd Anima 클래스의 축소 모델을 검사했다. 앞선 CPU 검사에서도
  같은 dtype 충돌이 발생했다.
- `use_adaln_lora=True`, FP32 timestep, BF16 기본 가중치, BF16 autocast에서
  **LyCORIS 어댑터 생성 전** 기본 모델이 다음 오류로 실패했다.

  ```text
  expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
  ```

- `library/anima_models.py:879`의 AdaLN 내부 context는 `use_fp32=False`일 때
  바깥 BF16 autocast를 끈다. 그 결과 882행의 Linear에 FP32 timestep embedding과
  BF16 가중치가 전달된다. 이 실패를 LyCORIS나 PR #6의 회귀로 볼 근거는 없다.
- 원본 sd-scripts의 FP32 축소 Anima에서는 448개 diffusion-only LoKR·DoRA,
  scale=1, 초기 no-op, 448개 유한한 비영 scalar gradient, SGD 이후 출력 변화,
  기본 가중치 보존 및 portable 복원을 확인했다. 최대 출력 절대 오차는 약 1.01e-6.
- sd-scripts를 임시 수정한 동안의 확대 BF16 실험은 최종 호환성 근거로 사용하지
  않는다. native 복원은 일치했지만 portable BF16 복원은 8개 조합 중 5개에서
  atol/rtol=0.03 기준을 넘었다(최대 절대 오차 약 0.0527). 해당 실험의 소스 수정과
  신규 테스트는 모두 제거했다. 전체 Anima BF16 학습이나 portable 수치 정확성을
  검증 완료했다고 주장하지 않는다.

## 최종 검증

- requirements의 모든 직접 의존성이 실제 tempsd requirements와 일치함을 비교했다.
  앞선 설치 없는 의존성 해석은 sd-scripts + LyCORIS core + torch/torchvision CPU
  조합을 65개 패키지로 해결했다. 선택적 옵티마이저 전체 실행 검증은 아니다.
- CPU: `python -m unittest -q test.lokr test.test_kohya_optimizer test.torch_compile`
  — 152개 중 150개 통과, 2개 선택적 테스트 skip. 모든 CPU 스레드는 1로 제한했다.
  최초 명령의 존재하지 않는 `test.test_compile_compat` 이름은 올바른
  `test.torch_compile`로 고쳐 위 명령을 재실행했다.
- RTX 4070: `python -m unittest -q test.test_lokr_cuda` — 기존 3개 테스트 통과.
  독립 DoRA 식/gradient/native 복원/정확한 undo의 24개 FP32·FP16·BF16 조합,
  zero-base 검사, 448개 어댑터의 모의 Anima BF16 optimizer step을 포함한다.
  수정하지 않은 실제 Anima의 BF16 baseline 실패를 해소했다는 뜻은 아니다.
- `git diff --check` 통과. 검사용 overlay, 모델, 소스 추출본 및 캐시는 정리했다.
  작은 실행 로그와 재현 probe는 `/tmp/lycoris-sd-compat-el25a0u0`에 보존했다.
- 실제 Anima checkpoint/데이터셋/VAE/전체 학습 루프, 장기 실행, Windows 및 MPS는
  이번 검사 범위 밖이다.
