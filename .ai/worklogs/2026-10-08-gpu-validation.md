# 2026-10-08 Linux GPU 검증

## 범위와 환경

- 사용자가 원본 동기화 후 GPU 검증을 요청했다. Windows는 사용하지 않는다고
  명시했다. 병합 커밋 `933834b`의 로컬 `main`에서 검사하고 재현한 커널 오류를
  수정했다. 검증 종료 시점에는 GPU 수정과 테스트가 로컬 미커밋 상태였으며
  원격 push는 하지 않았다. 이후 게시 승인은 아래에 기록한다.
- Linux / NVIDIA GeForce RTX 4070 12GB / driver 615.71.09 /
  CUDA toolkit 13.0.88 / Python 3.12.15 / torch 2.14.1+cu130 /
  Triton 3.8.0 / TileLang 0.1.15 / cuda-python 13.4.1.
- 기본 샌드박스에서는 NVIDIA 장치 접근이 차단되어 `nvidia-smi`와 CUDA 탐지가
  실패했다. GPU 접근이 가능한 실행에서 장치 진단과 실제 CUDA forward/backward가
  성공했다. 드라이버 문제로 단정했던 이전 기록을 정정했다. 사용자 측 준비 없이
  현재 장비로 검증할 수 있었다.
- 임시 환경 `.internal/gpu-validation`에서 실행했다. CPU/컴파일 스레드는 1로
  제한하고, CUDA 테스트를 순차 실행했다. 커널/컴파일/튜닝 캐시도 임시 환경 안으로
  지정했다. 본 검사에서는 튜닝을 껐고 별도 검사에서 자동 튜닝을 켰다.
- TileLang은 NVRTC로 실행했다. 설치된 CUDA toolkit과 호스트 GCC 조합으로
  C++ 경로를 검증하지 않았으며 NVRTC 결과를 다른 실행 백엔드에 일반화하지 않는다.

## 재현한 결함과 수정

1. **Triton LoRA merge backward**: 최초 커널 51개 중 4개가 컴파일 실패했다.
   두 역할 분기의 `acc`와 `gv`가 서로 다른 텐서 shape로 합쳐지는 SSA 문제였다.
   분기별 이름을 구분해 수식과 메모리 배치를 유지했다. 기존 실패 검사 4개를
   포함해 재검사가 성공했다.
2. **TileLang OFT NVRTC 호출**: 최초 커널 50개 중 OFT forward/backward와
   autograd 6개가 실패했다. Tensor 인수 `res`를 생성된 NVRTC 래퍼의 CUDA
   반환 코드 변수 `res`가 덮어썼다. 내부 prim_func 인수를 `rescale`로 변경했다.
   외부 함수 API와 위치 인수 순서는 동일하다.
3. **TileLang 작은 full-matrix LoKR backward**: 8x8 zero-base DoRA에서
   CUDA misaligned address가 발생했다. sticky CUDA 오류로 후속 검사가 연쇄
   실패했으므로 새 프로세스에서 2x2·4x4 factor로 격리했다. 기존 GradPack의
   네 view 포인터 정렬은 16바이트 기준 `[0, 0, 8, 8]`이었다. 별도 할당을
   사용한 대조군은 `[0, 0, 0, 0]`이고 PyTorch 기준 gradient와 일치했다.
   GradPack view 시작에 FP32 64개 단위 padding을 적용한 뒤 동일 재현과
   전체 회귀가 통과했다. homogeneous cast와 튜닝 scratch twin에도 동일한
   offset을 사용한다. 한 할당/한 cast를 유지하고 gradient 사이 추가 공간은
   최대 252바이트다. 새 작은 factor 회귀는 FP32/FP16/BF16 및 두 백엔드를 검사한다.

## 최종 검증 결과

| 검사 | 결과 |
| --- | --- |
| Triton kernel 수치/autograd/fallback | 54개 통과 |
| TileLang NVRTC kernel 수치/autograd | 53개 통과 |
| LoKR 선택 회귀, Triton | 567개 중 564개 통과, optional quantization 3개 skip |
| LoKR 선택 회귀, TileLang NVRTC | 567개 중 564개 통과, 동일한 3개 skip |
| 별도 CUDA DoRA·모의 Anima, Triton | 3개 메서드 통과 |
| 별도 CUDA DoRA·모의 Anima, TileLang NVRTC | 3개 메서드 통과 |
| 자동 튜닝을 켠 Triton LoKR | 5개 통과 |
| 자동 튜닝을 켠 TileLang NVRTC LoKR | 5개 통과 |
| Kohya optimizer/scope 및 enclosing compile CPU 호환성 | 17개 통과 |
| 변경 소스 Ruff, Black API 형식, Python compile, git diff --check | 통과 |

- LoKR 선택은 집중 검사 135개, module 384개, functional 16개, wrapper 32개다.
  CPU와 CUDA 장치/dtype 조합이 함께 포함된다. Skip은 설치하지 않은 선택적
  Quanto/bitsandbytes 런타임에 해당한다.
- 최종 Triton 통합 실행은 위 54+567+3 = **624개**, 실패 없이 3개 skip이었다.
  TileLang의 kernel/LoKR/별도 CUDA 선택도 각각 새 프로세스에서 성공했다.
- 별도 DoRA 검사는 Linear 및 groups=2 Conv1d/2d/3d, 두 norm 축,
  FP32/FP16/BF16의 **24개 nonzero 조합**을 독립적인 FP32 Kronecker/정규화 식과
  비교한다. Runtime multiplier 0/0.3/1, factor/scalar/magnitude gradient,
  native checkpoint의 bitwise 재구성, 병합 결과 및 정확한 undo를 검사했다.
  Zero-base 초기 no-op/zero gradient도 세 dtype에서 통과했다.
- 소형 mock Anima는 28개 Block의 **448개 어댑터**를 diffusion-only로 적용했다.
  BF16, network_dim=100000, factor=4, DoRA, scalar, regex dim 설정으로 한 AdaLN
  projection의 정확한 초기 no-op, finite nonzero scalar gradient, SGD 한 단계
  이후 출력 변화를 확인했다. 전체 실제 모델의 학습을 실행한 것은 아니다.
- 단기 소형 테스트의 PyTorch CUDA 최대 할당은 최종 Triton 통합 실행 약
  20.5MiB, TileLang LoKR 실행 약 18.6MiB였다. 실제 모델의 VRAM 요구량이나
  학습 처리량을 뜻하지 않는다.

## 재실행 방법과 한계

격리된 CUDA PyTorch 환경에서 다음 공통 설정을 사용한다. GPU 장치가 보이는
환경인지 먼저 확인해야 하며, CUDA 미탐지 상태의 skip 결과는 GPU 성공 근거가 아니다.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export TVM_NUM_THREADS=1 PYTHONPATH=.
export TILELANG_EXECUTION_BACKEND=nvrtc LYCORIS_KERNEL_TUNE=off
python -m unittest -v test.kernels.test_ops test.kernels.test_autograd
LYCORIS_KERNEL_BACKEND=triton python -m unittest -v test.lokr test.test_lokr_cuda
LYCORIS_KERNEL_BACKEND=tilelang python -m unittest -v test.lokr test.test_lokr_cuda
python -m unittest -q test.test_kohya_optimizer test.torch_compile
```

- 자동 튜닝 검사는 새 프로세스에서 `LYCORIS_KERNEL_TUNE=on`으로 작은
  full-matrix rebuild 세 dtype와 bypass FP16/FP32의 forward/backward를 검사했다.
  각 백엔드별 튜닝 결과 10개가 임시 캐시에 저장되었음을 확인했다.
  종료 시 `nvidia-smi`는 정상 응답했고 GPU 사용 메모리는 914MiB였다.
- 실제 Anima 체크포인트, 데이터셋, 훈련 실행 명령을 사용한 end-to-end 학습,
  장시간 실행/성능, MPS, 선택적 quantization, full Kohya/SDXL 및 Flux integration은
  미검증이다. 실제 학습까지 확인하려면 사용 중인 모델 경로와 훈련 설정/명령,
  소량의 학습 데이터가 필요하다.
- 사용자 요청 없이 commit/push/PR을 추가하지 않는다. 검증용 환경·캐시 약
  12GB는 결과 기록 후 정리했다. 새로운 테스트는 저장소에 보존했다.

## 후속 게시 승인

- 사용자가 검증 완료 후 커밋과 푸시를 명시적으로 요청했다.
- 대상은 포크 `munsy0227/LyCORIS`의 `origin/main`이며, GPU 수정·테스트·기록
  8개 파일과 앞서 완료한 원본 병합 커밋 `933834b`가 게시 범위에 포함된다.
- 게시 전 원격 `main`을 fetch했고 검증한 런타임/테스트 소스는 변경하지 않았다.
- CLI HTTPS 인증 정보가 없고 기존 SSH 키로도 GitHub 인증에 실패했다.
  연결된 GitHub 계정 `munsy0227`의 API 쓰기 권한을 확인해 게시 경로로 선택했다.
- 기존 로컬 GPU 커밋 `6b1dda96475b50190cd8261128574d861eab2614`와 원본
  병합 커밋은 `codex/pre-connected-publish-20261008`에 보존했다.
- GitHub API에서 원본 병합을 `e338170dda1cbe0d347666e9b2b0dc23b74c0676`으로
  재생성했다. Git tree가 원래 병합과 정확히 같고, 기존 포크/원본 두 부모의
  순서가 보존됐음을 확인했다. API의 커밋 메타데이터 생성으로 해시는 달라졌다.
- 전송한 수정 파일의 blob SHA를 모두 로컬 객체와 대조했다. 검증한 런타임과
  테스트는 동일하며, 인증 및 게시 기록만 이 문서에 추가했다.
