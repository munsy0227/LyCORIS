# MOSS LoKR+DoRA 학습 속도와 중복 코드 정리

사용자 요청: MOSS Image의 학습 속도를 최대한 높이고 Ponytail full 기준으로
불필요한 코드를 정리한다. 이전 RTX 5090의 3s/it → 12s/it 보고는 로그가 없어
재현하지 못했으며, 이번 최적화를 그 문제의 확정 원인/해결로 주장하지 않는다.

## 변경

- 기준: LyCORIS `deeb89e`, sd-scripts `c562bbf`.
- LyCORIS는 이 채팅에 연결된 `moss-training-speed` worktree에서 작업했다.
  기존 `/home/munsy0227/LyCORIS`의 10월 18일 브랜치는 보존했다.
- 한 대상에 어댑터 하나만 있으면 `ContextVar`/공유 가중치 dict가 필요 없다.
  이를 생략해 `ContextVar.set()`에서 끊기던 fullgraph 컴파일을 가능하게 했다.
- 중복된 rebuilt-weight forward를 한 helper로 합쳤다. 여러 어댑터를 겹친
  경우에는 기존 순서와 context 경로를 유지한다.
- 세 갈래에 중복되던 DoRA norm 계산을 하나로 합쳤다. norm 입력을 미리
  detach해 backward가 불필요한 full direction을 하나 더 저장하지 않게 했다.
- sd-scripts `AttentionParams`가 동일 길이 정보를 한 번 계산하도록 했다.
  매 attention 블록에서 반복되던 GPU `all`/`item` 동기화와 컴파일 중단을
  제거했다. 단일/동일 길이/서로 다른 길이 배치의 출력·gradient를 검증했다.
- 같은 attention 함수의 의미 없는 `q = q` 타입 주석 대입을 제거했다.
- checkpoint 키, FP32 DoRA magnitude, detached norm 수학, runtime multiplier,
  dtype 선택, merge/undo, 프리셋 범위, 사용자 TOML은 변경하지 않았다.
- 새 의존성이나 CUDA 커널은 추가하지 않았다.

## 성능

RTX 4070 12GB, PyTorch 2.13.0+cu130, Triton 3.7.1, CPU 스레드 1개.
실제 차원의 MOSS 블록 하나(dim=3072, FFN=8192, heads=24), 5440 토큰,
6개 full-matrix LoKR+DoRA/scalar(factor=4), BF16 base/입력과 FP32 adapter,
gradient checkpointing, 256개 텍스트 토큰의 mask, optimizer 없이 forward/backward.

| 조건 | 반복 중앙값 | 최대 allocated GPU 메모리 |
| --- | ---: | ---: |
| LyCORIS 단일 어댑터 수정 + 수정 전 sd-scripts, compile 켬 | 302.91ms | 3247.89MiB |
| LyCORIS + sd-scripts 수정, compile 켬 | 197.33ms | 1709.55MiB |

약 34.9% 시간 감소, 약 47.4% 메모리 감소다. mask 없는 별도 비교에서는
compile 수정 전 303.46ms → 수정 후 182.40ms였다. 초기 컴파일은 해당
캐시 없는 실행에서 15.23초가 걸렸다. compile을 끈 수정 후 실행은 302.11ms다.
위 값은 모델 전체·데이터 로딩·optimizer·저장 비용 및 RTX 5090 결과가 아니다.

42개 동일 구조의 adapted Linear를 각각 컴파일한 별도 CPU 검사에서는 backend
graph가 하나만 생성돼, 단일 어댑터 분기가 객체별 재컴파일을 강제하지 않음을 확인했다.
실제 MOSS 전체의 bucket별 재컴파일 수를 측정한 결과는 아니다.

## 검증과 적용

- LyCORIS focused LoKR, optimizer, compile, MOSS: 155 passed, 6 skipped,
  141 subtests passed. skip은 optional quantization/실제 모델 파일 관련이다.
- CUDA DoRA: 4 passed, 33 subtests passed. FP32/FP16/BF16, 양쪽 norm 축,
  Inductor+checkpoint gradient를 독립 기준식과 비교하고 checkpoint/정확한
  merge undo를 검증했다. PyTorch 2.14.1+cu130과 실제 학습 환경의
  2.13.0+cu130 양쪽에서 같은 4개 검사와 33개 subtest가 통과했다.
- 실제 sd-scripts checkout: 30 passed. attention fullgraph, 모델 기준 비교,
  LoKR+DoRA+checkpoint fullgraph, CUDA offload, 두 진입점의 축소 BF16
  학습·저장·재개를 포함한다. PyTorch 2.13.0+cu130을 사용했다.
- 수정한 LyCORIS Python 파일 Ruff와 Black, 두 저장소 diff whitespace 검사 통과.
- sd-scripts 기존 lint 전체를 정리하는 범위로 확장하지 않았다.
- 설치 패키지를 변경하지 않았다. 커밋/원격 푸시는 하지 않았다.

사용하려면 두 수정본을 같은 학습 환경에 적용하고 기존 TOML에 `compile=true`를
추가한다. 기본 compile mode를 유지한다. block swap은 Linear 컴파일을 끄므로
이 최적화 효과가 제한된다. 초기 튜닝 지연만 줄이려면
`LYCORIS_KERNEL_TUNE=0`을 사용한다. compile에 따른 부동소수점 반올림 차이는
있을 수 있으며 실제 전체 학습의 수렴/화질은 아직 검증하지 않았다.

재현 벤치마크, 원본 로그, JUnit과 검토용 패치는
`artifacts/moss-speed-20261025/`에 보관했다.

## 추가 검토에서 승인한 네 가지 개선

1. MOSS trainer의 Inductor에 `emulate_precision_casts=True`를 적용했다.
   BF16 cast를 없애는 컴파일 최적화가 eager와의 오차를 키우는 것을 확인했다.
   축소 블록의 최대 gradient 상대 norm 오차는 5.319% → 0.274%였고 출력의
   최대 절대 오차는 0이었다. 원래 elementwise tolerance를 완화하지 않았다.
   `torch.compile`은 mode와 options를 동시에 받지 않으므로 기존 mode preset을
   options에 합친다. 네 mode의 보존을 검사했다. 실제 trainer 경로로 compiled
   mask+checkpoint 블록의 출력·24개 adapter gradient, 5회 Prodigy 갱신,
   2회 갱신 시점에서 저장 후 재개한 결과를 비교했다.
2. 설치된 Prodigy 1.1.2를 상속해 public 옵션과 state_dict를 유지했다.
   통계는 tensor로 모아 한 번 CPU에 전송하되 Python float 합산 순서를
   원본대로 유지하고, 상태·파라미터 갱신에는 PyTorch foreach를 쓴다.
   분모 scratch는 64MiB 또는 큰 파라미터 하나로 제한한다. CPU, FSDP,
   복수 장치, 공유 storage, coupled decay, 다른 버전은 원래 step에 위임한다.
   FP32/FP64/FP16/BF16, 옵션 조합, lr=0·누락 gradient·혼합 dtype,
   closure/hook, 공유 storage, 버퍼 chunking, 양방향 state 재개를 검사했다.
   테스트한 파라미터와 optimizer 상태는 원본과 비트 단위로 같았다.
3. 일반 non-reentrant checkpointing은 입력 requires_grad를 강제하지 않는다.
   moss/moss-full의 전체 adapter gradient가 기존 경로와 정확히 같았다.
   swap hook과 Unsloth가 필요한 입력 gradient는 계속 유지한다.
4. MOSS 텍스트 캐시는 uint16 디스크 형식을 그대로 BF16 tensor로 읽고 묶는다.
   embedding의 CPU 보관·H2D 전송량이 FP32 경로의 절반이다. in-memory 캐시는
   잘린 크기만 소유하도록 clone해 패딩된 encoder batch를 붙잡지 않는다.
   65,536개 BF16 bit pattern을 그대로 보존하고 실제 dataset/학습/저장/재개를
   검사했다. 다른 모델의 NumPy 캐시 변환은 기존 동작을 유지한다.

### 최종 성능 측정

같은 RTX 4070, PyTorch 2.13.0+cu130에서 측정했다. 아래 블록의 입력은 기존
벤치와 같이 requires_grad=True다. 입력 gradient 생략 효과를 포함한 수치가 아니다.

| 구간 | 변경 전 중앙값 | 최종 중앙값 | 최대 allocated 메모리 전 → 후 |
| --- | ---: | ---: | ---: |
| 위의 5440-token 블록, mask·checkpoint·compile | 302.91ms | 193.41ms | 3247.89 → 1745.42MiB |
| Prodigy만, 1076 tensor / 319,058,525 FP32 요소 | 147.95ms | 113.06ms | 7366.84 → 7417.35MiB |

블록은 약 36.2% 시간 감소·46.3% allocated 메모리 감소, Prodigy는 약 23.6%
시간 감소다. Prodigy shape은 실제 MOSS 전체의 `moss-full`, factor=4,
full-matrix LoKR+DoRA/scalar에서 가져왔다. base 모델 없이 parameter·gradient와
optimizer state만 생성했다. Prodigy는 초기화 한 번 뒤 5회, 블록은 3회 워밍업 뒤
10회 측정했다. GPU 클럭·실행 순서 차이가 있어 작은 수치 차이는 일반화하지 않는다.
두 비율을 더해서 전체 학습의 개선율로 해석하면 안 된다.

### 최종 적용과 한계

추가 변경은 `/home/munsy0227/sd-scripts` 실제 checkout에 적용했다. 적용 직전
기존 tracked diff의 SHA-256을 비교하고 패치 적용을 검사했다. LyCORIS 코드는
연결된 worktree에 있으며 기존 설치의 import 경로를 자동으로 바꾸지 않았다.
기존 TOML·캐시 파일·설치 패키지·원격 저장소는 변경하지 않았다.

최종 actual checkout 테스트는 **203 passed, 8 skipped, 1 deselected**였다.
8개 skip은 최적화 대상이 아닌 CPU half reference 조합, deselect는 기존
`show_timesteps` 출력 검사다. artifacts의 `actual-final.xml`과
`actual-final.log`에 기록했다. 최종 새 Python 파일 세 개의 Ruff·Black,
수정한 Python의 AST 파싱과 두 저장소 diff whitespace 검사도 통과했다.
기존 dataset.py의 lint 항목은 변경 전후 같으며 일괄 정리하지 않았다.
앞의 155개 LoKR/compile/MOSS 검사 및 CUDA
4개/33개 subtest는 runtime 코드가 같은 앞선 검증 결과다.

원래 RTX 5090의 12s/it는 로그가 없어 원인을 확정하지 못했다. 실제 5B 모델의
장시간 학습·품질·수렴, Windows/MPS, 분산 FSDP 실행은 검증하지 않았다.
컴파일은 초기 지연과 bucket별 재컴파일이 있고, Prodigy의 첫 상태 초기화에는
원본의 scalar 검사 비용이 남는다. fallback 코드는 실제 다중 GPU에서 측정하지 않았다.

## 사용자 승인한 커밋·게시 범위

사용자가 이어서 두 저장소의 커밋과 푸시를 요청했다. LyCORIS는 새
`codex/moss-training-speed`, sd-scripts는 기존
`codex/sd-scripts-moss-dev-integration`에 일반 push한다. main, 기존 LyCORIS
통합 브랜치와 다른 worktree는 갱신하지 않는다. PR·merge·force push는 하지 않는다.
이번 코드·테스트·문서와 이 작업 기록만 커밋하며, 생성된 측정 로그·JUnit·패치는
로컬 artifacts에 보관한다. 게시 직전 코드가 검증한 패치와 동일함을 재확인했다.
푸시 후 원격 ref와 local HEAD를 대조하고 결과를 로컬 artifacts에 기록한다.
