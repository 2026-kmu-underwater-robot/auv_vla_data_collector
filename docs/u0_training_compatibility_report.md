# U0 학습 데이터 호환성 수정 보고서

작성일: 2026-10-08  
수정 대상: `/home/pc/Downloads/auv_vla_data_collector`  
대조 기준: `/home/pc/Downloads/auv_vla_training`

## 1. 목적과 판단 기준

이번 작업의 목적은 AUV에서 수집한 데이터가 U0 학습 코드에 들어갈 때 필드의 의미, 차원, 타입, 시간 순서, 파일 위치가 일관되도록 만드는 것이다.

이를 위해 학습 프로젝트의 `KMUauvDataConfig`, 자체 데이터 로더, modality 스키마, 모델 입력 transform, demo 데이터, 의존성 목록을 수집 패키지와 대조했다. 판단 순서는 다음과 같다.

1. 학습 설정에서 실제로 요구하는 필드와 관측·action 시간 범위를 확인한다.
2. 로더가 저장 벡터를 어떤 인덱스로 잘라 읽는지 확인한다.
3. 모델 transform이 연결·정규화·padding하는 단계를 확인한다.
4. 수집·export 단계에서 일치하지 않는 선언과 경로를 수정한다.
5. 실제 MP4·Parquet 생성과 학습 설정 소스 대조 테스트로 확인한다.

핵심 판단은 **저장 데이터의 차원과 모델에 최종 전달하는 텐서의 차원을 구분해야 한다**는 것이다. 저장 파일에 모델의 padding까지 미리 적용하면 원본 센서 규격과 모델 입력 규격이 섞인다.

## 2. 학습 코드에서 확인한 입력 규격

### 2.1 사용할 학습 설정

학습 프로젝트의 `gr00t/experiment/data_config.py`에는 다음 설정이 등록되어 있다.

```text
kmu_auv → KMUauvDataConfig
```

기존 수집 패키지 README에는 `--data-config kmu26_auv_real`이 적혀 있었지만, 해당 이름은 학습 설정 레지스트리에 없다. 문서를 `--data-config kmu_auv`로 수정했다.

`info.json`의 `robot_type: kmu26_auv_real`은 로봇을 설명하는 메타데이터 값이다. 이것이 학습 CLI의 `data-config` 이름을 등록하는 것은 아니다.

### 2.2 State: 저장 23차원, 학습 선택 19차원

| 저장 필드 | 저장 슬라이스 | 차원 | 현재 학습 사용 |
| --- | --- | ---: | --- |
| 이전 조작 명령 `prev_command` | `[0:4]` | 4 | 사용 |
| DVL 속도 `dvl_velocity` | `[4:7]` | 3 | 사용 |
| 각속도 `angular_velocity` | `[7:10]` | 3 | 사용 |
| 선형가속도 `linear_acceleration` | `[10:13]` | 3 | 사용 |
| 자세 `attitude` | `[13:17]` | 4 | 사용 |
| 수심 `depth` | `[17:18]` | 1 | 사용 |
| 고도 `altitude` | `[18:19]` | 1 | 사용 |
| 유효성 표시 `validity` | `[19:23]` | 4 | 선택하지 않음 |

슬라이스는 시작 인덱스를 포함하고 끝 인덱스를 제외한다. 자세는 `[w, x, y, z]` quaternion이다. validity는 전방 카메라, 분리부 카메라, DVL 속도, 고도의 유효성을 나타낸다.

학습 설정은 저장 순서와 별도로 다음 순서로 필드를 연결한다.

```text
attitude → dvl_velocity → depth → prev_command
→ angular_velocity → linear_acceleration → altitude
```

따라서 실제 선택 차원은 `4 + 3 + 1 + 4 + 3 + 3 + 1 = 19`이다. 로더가 modality 이름과 슬라이스를 통해 읽으므로 저장 벡터의 순서를 학습 연결 순서로 재배열할 필요가 없다.

현재 관측 인덱스는 `[0]`이다. 학습 transform은 선택한 state를 정규화하고 연결한 뒤 64차원으로 zero padding한다. 배치 차원을 제외한 흐름은 다음과 같다.

```text
저장 state [23]
→ 필요한 modality 선택 [1, 19]
→ 정규화·연결
→ 모델 입력 [1, 64]
```

23차원 저장 규격은 유지했다. validity를 보존하면 원본 데이터 품질을 확인할 수 있다. 다만 현재 `kmu_auv` 모델은 이 네 값을 직접 입력받지 않는다. 모델이 이를 활용하도록 바꾸려면 학습 설정도 별도로 수정해야 한다.

### 2.3 Action: frame당 4차원, 학습 horizon 16

Action은 정규화된 `[surge, sway, heave, yaw]` 명령이며 각 값의 범위는 `[-1, 1]`이다. 저장 규격은 `float32[4]`이다.

학습 설정의 `action_indices = list(range(16))`는 현재 frame부터 16개 frame의 action을 선택한다는 뜻이다. 배치 차원을 제외한 흐름은 다음과 같다.

```text
frame별 저장 action [4]
→ 현재 frame부터 16개 선택 [16, 4]
→ 정규화·연결
→ 모델 입력 [16, 32]
```

즉, 4는 조작 축 개수, 16은 시간 길이, 32는 모델의 action 차원이다. 세 값을 동일하게 맞추는 것은 올바른 대응이 아니다. 저장 단계에서는 실제 조작 명령 4개를 기록하고 시간 묶기와 padding은 학습 transform에 맡긴다.

Episode 끝에서 미래 action이 부족하면 현재 로더는 absolute action에 대해 마지막 값을 반복해 padding한다. Export metadata에 `absolute: true`를 명시했다. 이는 좌표계가 월드 좌표라는 뜻이 아니라, action이 직전 값에 더하는 변화량이 아니라 해당 시점의 조작 명령이라는 의미이다.

### 2.4 카메라와 시간 규격

학습 설정은 `video.ego`, `video.buoy_release` 두 카메라를 사용한다. Export 영상 메타데이터의 shape는 `[H, W, 3]`이고 학습 transform이 crop과 224×224 resize를 수행한다. 따라서 수집 영상을 저장할 때 미리 224×224로 맞출 필요는 없다.

기본 수집 속도는 10 Hz이다. Export는 기록된 FPS를 사용한다. 기존의 서로 다른 FPS 및 샘플링 불연속 거부 정책을 유지했다. FPS 숫자만 바꾸면 센서와 action의 물리적인 시간 의미가 바뀌기 때문이다.

## 3. 실제 수정 내용과 논리적 이유

### 3.1 데이터 형식 버전 선언 정렬

**수정 전:** Export가 `codebase_version: v2.1`을 선언했다.  
**수정 후:** 학습 프로젝트 demo와 같은 `v2.0`을 선언하도록 `CODEBASE_VERSION` 상수를 추가했다.

학습 프로젝트는 `pyproject.toml`에서 Python `lerobot` 패키지를 직접 의존성으로 선언하지 않는다. `gr00t/data/dataset.py`가 Parquet, MP4, JSON/JSONL을 자체적으로 읽는다. 따라서 이번 작업에서는 설치된 LeRobot 패키지 버전보다 이 로더가 사용하는 파일 구조와 필드가 직접적인 호환성 기준이다.

이번 버전 변경은 이미 생성하는 구조와 제공된 demo의 선언을 정렬한 것이다. **기존 `v2.1` 문자열 때문에 로더가 실패했다는 실행 결과를 확인한 것은 아니다.** 또한 모든 외부 LeRobot 도구에 대한 호환성을 인증한 작업도 아니다.

기존 `requirements-export.txt`의 `pandas==2.2.3`, `pyarrow==14.0.1`, `numpy==1.26.4`를 확인했다. pandas와 pyarrow는 학습 프로젝트의 지정 버전과 일치하고 NumPy는 학습 프로젝트의 `>=1.23.5,<2.0.0` 범위에 들어간다. 이 의존성 파일은 수정하지 않았다.

### 3.2 State 슬라이스의 공통 정의

**수정 전:** Export의 modality 생성과 `policy_observation()`이 각자 state 구간을 정의했다.  
**수정 후:** `contract.py`의 `STATE_SLICES`를 두 곳에서 공통으로 참조한다.

두 정의가 현재 같더라도 이후 센서나 필드가 추가될 때 한쪽만 수정되면 학습과 추론이 서로 다른 의미의 값을 받을 수 있다. 예를 들어 `[13:17]`을 한쪽에서만 변경하면 같은 `state.attitude` 이름 아래 다른 값이 전달될 수 있다. 하나의 정의를 공유하면 이런 불일치의 발생 지점을 줄인다.

### 3.3 Modality 타입과 원본 필드 명시

**수정 전:** modality metadata에는 주로 `start`, `end`만 있었다. 학습 스키마의 dtype 기본값은 `float64`이다.  
**수정 후:** state와 action에 `dtype: float32`, 각각 `original_key: observation.state`와 `original_key: action`을 명시했다.

실제 저장 배열은 이미 float32였다. 이번 수정은 저장값과 메타데이터의 선언을 일치시킨 것이다. dtype 선언 누락이 반드시 실행 오류를 일으켰다고 단정할 수는 없지만, 로더가 기본값을 추정하도록 두는 것보다 실제 타입을 명시하는 것이 데이터 계약을 정확하게 만든다.

### 3.4 Chunk 경로 수정

**수정 전:** `chunks_size: 1000`을 선언하면서 모든 Parquet와 영상 파일을 `chunk-000`에 저장했다.  
**수정 후:** `episode_chunk = episode_index // 1000`을 계산하여 데이터와 영상 모두 같은 chunk에 저장한다. 전체 chunk 수 역시 episode 수로 계산한다.

학습 로더도 episode 인덱스와 chunk 크기로 파일 위치를 계산한다. 따라서 0부터 세었을 때 `episode_index=1000`, 즉 1,001번째 episode에서 기존 export는 `chunk-000`에 저장하지만 로더는 `chunk-001`에서 찾는 문제가 있었다.

| Episode 인덱스 | 수정 후 저장 chunk |
| --- | --- |
| 0–999 | `chunk-000` |
| 1000–1999 | `chunk-001` |
| 2000–2999 | `chunk-002` |

이는 파일 저장 규칙과 파일 조회 규칙을 같은 수식으로 맞춘 수정이다.

### 3.5 Train 범위와 컬럼 메타데이터 수정

**수정 전:** train 범위가 episode 수와 관계없이 `0:100`이었다. 일부 telemetry와 종료·성공 관련 컬럼은 Parquet에 저장하면서 `info.json`의 features에는 선언하지 않았다.  
**수정 후:** train 범위를 `0:실제_episode_수`로 만들고 저장하는 추가 컬럼의 dtype과 shape도 features에 등록했다.

이유는 메타데이터가 실제 파일 내용을 설명해야 하기 때문이다. 현재 자체 로더는 `episodes.jsonl`로 trajectory를 열거하므로 기존 train 문자열이 곧바로 100개 제한을 일으켰다고 단정할 수는 없다. 하지만 실제 수량과 다른 범위는 다른 도구와의 연동이나 데이터 점검에서 오해를 만든다.

### 3.6 출력 생성 전 데이터 계약 검사 강화

기존에 수행하던 일부 shape 검사를 출력 생성 전으로 옮기고, 다음 검사를 추가했다.

- task description이 비어 있지 않은 문자열인지 확인한다.
- manifest에 `state_names`, `action_names`가 있으면 현재 계약의 순서와 비교한다.
- episode frame 수가 양수인지 확인한다.
- state, action, RC telemetry, timestamp, source age·timestamp의 shape를 확인한다.
- state에 NaN 또는 무한대가 없는지 확인한다.

필드 순서는 차원만큼 중요하다. 예를 들어 action의 네 값이 모두 존재해도 surge와 yaw 순서가 바뀌면 학습 label의 의미가 달라진다. 또한 NaN이 정규화 통계에 들어가면 정상 frame의 처리까지 영향을 줄 수 있다. 이런 문제는 출력 파일 생성 전에 발견하는 것이 진단하기 쉽다.

다만 오래된 manifest에 필드 이름 목록 자체가 없다면 순서 검사는 생략한다. 이 경우 shape가 맞다는 사실만으로 실제 의미의 순서까지 증명할 수는 없다. 영상 파일 관련 검사는 계속 export 과정에서 수행하므로 모든 실패가 출력 생성 전에 발생하는 것은 아니다.

### 3.7 기존 인덱스 타입 확인 및 테스트 추가

`frame_index`, `episode_index`, 전체 `index`, `task_index`는 기존에도 int64로 생성하고 있었다. 이 타입을 변경한 것은 아니며 실제 Parquet 출력에서 유지되는지 테스트를 추가했다.

| 인덱스 | 역할 |
| --- | --- |
| `frame_index` | episode 내부에서 0부터 증가 |
| `episode_index` | export된 episode를 0부터 식별 |
| `index` | 전체 데이터의 frame을 끊김 없이 식별 |
| `task_index` | `tasks.jsonl`의 instruction을 참조 |

이 네 값은 서로 다른 범위를 가진다. Episode가 바뀔 때 frame index는 초기화되지만 전체 index는 계속 증가해야 한다.

## 4. 변경 파일

| 파일 | 역할 및 변경 |
| --- | --- |
| `kmu26_auv_vla_data_collector/contract.py` | 공통 `STATE_SLICES` 추가 |
| `kmu26_auv_vla_data_collector/collector.py` | 추론 관측 생성 시 공통 슬라이스 사용 |
| `kmu26_auv_vla_data_collector/export_lerobot.py` | 버전·dtype 선언, chunk 경로, train 범위, features 및 사전 검사 수정 |
| `test/test_export_contract.py` | 실제 학습 설정 소스를 읽어 키·순서·차원·horizon·padding 크기 대조 |
| `test/test_export_end_to_end.py` | MP4·Parquet 출력, 인덱스, chunk 경계, 잘못된 입력 거부 테스트 추가 |
| `README.md` | 올바른 CLI 설정과 저장·학습 차원 대응 문서화 |

학습 프로젝트 자체의 파일은 수정하지 않았다.

## 5. 검증 결과와 증거의 범위

별도 임시 export 환경에 지정 pandas·pyarrow 버전을 설치하고 다음 테스트를 실행했다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /tmp/auv-vla-export-check/bin/python -m pytest test -q -rs
```

결과는 **22개 통과, ROS 테스트 모듈 1개 생략**이다. ROS 테스트는 `auv_dvl_a50_msg.msg`를 import할 수 없어 생략되었다. `git diff --check`도 통과했다.

| 검증 항목 | 확인한 내용 | 범위 |
| --- | --- | --- |
| 학습 설정 대조 | 실제 `KMUauvDataConfig`의 필드·순서, 19차원 선택, 16-frame horizon, 64/32 padding 설정 | 소스 AST 검사, 모델 import 없음 |
| 파일 변환 | 실제 JPEG에서 MP4 생성, NPZ에서 Parquet 생성 | 테스트용 소형 episode |
| 영상 | 두 카메라 MP4의 frame 수와 FPS 확인 | 기본 end-to-end 사례 |
| 인덱스·타입 | float32 state/action, int64 인덱스, episode별 frame 초기화와 전체 index 연속성 | 실제 Parquet 읽기 |
| Chunk 경계 | info의 경로 템플릿으로 Parquet와 영상에 접근 가능 | 테스트에서 chunk 크기를 2로 줄여 3개 episode로 경계 재현 |
| 잘못된 입력 | 필드 순서 불일치, state 차원 오류·NaN, 빈 episode·task 거부 | 출력 폴더 생성 전 실패 확인 |
| 기존 계약 | RC 처리, DVL 축 변환, timestamp 연속성 등 기존 테스트 | 순수 Python 테스트 |

실제 1,001개 episode를 생성한 테스트는 아니다. 같은 chunk 계산을 작은 경계에서 검사한 것이다. 또한 전체 `LeRobotSingleDataset`을 통한 모델 학습, GPU forward/backward, 실제 차량 ROS 수집은 실행하지 않았다. 따라서 확인한 결과는 **설정 소스의 일치와 export 산출물의 테스트 통과**이며, 전체 학습 실행 성공을 의미하지 않는다.

## 6. 적용 방법과 남은 확인 사항

Export 출력 폴더를 학습에 사용할 때 다음 인자를 지정한다.

```text
--dataset-path /path/to/lerobot_train
--data-config kmu_auv
--embodiment-tag new_embodiment
```

이미 만들어 둔 export 데이터는 이번 코드 수정만으로 자동 갱신되지 않는다. 이번 변경을 적용하려면 staging 데이터를 새 빈 출력 폴더로 다시 export해야 한다.

현재 validity 네 값은 저장만 하며 학습 state에는 포함하지 않는다. DVL 누락 여부 등을 모델이 직접 구분하도록 설계를 확장하려면 validity의 학습 입력 포함 여부를 별도로 결정해야 한다.

다음 실제 실행 검증은 U0 학습 환경에서 export 데이터 로딩과 소규모 학습을 수행해 시간 묶기, 정규화, 영상 decode, 모델 텐서가 함께 연결되는지 확인하는 것이다. 차량에서는 ROS 메시지 의존성을 갖춘 환경에서 수집 노드와 센서 입력을 확인해야 한다.

## 7. 근거 파일

- 수집 계약: [contract.py](../kmu26_auv_vla_data_collector/contract.py)
- Export 구현: [export_lerobot.py](../kmu26_auv_vla_data_collector/export_lerobot.py)
- 추론 관측 생성: [collector.py](../kmu26_auv_vla_data_collector/collector.py)
- 학습 설정: `/home/pc/Downloads/auv_vla_training/gr00t/experiment/data_config.py`
- 학습 데이터 로더: `/home/pc/Downloads/auv_vla_training/gr00t/data/dataset.py`
- Modality 스키마: `/home/pc/Downloads/auv_vla_training/gr00t/data/schema.py`
- 모델 입력 transform: `/home/pc/Downloads/auv_vla_training/gr00t/model/transforms.py`
- Demo 버전 선언: `/home/pc/Downloads/auv_vla_training/demo_data/robot_sim.PickNPlace/meta/info.json`
- 학습 의존성: `/home/pc/Downloads/auv_vla_training/pyproject.toml`
