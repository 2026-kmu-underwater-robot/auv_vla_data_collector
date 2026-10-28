# KMU26 AUV VLA data collector

실제 KMU26 AUV에서 U0 파인튜닝용 episode를 수집하는 ROS 2 패키지입니다. 다음 데이터를
10 Hz로 묶어 저장합니다.

- 전방 compressed RGB 카메라
- 부표 분리부 compressed RGB 카메라
- DVL 속도와 고도/validity
- MAVROS IMU 각속도, 선형가속도, 자세 quaternion
- 압력 센서에서 변환된 수심
- `/mavros/rc/override`의 surge, sway, heave, yaw 조작 명령
- 자연어 task description

RC override는 motor PWM이 아니라 다음 순서의 정규화된 action label로 저장됩니다.

```text
[surge, sway, heave, yaw] ∈ [-1, 1]
```

ArduSub 기본 채널은 각각 `[5, 6, 3, 4]`이며 파라미터로 변경할 수 있습니다. `0`
(`CHAN_RELEASE`)는 해당 축의 제어권을 무효화합니다. `65535` (`CHAN_NOCHANGE`)는
직전 명령을 유지하지만 유효 시간을 갱신하지 않습니다.

## 빌드

```bash
cd /home/kuuve/auv_ros2
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install \
  --packages-ignore mavros_msgs \
  --packages-select auv_dvl_a50_msg kmu26_auv_vla_data_collector
source install/setup.bash
```

이 워크스페이스에는 ROS 1 보관 소스와 ROS 2 MAVROS 소스가 함께 있어, 위 명령은 ROS 1
디렉터리의 동명 `mavros_msgs`가 검색되는 것을 피하고 `/opt/ros/humble`의 메시지를 사용합니다.

## 실행

두 번째 카메라의 실제 토픽을 launch argument로 지정합니다.

```bash
ros2 launch kmu26_auv_vla_data_collector collector.launch.py \
  buoy_release_image_topic:=/actual/release/camera/image_raw/compressed
```

전체 설정은 [config/collector.yaml](config/collector.yaml)에 있습니다. 특히 수집 전에 다음을
실제 장비와 대조해야 합니다.

- 두 번째 카메라 토픽
- RC channel 순서 및 각 축 부호
- `/dvl/twist`가 사용하는 좌표축
- `/depth/pose.position.z`가 위쪽 양수인지 여부
- PWM 중립점과 span

## Episode 수집

먼저 영어 task instruction을 publish합니다.

```bash
ros2 topic pub --once /vla/task_description std_msgs/msg/String \
  "{data: 'Approach the red buoy.'}"
```

모든 필수 입력이 최신 상태일 때만 episode가 시작됩니다.

```bash
ros2 service call /vla_data_collector/start_episode std_srvs/srv/Trigger "{}"
```

성공한 수행을 저장합니다.

```bash
ros2 service call /vla_data_collector/stop_episode std_srvs/srv/SetBool "{data: true}"
```

실패했지만 recovery 학습에 사용할 수행은 `data: false`로 저장합니다. 센서 설정 오류나
사람이 끼어든 수행처럼 학습에 사용하면 안 되는 episode는 저장 전에 폐기합니다.

```bash
ros2 service call /vla_data_collector/discard_episode std_srvs/srv/Trigger "{}"
```

노드가 recording 도중 종료되면 수집된 frame이 있는 episode는 `success=false`와
`termination_reason=node_shutdown`으로 보존됩니다.

## 저장 구조

수집 단계에서는 ROS 런타임에 pandas/pyarrow를 요구하지 않도록 JPEG와 NPZ로 저장합니다.

```text
vla_data/staging/
└── episode_000000/
    ├── manifest.json
    ├── samples.npz
    └── frames/
        ├── ego/frame_000000.jpg
        └── buoy_release/frame_000000.jpg
```

`samples.npz`에는 23차원 `observation_state`, 4차원 `action`, 원본 RC PWM과 각 채널의
update mask, ROS timestamp, 각 센서의 원본 timestamp와 sample 시점 기준 age가 포함됩니다.

## LeRobot/U0 형식으로 변환

변환은 `pandas`, `pyarrow`, `ffmpeg`가 설치된 U0 학습 환경에서 실행합니다.

```bash
source /home/kuuve/auv_ros2/install/setup.bash
ros2 run kmu26_auv_vla_data_collector export_lerobot \
  /home/kuuve/auv_ros2/vla_data/staging \
  /home/kuuve/auv_ros2/vla_data/lerobot_train
```

출력 폴더가 비어 있지 않으면 변환기는 덮어쓰지 않고 중단합니다. 변환 결과는
`/home/pc/Downloads/auv_vla_training`의 `--data-config kmu_auv`과 대응합니다.

### 학습 프로젝트와의 데이터 계약

학습 프로젝트는 Python `lerobot` 패키지를 의존성으로 사용하지 않고
`gr00t/data/dataset.py`의 자체 로더로 읽습니다. 따라서 맞춰야 하는 버전은
pip 패키지 버전이 아니라 데이터 저장 형식입니다. Export는 프로젝트의 demo와 같은
`codebase_version: v2.0` 구조(episode별 Parquet/MP4, JSON/JSONL meta)를 생성합니다.
최신 LeRobot v3 형식으로 재변환하지 말고 이 출력 폴더를 그대로 사용합니다.
Export 의존성의 `pandas==2.2.3`, `pyarrow==14.0.1`, NumPy 1.x는 학습 프로젝트와 맞습니다.

`meta/modality.json`의 인덱스는 Python 슬라이스처럼 시작 포함, 끝 제외입니다.

| State 항목 | 저장 인덱스 | 차원 |
| --- | --- | --- |
| prev_command | `[0:4]` | 4 |
| dvl_velocity | `[4:7]` | 3 |
| angular_velocity | `[7:10]` | 3 |
| linear_acceleration | `[10:13]` | 3 |
| attitude (w, x, y, z) | `[13:17]` | 4 |
| depth | `[17:18]` | 1 |
| altitude | `[18:19]` | 1 |
| validity | `[19:23]` | 4 |

저장 state는 `float32[23]`입니다. 현재 `KMUauvDataConfig`는 validity를 제외하고
`attitude → dvl_velocity → depth → prev_command → angular_velocity → linear_acceleration
→ altitude` 순서로 19차원을 연결합니다. 유효성 표시는 파일에 보존되지만 현재 모델
입력에는 포함되지 않습니다. 학습 transform이 이를 64차원으로 zero padding합니다.
저장 데이터를 64차원으로 늘릴 필요는 없습니다.

Action은 `float32[4]`의 `[surge, sway, heave, yaw]`이고, 학습 시 현재 frame부터
16개(`0..15`)의 action을 모아 `[16, 32]`로 padding합니다. 저장 파일에는 frame당
4차원만 씁니다. Episode 끝의 부족한 action은 로더가 마지막 값으로 padding합니다.
관측은 현재 frame 하나(`observation_indices=[0]`)입니다.

두 카메라는 RGB, 저장 shape는 `[H, W, 3]`이며 학습 transform에서 각각 224×224로
resize합니다. 수집 기본값은 10 Hz입니다. 모든 episode의 실제 수집 FPS가 같아야 하며,
`--fps`로 시간을 바꿀 수 없습니다.

`frame_index`는 episode마다 0부터, `episode_index`와 `task_index`도 0부터 시작합니다.
`index`는 전체 데이터에서 끊김 없이 증가하며 네 인덱스 모두 `int64`입니다.
저장 chunk는 episode 1,000개 단위로 증가하고 train 범위는 실제 episode 수를 사용합니다.

학습 실행 시 다음 인자를 사용합니다(모델과 나머지 학습 옵션은 기존 설정 사용).

```text
--dataset-path /path/to/lerobot_train --data-config kmu_auv --embodiment-tag new_embodiment
```

`u0_bot` 설정은 joint/PWM/target position을 사용하는 다른 데이터 계약입니다.
현재 AUV 수집 데이터에는 `kmu_auv`를 사용합니다.

계약 및 export 테스트는 ROS와 별도의 export 환경에서 실행할 수 있습니다.

```bash
python -m pip install -r requirements-export.txt pytest
# OpenCV와 시스템 ffmpeg도 필요합니다.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  AUV_VLA_TRAINING_ROOT=/home/pc/Downloads/auv_vla_training python -m pytest test -q
```

학습 프로젝트 대조 테스트는 GPU 의존성 없이 실제 설정 소스를 검사합니다.
전체 학습 로더와 모델 실행 검증은 U0 학습 환경에서 별도로 수행해야 합니다.

## 중요한 제한

현재 ROV 기본 DVL 축 변환과 시각/RC 유효성 정책은 아래 Current ROV 절을 따릅니다.
다른 장착 방향은 앞단에서 body frame으로 변환해야 합니다.

## Current ROV / MuJoCo integration

Use the checked-in `auv_dvl_a50_msg` dependency, not legacy `dvl_msgs`.
The defaults now use `/imx219/camera0/image_raw/compressed` and
`/imx219/camera1/image_raw/compressed`. Camera 1 must actually depict the release
area for this embodiment; a stereo partner is not automatically a release camera.
The dataset directory defaults to `~/vla_data/staging`.

```bash
colcon build --packages-select auv_dvl_a50_msg kmu26_auv_vla_data_collector
source install/setup.bash
# Simulation only; omit use_sim_time for the vehicle.
ros2 launch kmu26_auv_vla_data_collector collector.launch.py use_sim_time:=true
```

`dvl_link` FRD vectors are converted once to body FLU (`x, -y, -z`). This matches
the current ROV's X-pi DVL mounting transform. For a different mounting, provide
an upstream body-frame velocity topic and set `dvl_input_frame:=base_link` and
`dvl_convention:=FLU` in a custom YAML. Unknown DVL/IMU frames are rejected;
IMU must already be in `body_frame` (default `base_link`). The collector does
not infer arbitrary mounting transforms. Depth remains positive down in metres.

Capture and receipt ages must both be valid. Clock resets and missed sampling
intervals end the current recording with `sampling_discontinuity`; start a new
episode after recovery. Export rejects gaps/rate changes instead of silently
compressing time. Normal timer jitter up to 25% of a period is accepted.
RC RELEASE invalidates the affected action immediately; NOCHANGE cannot renew
an expired action. All four primary RC channels must have recent explicit
commands. The prior helper remains available, but the recorder uses per-axis
ownership tracking. Keep the same ArduSub mode, axis signs, neutral and PWM span
across demonstrations and policy execution; default span 300 matches joy2mavros.

Use `requirements-export.txt` in a separate export environment (plus OpenCV and
system `ffmpeg`). Keep failed/recovery episodes separate unless deliberately
including them in training; export does not silently filter `success=false`.
The public `policy_observation(previous_command)` method exposes the same sensor
contract to an inference adapter without publishing RC or requiring an RC source.
