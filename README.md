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
(`CHAN_RELEASE`)과 `65535` (`CHAN_NOCHANGE`)는 새로운 명령으로 학습하지 않고 직전 명령을
유지합니다.

## 빌드

```bash
cd /home/kuuve/auv_ros2
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install \
  --packages-ignore mavros_msgs \
  --packages-select dvl_msgs kmu26_auv_vla_data_collector
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
`kmu26_auv_vla` 저장소의 `--data-config kmu26_auv_real`과 바로 대응합니다.

## 중요한 제한

현재 collector는 수신한 DVL/IMU 벡터를 숫자 그대로 저장합니다. 학습 전에 두 토픽이 모델
계약과 동일한 body frame인지 반드시 확인해야 합니다. TF가 필요한 센서 프레임이라면
collector 앞단에서 `base_link` 기준 토픽으로 변환해야 합니다.
