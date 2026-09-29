"""Clock and vehicle transitions cannot join unrelated demonstrations."""

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from mavros_msgs.msg import State
from std_srvs.srv import Trigger

from kmu26_auv_vla_data_collector.collector import Latest, VlaDataCollector


@pytest.fixture
def node(tmp_path):
    rclpy.init(args=["--ros-args", "-p", f"dataset_root:={tmp_path}"])
    n = VlaDataCollector()
    n._now = lambda: 10.0
    n._task_description = "Check connection."
    for key in ("ego", "release"):
        setattr(n, "_" + key, Latest(np.zeros((8, 8, 3), np.uint8), 10, 10, key))
    n._imu = Latest(
        (np.zeros(3), np.zeros(3), np.array([1, 0, 0, 0])), 10, 10, "base_link"
    )
    n._depth = Latest(1.0, 10, 10, "odom")
    n._rc_tracker.update([1500] * 18, 10)
    n._control = Latest(np.full(4, 1500), 10, 10, "")
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_clock_rewind_clears_cached_inputs(node):
    node._watch_clock()
    node._now = lambda: 0.0
    node._watch_clock()
    assert node._ego is None
    assert not node._rc_tracker.fresh(10, 0.5)


def test_mode_change_ends_episode(node):
    node._on_vehicle_state(State(connected=True, armed=True, mode="STABILIZE"))
    assert node._on_start_episode(Trigger.Request(), Trigger.Response()).success
    node._record_sample()
    node._on_vehicle_state(State(connected=True, armed=True, mode="MANUAL"))
    assert not node._active


def test_stale_camera_stops_demonstration(node):
    assert node._on_start_episode(Trigger.Request(), Trigger.Response()).success
    node._record_sample()
    node._collection_kind = "task_demonstration"
    node._demonstration_ready = lambda: True
    node._now = lambda: 10.1
    node._record_sample()  # Same camera stamp despite next sample.
    assert not node._active


def test_driver_held_altitude_is_invalid_on_bottom_lock_loss(node):
    node._dvl_data = Latest((2.0, False), 10, 10, "dvl_link")
    obs = node.policy_observation(np.zeros(4))
    assert obs["state.validity"][0, 3] == 0
    assert obs["state.altitude"][0, 0] == 0


def test_rejected_new_dvl_packet_cannot_revalidate_old_twist(node):
    node._dvl_data = Latest((2.0, True), 10, 10, "dvl_link")
    node._dvl_twist = Latest(np.array([1, 2, 3]), 9.9, 9.9, "dvl_link")
    assert node.policy_observation(np.zeros(4))["state.validity"][0, 2] == 0


def test_clock_pause_ends_episode(node, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        "kmu26_auv_vla_data_collector.collector.time.monotonic", lambda: clock[0]
    )
    assert node._on_start_episode(Trigger.Request(), Trigger.Response()).success
    node._record_sample()
    node._watch_clock()
    clock[0] += 1.1
    node._watch_clock()
    assert not node._active
    assert node._ego is None


def test_gap_ends_episode_without_compressing_time(node):
    assert node._on_start_episode(Trigger.Request(), Trigger.Response()).success
    node._record_sample()
    node._now = lambda: 10.3
    node._record_sample()
    assert not node._active
    assert len(node._states) == 1


@pytest.mark.parametrize("stream", ["imu", "depth", "rc"])
def test_required_stream_loss_interrupts_real_demonstration(node, stream):
    node._data_source = "real"
    node._collection_kind = "task_demonstration"
    node._session_id = "real-session"
    node._provenance_context = {"calibration": "test fixture"}
    node.count_publishers = lambda topic: 1
    node._on_vehicle_state(State(connected=True, armed=True, mode="STABILIZE"))
    assert node._on_start_episode(Trigger.Request(), Trigger.Response()).success
    node._record_sample()
    node._now = lambda: 10.1
    for key in ("ego", "release"):
        getattr(node, "_" + key).source_time = 10.1
    if stream == "rc":
        node._rc_tracker.update([0] * 18, 10.1)
    else:
        getattr(node, "_" + stream).source_time = 1.0
    node._record_sample()
    assert not node._active
    import json

    manifest = json.loads(
        next(node._dataset_root.glob("episode_*/manifest.json")).read_text()
    )
    assert manifest["success"] is False
    assert manifest["termination_reason"] == "stale_" + stream


@pytest.mark.parametrize(
    "publishers,age,ready",
    [(1, 0.1, True), (2, 0.1, False), (0, 0.1, False), (1, 2.1, False)],
)
def test_real_demonstration_checks_control_owner_and_connection(
    node, monkeypatch, publishers, age, ready
):
    clock = [100.0]
    monkeypatch.setattr(
        "kmu26_auv_vla_data_collector.collector.time.monotonic", lambda: clock[0]
    )
    node._data_source = "real"
    node._session_id = "real-session"
    node._provenance_context = {"calibration": "test fixture"}
    node.count_publishers = lambda topic: publishers
    node._on_vehicle_state(State(connected=True, armed=True, mode="STABILIZE"))
    clock[0] += age
    assert node._demonstration_ready() is ready
