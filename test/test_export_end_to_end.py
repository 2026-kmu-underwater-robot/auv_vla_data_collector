import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from kmu26_auv_vla_data_collector.export_lerobot import export_dataset
from kmu26_auv_vla_data_collector.contract import ACTION_NAMES, STATE_NAMES

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _write_frames(directory: Path, count: int, shape: tuple[int, int, int]) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        image = np.full(shape, index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"frame_{index:06d}.jpg"), image)


def _write_episode(staging, index=7):
    episode = staging / f"episode_{index:06d}"
    _write_frames(episode / "frames" / "ego", 3, (8, 12, 3))
    _write_frames(episode / "frames" / "buoy_release", 3, (6, 10, 3))
    np.savez_compressed(
        episode / "samples.npz",
        observation_state=np.zeros((3, 23), dtype=np.float32),
        action=np.zeros((3, 4), dtype=np.float32),
        rc_pwm=np.full((3, 4), 1500, dtype=np.int32),
        rc_update_mask=np.ones((3, 4), dtype=np.float32),
        ros_timestamp=np.arange(3, dtype=np.float64) / 10.0,
        source_age=np.zeros((3, 7), dtype=np.float32),
        source_timestamp=np.zeros((3, 7), dtype=np.float64),
    )
    (episode / "manifest.json").write_text(
        json.dumps(
            {
                "episode_index": index,
                "task": "Approach the red buoy.",
                "success": True,
                "frames": 3,
                "fps": 10.0,
            }
        )
    )
    return episode


def test_export_dataset_creates_u0_compatible_layout(tmp_path):
    staging = tmp_path / "staging"
    _write_episode(staging)

    output = tmp_path / "lerobot"
    export_dataset(staging, output, requested_fps=None)

    table = pd.read_parquet(output / "data" / "chunk-000" / "episode_000000.parquet")
    assert len(table) == 3
    assert np.stack(table["observation.state"]).shape == (3, 23)
    assert np.stack(table["action"]).shape == (3, 4)
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["features"]["observation.images.ego"]["shape"] == [8, 12, 3]
    assert info["features"]["observation.images.buoy_release"]["shape"] == [6, 10, 3]
    assert info["codebase_version"] == "v2.0"
    assert info["splits"]["train"] == "0:1"
    assert table["index"].tolist() == [0, 1, 2]
    for key in ("index", "episode_index", "frame_index", "task_index"):
        assert table[key].dtype == np.dtype("int64")
    assert np.stack(table["observation.state"]).dtype == np.dtype("float32")
    assert np.stack(table["action"]).dtype == np.dtype("float32")
    assert set(table.columns) <= set(info["features"])
    for camera in ("observation.images.ego", "observation.images.buoy_release"):
        capture = cv2.VideoCapture(
            str(output / "videos" / "chunk-000" / camera / "episode_000000.mp4")
        )
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
        capture.release()


def test_chunk_boundary_and_global_indices(tmp_path, monkeypatch):
    import kmu26_auv_vla_data_collector.export_lerobot as exporter

    # Exercise the boundary with three real encoded episodes instead of 1001.
    monkeypatch.setattr(exporter, "CHUNKS_SIZE", 2)
    staging = tmp_path / "staging"
    for index in (7, 12, 31):
        _write_episode(staging, index)
    output = tmp_path / "lerobot"
    export_dataset(staging, output, requested_fps=None)
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_chunks"] == 2
    assert info["splits"]["train"] == "0:3"
    for index in range(3):
        values = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
        table = pd.read_parquet(output / info["data_path"].format(**values))
        assert table["episode_index"].tolist() == [index] * 3
        assert table["frame_index"].tolist() == [0, 1, 2]
        assert table["index"].tolist() == list(range(index * 3, index * 3 + 3))
        np.testing.assert_allclose(table["timestamp"], [0, 0.1, 0.2])
        for key in exporter.VIDEO_KEYS:
            assert (
                output / info["video_path"].format(**values, video_key=key)
            ).is_file()


@pytest.mark.parametrize(
    "error_kind",
    ["state_order", "action_order", "state_shape", "state_nan", "empty", "task"],
)
def test_invalid_contract_is_rejected_before_output(tmp_path, error_kind):
    staging = tmp_path / "staging"
    episode = _write_episode(staging)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if error_kind == "state_order":
        manifest["state_names"] = list(reversed(STATE_NAMES))
    elif error_kind == "action_order":
        manifest["action_names"] = list(reversed(ACTION_NAMES))
    elif error_kind == "empty":
        manifest["frames"] = 0
    elif error_kind == "task":
        manifest["task"] = " "
    else:
        with np.load(episode / "samples.npz") as samples:
            arrays = {key: samples[key] for key in samples.files}
        if error_kind == "state_shape":
            arrays["observation_state"] = np.zeros((3, 19), dtype=np.float32)
        else:
            arrays["observation_state"][0, 0] = np.nan
        np.savez_compressed(episode / "samples.npz", **arrays)
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "lerobot"
    with pytest.raises(ValueError):
        export_dataset(staging, output, requested_fps=None)
    assert not output.exists()
