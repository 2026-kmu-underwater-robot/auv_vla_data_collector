import ast
import os
from pathlib import Path

import pytest

from kmu26_auv_vla_data_collector.contract import ACTION_NAMES, STATE_NAMES
from kmu26_auv_vla_data_collector.export_lerobot import _modality


def test_exported_modality_matches_model_contract():
    modality = _modality()

    assert modality["state"]["validity"]["end"] == len(STATE_NAMES)
    assert modality["action"]["motion"]["end"] == len(ACTION_NAMES)
    assert modality["video"]["ego"]["original_key"] == "observation.images.ego"
    assert (
        modality["video"]["buoy_release"]["original_key"]
        == "observation.images.buoy_release"
    )


def test_contract_matches_local_training_project():
    """Read the real config without importing GPU/model dependencies."""
    project = Path(
        os.environ.get(
            "AUV_VLA_TRAINING_ROOT",
            str(Path(__file__).resolve().parents[2] / "auv_vla_training"),
        )
    )
    config_path = project / "gr00t" / "experiment" / "data_config.py"
    if not config_path.is_file():
        pytest.skip("Set AUV_VLA_TRAINING_ROOT to the U0 training checkout")
    module = ast.parse(config_path.read_text())
    config = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "KMUauvDataConfig"
    )
    fields = {
        node.targets[0].id: node.value
        for node in config.body
        if isinstance(node, ast.Assign)
    }
    modality = _modality()
    selected_state = ast.literal_eval(fields["state_keys"])
    selected_action = ast.literal_eval(fields["action_keys"])
    assert selected_state == [
        "state.attitude",
        "state.dvl_velocity",
        "state.depth",
        "state.prev_command",
        "state.angular_velocity",
        "state.linear_acceleration",
        "state.altitude",
    ]
    assert (
        sum(
            modality["state"][key.split(".", 1)[1]]["end"]
            - modality["state"][key.split(".", 1)[1]]["start"]
            for key in selected_state
        )
        == 19
    )
    assert selected_action == ["action.motion"]
    assert ast.literal_eval(fields["video_keys"]) == ["video.ego", "video.buoy_release"]
    assert ast.literal_eval(fields["language_keys"]) == [
        "annotation.human.action.task_description"
    ]
    assert ast.literal_eval(fields["observation_indices"]) == [0]
    assert ast.unparse(fields["action_indices"]) == "list(range(16))"
    model_transform = next(
        node
        for node in ast.walk(config)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GR00TTransform"
    )
    budgets = {item.arg: item.value for item in model_transform.keywords}
    assert ast.literal_eval(budgets["max_state_dim"]) == 64
    assert ast.literal_eval(budgets["max_action_dim"]) == 32
    for group in ("state", "action"):
        for item in modality[group].values():
            assert item["dtype"] == "float32"
    registry = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "DATA_CONFIG_MAP"
    )
    registered = dict(
        zip(
            (ast.literal_eval(key) for key in registry.value.keys),
            registry.value.values,
        )
    )
    assert registered["kmu_auv"].func.id == config.name
