# Real-vehicle recording fixes

These changes prevent corrupt or interrupted recordings from being mistaken for
training demonstrations. They do not arm the vehicle, publish RC commands, change
sensor mounting, or change the existing neutral PWM (1500) / span (300) defaults.
The `uuvsim` branch contains the separate simulator integration.

## What changed

| Failure | Behavior after this change |
| --- | --- |
| A camera freezes or replays the previous capture | Task recordings stop with `invalid_or_duplicate_camera`. |
| A50 loses bottom lock but holds its previous altitude | Altitude is zero with validity false; an unmatched raw/twist timestamp cannot revalidate an older velocity. |
| PWM exceeds the declared normalization range | The command is invalid instead of silently clipping to a different valid label. |
| Connection, arm state, mode, or RC publisher ownership changes | A task recording is interrupted and saved as unsuccessful. |
| Required IMU, depth, or RC input becomes stale | A task recording stops immediately and records the missing stream. |
| The collection clock rewinds or stops | The episode ends, and cached sensor/control/vehicle state is cleared. |
| Export receives invalid arrays, state/action ordering, or missing frames | Export rejects the input before creating output files. |
| DVL has no valid reading | NaN DVL timestamps remain allowed only when the corresponding validity flags are false. |
| Export loses collection evidence | Source manifests, termination reasons, collector source, FCU telemetry and receipt/axis timestamps are preserved. |
| FFmpeg is missing | Export reports it before starting; PATH and `~/.local/bin/ffmpeg` are checked. |

An RC publisher count is a ROS-side check. It cannot detect every direct MAVLink
sender or prove that the FCU applied a requested command. Recorded RCIn/RCOut are
separate diagnostic evidence, not instantaneous measured thrust.

## Collecting real demonstrations

The default `collection_kind: connection_check` remains suitable for inspecting
sensor wiring while disarmed. It must not be labelled as a training demonstration.
For real task collection, copy `config/collector.yaml` and set:

```yaml
collection_kind: task_demonstration
data_source: real
session_id: pool_session_001
provenance_file: /absolute/path/acquisition.json
expected_mode: STABILIZE
```

Keep these under the file's existing `ros__parameters` section. Launch with
`use_sim_time:=false`. Use the actual expected control mode and calibrated neutral
and span; do not copy simulator PWM settings. Populate the acquisition JSON with
the sensor calibration, launch configuration and FCU parameter evidence used for
that session. `tools/create_acquisition_context.py` can snapshot explicitly
selected local configuration files and their hashes. It does not read back the FCU
or certify calibration.

A nonempty task instruction, fresh required inputs, a connected and armed vehicle
in the expected mode, one RC publisher and the session/acquisition information are
required to begin a task recording. Success remains an operator-supplied label.
Interrupted recordings are kept with `success=false` and their termination reason.

Existing files are not rewritten. Legacy files can still be exported when their
arrays are valid, but absent provenance stays unknown. Re-exporting cannot invent
missing calibration, session, or per-frame flight-controller evidence. The model's
new real-data training profile rejects unverified legacy/connection-check data.
Keep all demonstrations from one session in the same train/validation split.

## Validation

Run `python3 -m pytest -q test` after sourcing ROS 2 Humble and the workspace
containing `auv_dvl_a50_msg` and MAVROS messages; install `requirements-export.txt`
and FFmpeg for exporter tests. ROS tests skip when ROS is absent, so inspect skips.

The PR was checked with 43 passing tests in an isolated, network-disabled ROS
container. Seven focused regressions fail against upstream `4b4e232` and pass with
these changes. A synthetic 20-frame, real-clock recording was exported and loaded
by the actual U0 loader into five complete 16-action chunks, with 23 valid state
values and four action channels. This is software integration validation, not a
physical-vehicle or underwater deployment test.
