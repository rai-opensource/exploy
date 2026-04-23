# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from mjlab.sensor.builtin_sensor import BuiltinSensor


class MjLabBuiltinSensorProxy:
    """Duck-type proxy for ``BuiltinSensor`` that returns a managed ONNX-traceable tensor.

    Replaces a ``BuiltinSensor`` in ``scene._sensors`` so that calls to
    ``scene["robot/imu_lin_vel"].data`` (and similar) return a plain PyTorch
    tensor that can be traced for ONNX export.

    After each simulation sense step, ``sync_from(original_sensor)`` must be
    called to refresh the managed tensor with the current sensor reading.
    """

    def __init__(self, original_sensor: BuiltinSensor):
        self._data = original_sensor.data.clone()

    @property
    def data(self) -> torch.Tensor:
        return self._data

    def sync_from(self, original_sensor: BuiltinSensor) -> None:
        """Refresh the managed tensor from the live warp sensor buffer.

        Reads directly from ``original_sensor._data_view`` (the raw warp tensor
        view) so that the most recent ``sim.sense()`` result is captured without
        going through the sensor's internal cache.
        """
        if original_sensor._data_view is not None:
            self._data.copy_(original_sensor._data_view)
        else:
            self._data.copy_(original_sensor.data)
