# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss-function package: base class, schedules, and reduction helpers.

Step 1 exposes only the abstract :class:`BaseLossFunction`, the four
concrete weight schedules, the :class:`LossWeightSchedule` protocol, and
the graph-aware reduction primitives. The composition class
(``ComposedLossFunction``) and the concrete losses (``EnergyLoss``,
``ForceLoss``, ``StressLoss``) are added in later steps.
"""

from __future__ import annotations

from nvalchemi.training.losses._base import BaseLossFunction
from nvalchemi.training.losses._reductions import (
    frobenius_mse,
    per_graph_mean,
    per_graph_mse,
    per_graph_sum,
)
from nvalchemi.training.losses._schedules import (
    ConstantWeight,
    CosineWeight,
    LinearWeight,
    LossWeightSchedule,
    PiecewiseWeight,
    WeightScheduleField,
)

__all__ = [
    "BaseLossFunction",
    "ConstantWeight",
    "CosineWeight",
    "LinearWeight",
    "LossWeightSchedule",
    "PiecewiseWeight",
    "WeightScheduleField",
    "frobenius_mse",
    "per_graph_mean",
    "per_graph_mse",
    "per_graph_sum",
]
