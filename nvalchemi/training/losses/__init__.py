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
"""Loss-function abstractions, schedules, terms, and reductions.

Loss terms are Pydantic-serializable :class:`BaseLossFunction` instances
combinable via arithmetic (``2.0 * energy_loss + 10.0 * force_loss``).
:class:`ComposedLossFunction` represents the resulting weighted sum and
round-trips through :class:`~nvalchemi.training.BaseSpec`.
"""

from __future__ import annotations

from nvalchemi.training.losses.base import LossWeightSchedule
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    ComposedLossFunction,
)
from nvalchemi.training.losses.reductions import (
    frobenius_mse,
    per_graph_mean,
    per_graph_mse,
    per_graph_sum,
)
from nvalchemi.training.losses.schedules import (
    ConstantWeight,
    CosineWeight,
    LinearWeight,
    PiecewiseWeight,
    WeightScheduleField,
)

__all__ = [
    "BaseLossFunction",
    "ComposedLossFunction",
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
