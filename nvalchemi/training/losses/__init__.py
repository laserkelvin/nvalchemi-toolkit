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
that consume prediction and target tensors directly. Addition
(``energy_loss + force_loss``) builds a :class:`ComposedLossFunction`,
which routes keyed prediction/target mappings into those tensor-first
terms and returns a :class:`ComposedLossOutput` with the total loss and
per-component contributions. Loss coefficients and schedules belong on
the leaf loss terms' ``weight`` fields.
Schedule instances attached to a leaf loss's ``weight`` field are
reconstructed by ``TrainingStrategy`` from their ``(instance, spec)``
pair, mirroring the pattern used for models and optimizers.
"""

from __future__ import annotations

from nvalchemi.training.losses.base import LossWeightSchedule
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
    assert_same_shape,
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
)
from nvalchemi.training.losses.terms import (
    EnergyLoss,
    ForceLoss,
    StressLoss,
)

__all__ = [
    "BaseLossFunction",
    "ComposedLossFunction",
    "ComposedLossOutput",
    "ConstantWeight",
    "CosineWeight",
    "EnergyLoss",
    "ForceLoss",
    "LinearWeight",
    "LossWeightSchedule",
    "PiecewiseWeight",
    "StressLoss",
    "assert_same_shape",
    "frobenius_mse",
    "per_graph_mean",
    "per_graph_mse",
    "per_graph_sum",
]
