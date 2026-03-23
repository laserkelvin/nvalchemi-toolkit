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
"""Training stage enumeration for hook dispatch."""

from __future__ import annotations

from enum import Enum


class TrainingStage(Enum):
    """Stages in a training step where hooks can fire.

    Each member corresponds to a specific point in the training loop,
    allowing hooks to be triggered before or after key operations.

    Attributes
    ----------
    BEFORE_EPOCH : int
        Fired at the start of each epoch.
    AFTER_EPOCH : int
        Fired at the end of each epoch.
    BEFORE_BATCH : int
        Fired before processing a batch.
    AFTER_BATCH : int
        Fired after processing a batch.
    BEFORE_FORWARD : int
        Fired before the model forward pass.
    AFTER_FORWARD : int
        Fired after the model forward pass.
    BEFORE_BACKWARD : int
        Fired before the backward pass.
    AFTER_BACKWARD : int
        Fired after the backward pass.
    BEFORE_OPTIMIZER_STEP : int
        Fired before the optimizer step.
    AFTER_OPTIMIZER_STEP : int
        Fired after the optimizer step.
    ON_VALIDATION : int
        Fired when validation is performed.
    ON_CONVERGE : int
        Fired when a convergence criterion is met.
    """

    BEFORE_EPOCH = 0
    AFTER_EPOCH = 1
    BEFORE_BATCH = 2
    AFTER_BATCH = 3
    BEFORE_FORWARD = 4
    AFTER_FORWARD = 5
    BEFORE_BACKWARD = 6
    AFTER_BACKWARD = 7
    BEFORE_OPTIMIZER_STEP = 8
    AFTER_OPTIMIZER_STEP = 9
    ON_VALIDATION = 10
    ON_CONVERGE = 11
