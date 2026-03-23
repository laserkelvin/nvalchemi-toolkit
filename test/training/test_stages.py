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
"""Tests for TrainingStage enumeration."""

from __future__ import annotations

from nvalchemi.training import TrainingStage


class TestTrainingStage:
    """Test suite for TrainingStage enumeration."""

    def test_all_stages_exist(self) -> None:
        """Verify all 12 enum members exist."""
        expected = [
            "BEFORE_EPOCH",
            "AFTER_EPOCH",
            "BEFORE_BATCH",
            "AFTER_BATCH",
            "BEFORE_FORWARD",
            "AFTER_FORWARD",
            "BEFORE_BACKWARD",
            "AFTER_BACKWARD",
            "BEFORE_OPTIMIZER_STEP",
            "AFTER_OPTIMIZER_STEP",
            "ON_VALIDATION",
            "ON_CONVERGE",
        ]
        actual = [m.name for m in TrainingStage]
        assert len(actual) == 12
        assert set(actual) == set(expected)

    def test_values_are_integers(self) -> None:
        """Verify enum values are integers."""
        for member in TrainingStage:
            assert isinstance(member.value, int)

    def test_values_unique(self) -> None:
        """Verify all enum values are unique."""
        values = [m.value for m in TrainingStage]
        assert len(values) == len(set(values))

    def test_ordering(self) -> None:
        """Verify logical ordering of stages."""
        S = TrainingStage
        assert S.BEFORE_EPOCH.value < S.AFTER_EPOCH.value
        assert S.AFTER_EPOCH.value < S.BEFORE_BATCH.value
        assert S.BEFORE_BATCH.value < S.AFTER_BATCH.value
        assert S.BEFORE_FORWARD.value < S.AFTER_FORWARD.value
        assert S.BEFORE_BACKWARD.value < S.AFTER_BACKWARD.value
        assert S.BEFORE_OPTIMIZER_STEP.value < S.AFTER_OPTIMIZER_STEP.value
