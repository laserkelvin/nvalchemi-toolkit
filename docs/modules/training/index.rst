.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Training module
===============

The :mod:`nvalchemi.training` subpackage defines the training-lifecycle
stage enum used to register hooks at specific points in a training run —
before and after the training run, each epoch, each batch, and each of
the forward, loss, backward, and optimizer-step phases.

.. seealso::

   - **User guide**: :ref:`hooks_guide` — conceptual overview of the
     hook protocol, context, and registry.
   - **Core framework**: :ref:`hooks-api` — the ``Hook`` protocol,
     ``HookContext``, and ``HookRegistryMixin``.
   - **Dynamics hooks**: :ref:`dynamics-hooks` — the sibling stage
     enum and built-in dynamics hooks.

.. toctree::
   :maxdepth: 1

   stages
