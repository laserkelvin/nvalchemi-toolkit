.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Data module (AtomicData, Batch, readers/writers)
================================================

.. currentmodule:: nvalchemi.data

Core classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomicData
   Batch

I/O and pipelines
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomicDataZarrWriter
   AtomicDataZarrReader
   Dataset
   DataLoader
   SizeAwareBatchSampler
   Reader

Capacity schedules
------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   LinearCapacitySchedule
   CosineCapacitySchedule
   PiecewiseCapacitySchedule

Write configuration
-------------------

.. currentmodule:: nvalchemi.data.datapipes

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ZarrArrayConfig
   ZarrWriteConfig
