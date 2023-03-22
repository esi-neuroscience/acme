#
# Central pytest configuration
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import sys
import pytest

# Import main actors here
from acme.shared import is_slurm_node, is_esi_node

# Construct decorators for skipping certain tests
skip_if_not_linux = pytest.mark.skipif(sys.platform != "linux", reason="Only works in Linux")

# Perform SLURM-specific tests only on cluster nodes
useSLURM = is_slurm_node()

# Perform ESI-specific tests only the ESI HPC cluster
onESI = is_esi_node()

# Use a default partition if running on the ESI cluster
if onESI:
    defaultQ ="8GBXS"
else:
    defaultQ ="auto"
