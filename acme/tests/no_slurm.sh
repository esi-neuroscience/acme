#!/bin/bash

# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause

# Sabotage local tox environment so that `sinfo` is not working any more
if [ -n "$NO_SLURM" ]; then
    echo "ofnis" >| $VIRTUAL_ENV/bin/sinfo && chmod a+x $VIRTUAL_ENV/bin/sinfo
fi
