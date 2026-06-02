#
# Output setup utilities for ACME
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import getpass
import datetime
import inspect
import logging
import h5py
import numpy as np
from typing import Optional, List, Tuple

# Fetch logger
log = logging.getLogger("ACME")


class OutputSetupError(Exception):
    """Output directory or file setup failed"""

    pass


class OutputDirectoryManager:
    """Handle output directory creation and management"""

    @staticmethod
    def create_output_directory(
        output_dir: Optional[str], single_file: bool, write_pickle: bool, func_name: str
    ) -> str:

        if not single_file and not write_pickle:
            log.debug("Preparing payload directory for HDF5 containers")
            payloadName = f"{func_name}_payload"
            outputDir = os.path.join(output_dir, payloadName)  # type: ignore
        else:
            msg = (
                "Either single-file output or pickling was requested. "
                + "Not creating payload directory"
            )
            log.debug(msg)
            outputDir = output_dir
        try:
            os.makedirs(outputDir)
            log.debug("Created %s", outputDir)
        except Exception as exc:
            err = "automatic creation of output folder %s failed: %s"
            log.error(err, outputDir, str(exc))
            raise OSError(err % (outputDir, str(exc)))

        return outputDir


class HDF5ContainerFactory:
    """Factory for creating HDF5 result containers"""

    @staticmethod
    def create_payload_directory(out_dir: str, func_name: str) -> str:
        """Create payload directory for worker files"""
        payload_name = f"{func_name}_payload"
        payload_dir = os.path.join(out_dir, payload_name)
        os.makedirs(payload_dir, exist_ok=True)
        return payload_dir

    @staticmethod
    def create_single_file_container(
        filename: str,
        task_ids: List[int],
        result_shape: Optional[tuple],
        result_dtype: str,
    ) -> str:
        """Create single HDF5 container with groups or dataset"""
        with h5py.File(filename, "w") as h5f:
            if result_shape is None:
                for i in task_ids:
                    h5f.create_group(f"comp_{i}")
            else:
                if np.inf in result_shape:
                    act_shape = tuple(
                        spec if spec is not np.inf else 1 for spec in result_shape
                    )
                    max_shape = tuple(
                        spec if spec is not np.inf else None for spec in result_shape
                    )
                else:
                    act_shape = result_shape
                    max_shape = None

                h5f.create_dataset(
                    "result_0", shape=act_shape, maxshape=max_shape, dtype=result_dtype
                )
        return filename

    @staticmethod
    def create_virtual_dataset_container(
        filename: str,
        task_ids: List[int],
        worker_filenames: List[str],
        result_shape: Optional[tuple],
        stacking_dim: int,
        result_dtype: str,
        payload_dir: str,
    ) -> str:
        """Create HDF5 container with virtual dataset pointing to worker files"""
        if result_shape is None:
            # For no shape, create external links instead of virtual dataset
            with h5py.File(filename, "w") as h5f:
                for i, fname in enumerate(worker_filenames):
                    relPath = os.path.join(
                        os.path.basename(payload_dir), os.path.basename(fname)
                    )
                    h5f[f"comp_{i}"] = h5py.ExternalLink(relPath, "/")
            return filename

        VSourceShape = [spec if spec is not np.inf else None for spec in result_shape]
        VSourceShape.pop(stacking_dim)
        VSourceShape = tuple(VSourceShape)

        if None in VSourceShape:
            resActShape = tuple(
                spec if spec is not np.inf else 1 for spec in result_shape
            )
            resMaxShape = tuple(
                spec if spec is not np.inf else None for spec in result_shape
            )
            vsActShape = tuple(spec if spec is not None else 1 for spec in VSourceShape)
            vsMaxShape = VSourceShape
        else:
            resActShape = result_shape
            resMaxShape = None
            vsActShape = VSourceShape
            vsMaxShape = None

        layout = h5py.VirtualLayout(
            shape=resActShape, dtype=result_dtype, maxshape=resMaxShape
        )

        idx = [
            slice(None) if spec is not np.inf else slice(h5py.h5s.UNLIMITED)
            for spec in result_shape
        ]
        jdx = list(idx)
        jdx.pop(stacking_dim)

        for i, fname in enumerate(worker_filenames):
            idx[stacking_dim] = i
            rel_path = os.path.join(
                os.path.basename(payload_dir), os.path.basename(fname)
            )
            vsource = h5py.VirtualSource(
                fname, "result_0", shape=vsActShape, maxshape=vsMaxShape
            )
            layout[tuple(idx)] = vsource[tuple(jdx)]

        with h5py.File(filename, "w", libver="latest") as h5f:
            h5f.create_virtual_dataset("result_0", layout)

        return filename
