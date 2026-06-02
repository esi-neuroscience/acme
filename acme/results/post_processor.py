#
# Result post-processing utilities for ACME
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import shutil
import logging
import h5py
import numpy as np
from typing import Union, List, Optional

# Fetch logger
log = logging.getLogger("ACME")


class ResultPostProcessor:
    """Handle post-processing of distributed computation results"""

    def __init__(self, client, results_dir: Optional[str] = None):
        self.client = client
        self.results_dir = results_dir
        self.successMsg = "SUCCESS!"
        self.finalMsg = "Results have been saved to {}"

    def process_futures(
        self,
        futures: List,
        collect_results: bool,
        result_shape: Optional[tuple],
        stacking_dim: Optional[int],
        result_dtype: str,
        acme_func,
        original_func,
        kwargv: dict,
    ) -> Union[List, str, None]:
        """Process completed futures and return results"""

        # Determine output mode
        write_worker_results = acme_func == original_func
        single_file = kwargv.get("singleFile") is not None
        write_pickle = write_worker_results and not self.results_dir
        msg = "Inferred that `write_worker_results = %s`, `single_file = %s`, `write_pickle = %s`"
        log.debug(msg, str(write_worker_results), str(single_file), str(write_pickle))

        # Handle in-memory collection
        if collect_results:
            results = self._collect_in_memory(
                futures, result_shape, stacking_dim, result_dtype
            )

        # Handle file-based results
        if write_worker_results:
            results = self._process_file_results(
                futures, kwargv, result_shape, stacking_dim
            )

        # Print final triumphant output message and force-flush all logging handlers
        if len(self.successMsg) > 0:
            log.announce(self.successMsg)  # type: ignore
        log.info(self.finalMsg)
        for h in log.handlers:
            if hasattr(h, "flush"):
                h.flush()

        return results

    def _collect_in_memory(
        self,
        futures: List,
        result_shape: Optional[tuple],
        stacking_dim: Optional[int],
        result_dtype: str,
    ) -> Union[List, np.ndarray]:
        """Collect results from futures into local memory"""

        if not isSpyModule:
            log.info("Gathering results in local memory")

        collected = self.client.gather(futures)
        log.debug("Gathered results from client in a %d-element list", len(collected))
        self.finalMsg = "Finished parallel computation"

        if result_shape is not None:
            log.debug(
                "Returning single NumPy array of shape %s and type %s",
                str(result_shape),
                str(result_dtype),
            )

            arr_val = np.empty(shape=result_shape, dtype=result_dtype)
            idx = [slice(None)] * len(result_shape)
            values = []

            for i, res in enumerate(collected):
                if not isinstance(res, (list, tuple)):
                    res = [res]
                idx[stacking_dim] = i
                arr_val[tuple(idx)] = res[0]
                for r in res[1:]:
                    values.append(r)

            values.insert(0, arr_val)

            if len(values) == 1:
                return values[0]
            return values

        log.debug("Returning a list of values")
        return collected

    def _process_file_results(
        self,
        futures: List,
        kwargv: dict,
        result_shape: Optional[tuple],
        stacking_dim: Optional[int],
    ) -> str:
        """Process file-based results and handle error recovery"""
        write_pickle = self.results_dir is None
        single_file = kwargv.get("singleFile") is not None

        if write_pickle:
            return self._handle_pickle_results(kwargv, results_dir)
        elif single_file:
            return self._handle_single_file_results()
        else:
            return self._handle_multiple_files_results(
                kwargv, result_shape, stacking_dim, results_dir
            )

    def _handle_pickle_results(self, kwargv: dict) -> List[str]:
        """Handle results saved as pickle files"""
        log.debug("Saved results as pickle files")
        values = list(kwargv["outFile"])
        self.finalMsg.format(self.results_dir)
        log.debug("Returning a list of file-names")
        return values

    def _handle_single_file_results(self) -> List[str]:
        """Handle results saved to single shared container"""
        log.debug("Saved results to single shared container")
        self.finalMsg = "Results have been saved to %s"
        self.finalMsg.format(self.results_dir)
        log.debug("Returning container name as single-element list")
        return [self.results_dir]

    def _handle_multiple_files_results(
        self,
        kwargv: dict,
        result_shape: Optional[tuple],
        stacking_dim: Optional[int],
    ) -> List[str]:
        """Handle results saved to multiple files with payload directory"""
        log.debug("Scanning payload directory for emergency pickles")
        picklesFound = False
        values = []
        for fname in kwargv["outFile"]:
            pklName = fname.rstrip(".h5") + ".pickle"
            if os.path.isfile(fname):
                values.append(fname)
            elif os.path.isfile(pklName):
                values.append(pklName)
                picklesFound = True
                log.debug("Found emergency pickle %s", pklName)
            else:
                missing = fname.rstrip(".h5")
                values.append("Missing %s" % (missing))
                log.debug("Missing file %s", missing)
        payloadDir = os.path.dirname(values[0])

        # If pickles are found, handle emergency recovery
        if picklesFound:
            return self._handle_emergency_pickles(kwargv, values, payloadDir)

        # All good, no pickle gymnastics was needed
        return self._handle_normal_file_results(
            kwargv, values, payloadDir, result_shape, stacking_dim
        )

    def _handle_emergency_pickles(
        self, kwargv: dict, values: List[str], payloadDir: str
    ) -> List[str]:
        """Handle emergency pickle recovery scenario"""
        results_container = kwargv.get("results_container")
        os.unlink(results_container)
        wrng = (
            "Some compute runs could not be saved as HDF5, "
            + "collection container %s has been removed as it would "
            + "comprise invalid file-links"
        )
        log.warning(wrng, results_container)

        # Move files out of payload dir and update return values
        target = os.path.abspath(os.path.join(payloadDir, os.pardir))
        for i, fname in enumerate(values):
            shutil.move(fname, target)
            kwargv["outFile"][i] = os.path.join(target, os.path.basename(fname))
            log.debug("Moved %s to %s", fname, target)
        values = list(kwargv["outFile"])
        log.debug("Returning a list of file-names")
        shutil.rmtree(payloadDir)
        log.debug("Deleted payload directory %s", payloadDir)
        self.successMsg = ""
        self.finalMsg.format(target)

        return values

    def _handle_normal_file_results(
        self,
        kwargv: dict,
        values: List[str],
        payloadDir: str,
        result_shape: Optional[tuple],
        stacking_dim: Optional[int],
    ) -> List[str]:
        """Handle normal file-based results scenario"""
        results_container = kwargv.get("results_container")
        log.debug("No emergency pickles found")

        # In case of multiple return values present in by-worker
        # containers but missing in collection container
        if stacking_dim is not None:
            self._add_missing_return_values(results_container, values, payloadDir)

        self.finalMsg.format(results_container)
        msg = "Container ready, links to data payload located in %s"
        log.debug(msg, payloadDir)
        log.debug("Returning a list of file-names")
        return values

    def _add_missing_return_values(
        self, results_container: str, values: List[str], payloadDir: str
    ) -> None:
        """Add missing return values to collection container via external links"""
        log.debug(
            "Check if additional return values need to be added to container with pre-allocated dataset"
        )
        with h5py.File(results_container, "r") as h5r:
            with h5py.File(values[0], "r") as h5Tmp:
                missingReturns = set(h5Tmp.keys()).difference(h5r.keys())
        if len(missingReturns) > 0:
            log.debug("Found return values to be added")
            with h5py.File(results_container, "a") as h5r:
                for retVal in missingReturns:
                    for i, fname in enumerate(values):
                        relPath = os.path.join(
                            os.path.basename(payloadDir),
                            os.path.basename(fname),
                        )
                        h5r[f"comp_{i}/{retVal}"] = h5py.ExternalLink(relPath, retVal)
                        log.debug(
                            "Added return value via external link comp_%d/%s",
                            i,
                            retVal,
                        )
