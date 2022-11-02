# -*- coding: utf-8 -*-
#
# Testing module for ACME's dask components
#

# Builtin/3rd party package imports
import getpass
import numpy as np

# Import main actors here
from acme.dask_helpers import customIOError
from acme import cluster_cleanup, esi_cluster_setup
from conftest import skip_if_not_linux, useSLURM, onESI, defaultQ

def test_cluster_setup():

    # ESI-cluster specific:
    # - don't start new cluster on top of existing one
    # - specify n_cores
    # - specify job_extra
    # - specify output

    # Test custom SLURM cluster setup
    if useSLURM:

        # Ensure invalid partition/memory specifications are caught
        with pytest.raises(customIOError):
            esi_cluster_setup(partition="invalid", interactive=False)
        cluster_cleanup()
        with pytest.raises(ValueError):
            esi_cluster_setup(mem_per_worker="invalidGB", interactive=False)
        cluster_cleanup()
        with pytest.raises(ValueError):
            esi_cluster_setup(mem_per_worker="-20MB", interactive=False)
        cluster_cleanup()

        # Over-allocation of memory should default to partition max
        client = esi_cluster_setup(partition="8GBDEV", n_workers=1, mem_per_worker="9000MB", interactive=False)
        memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
        assert memory.size == 1
        assert np.round(memory / 1000**3)[0] == 8
        cluster_cleanup(client)

        # Test if invalid extra args are caught
        slurmOut = "/cs/home/{}/acme_out".format(getpass.getuser())
        with pytest.raises(TypeError):
            esi_cluster_setup(job_extra="--output={}".format(slurmOut), interactive=False)
        cluster_cleanup()
        with pytest.raises(ValueError):
            esi_cluster_setup(job_extra=["output={}".format(slurmOut)], interactive=False)
        cluster_cleanup()
        with pytest.raises(ValueError):
            esi_cluster_setup(job_extra=["--output=/path/to/nowhere"], interactive=False)
        cluster_cleanup()

        # Supply extra args to start client for actual tests
        client = esi_cluster_setup(partition=defaultQ,
                                    n_workers=10,
                                    job_extra=["--output={}".format(slurmOut)],
                                    interactive=False)
        assert "--output={}".format(slurmOut) in client.cluster.job_header

        # Invoking `esi_cluster_setup` with existing client must not start a new one
        clnt = esi_cluster_setup(partition="16GBXS", n_workers=2, interactive=False)
        assert clnt == client


def test_local_setup():
    pass
