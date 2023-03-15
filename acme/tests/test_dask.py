# -*- coding: utf-8 -*-
#
# Testing module for ACME's dask components
#

# Builtin/3rd party package imports
import pytest
import getpass
import numpy as np

# Import main actors here
from acme import cluster_cleanup, esi_cluster_setup, slurm_cluster_setup, local_cluster_setup
from conftest import useSLURM, onESI, defaultQ

def test_cluster_setup():

    # Tests which should work on any SLURM cluster
    if useSLURM:

        # Ensure invalid partition/memory specifications are caught
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition="invalid")
        cluster_cleanup()
        with pytest.raises(TypeError):
            slurm_cluster_setup(partition=defaultQ, mem_per_worker=(3))
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, mem_per_worker="invalidGB")
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, mem_per_worker="-110MB")
        cluster_cleanup()

        # Ensure additional sbatch parameters are processed correctly
        with pytest.raises(TypeError):
            slurm_cluster_setup(partition=defaultQ, job_extra="--output")
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, job_extra=["invalid"])
        cluster_cleanup()
        client = slurm_cluster_setup(partition=defaultQ,
                                     job_extra=["--job-name='averycustomjobname'"],
                                     interactive=False)
        assert 'averycustomjobname' in client.cluster.job_header
        cluster_cleanup()

        # Ensure output directory specification is parsed for correctness
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, job_extra=["--output=/path/to/nowhere"])
        cluster_cleanup()
        slurmOut = "/tmp/{}".format(getpass.getuser())
        client = slurm_cluster_setup(partition=defaultQ,
                                     n_workers=1,
                                     timeout=120,
                                     job_extra=["--output={}".format(slurmOut), "--job-name='averycustomjobname'"],
                                     interactive=False)
        assert "--output={}".format(slurmOut) in client.cluster.job_header
        assert 'averycustomjobname' in client.cluster.job_header
        cluster_cleanup(client)

        # Tests specific to the ESI HPC cluster
        if onESI:

            # Over-allocation of memory should default to partition max
            # (this should work on all clusters but we don't know partition
            # names, QoS rules etc.)
            client = esi_cluster_setup(partition="8GBDEV",
                                       timeout=120,
                                       n_workers=1,
                                       mem_per_worker="9000MB",
                                       interactive=False)
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert np.round(memory / 1000**3)[0] == 8

            # Invoking `esi_cluster_setup` with existing client must not start a new one
            clnt = esi_cluster_setup(partition="16GBXS", n_workers=2, interactive=False)
            assert clnt == client
            cluster_cleanup(client)

            # Specify CPU count manually
            client = esi_cluster_setup(partition="16GBDEV",
                                       timeout=120,
                                       n_workers=1,
                                       n_cores=1,
                                       interactive=False)
            assert [w["nthreads"] for w in client.cluster.scheduler_info["workers"].values()][0] == 1
            cluster_cleanup(client)

    else:

        # Trigger an exception by invoking `slurm_cluster_setup`` on non-SLURM node
        with pytest.raises(IOError) as err:
            slurm_cluster_setup()
            assert "Cannot access SLURM queuing system" in str(err)


def test_local_setup():

    # Allocate local distributed computing client w/custom settings
    client = local_cluster_setup(n_workers=2, mem_per_worker="2GB", interactive=False)
    wMem = [w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()]
    assert len(wMem) == 2
    memory = np.unique(wMem)
    assert memory.size == 1
    assert np.round(memory / 1000**3)[0] == 2
    cluster_cleanup()

    # Allocate local distributed computing client w/default settings
    client = local_cluster_setup(interactive=False)
    assert len(client.cluster.scheduler_info["workers"].keys()) > 1
    cluster_cleanup()


def test_backcompat_cluster():

    if useSLURM:
        client = slurm_cluster_setup(partition=defaultQ,
                                     timeout=120,
                                     n_jobs=1,
                                     workers_per_job=1,
                                     mem_per_job="1GB",
                                     n_jobs_startup=1,
                                     interactive=False)
        assert len(client.cluster.workers) == 1
        memory = [w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()]
        assert round(memory[0] / 1000**3) == 1
        threads = [w["nthreads"] for w in client.cluster.scheduler_info["workers"].values()]
        assert threads[0] == 1
        cluster_cleanup(client)
