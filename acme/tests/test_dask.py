# -*- coding: utf-8 -*-
#
# Testing module for ACME's dask components
#

# Builtin/3rd party package imports
import pytest
import getpass
import numpy as np

# Import main actors here
from acme.dask_helpers import customIOError, customValueError, customTypeError
from acme import cluster_cleanup, esi_cluster_setup, slurm_cluster_setup
from conftest import skip_if_not_linux, useSLURM, onESI, defaultQ

def test_cluster_setup():

    # ESI-cluster specific:
    # - don't start new cluster on top of existing one
    # - specify n_cores
    # - specify job_extra
    # - specify output
    # - get cpu count from mem spec

    # Tests which should work on any SLURM cluster
    if useSLURM:

        defaultInput = {"partition" : defaultQ,
                        "n_cores" : 1,
                        "n_workers": 1,
                        "processes_per_worker" : 1,
                        "mem_per_worker": "1GB",
                        "n_workers_startup" : 1,
                        "timeout" : 180,
                        "interactive" : False,
                        "interactive_wait" : 10,
                        "start_client" : True,
                        "job_extra" : []}

        # Ensure invalid partition/memory specifications are caught
        with pytest.raises(ValueError):
            faultyInput = dict(defaultInput)
            faultyInput["partition"] = "invalid"
            slurm_cluster_setup(**faultyInput)
        cluster_cleanup()
        with pytest.raises(TypeError):
            faultyInput = dict(defaultInput)
            faultyInput["mem_per_worker"] = (3)
            slurm_cluster_setup(**faultyInput)
        cluster_cleanup()
        with pytest.raises(ValueError):
            faultyInput = dict(defaultInput)
            faultyInput["mem_per_worker"] = "invalidGB"
            slurm_cluster_setup(**faultyInput)
        cluster_cleanup()
        with pytest.raises(ValueError):
            faultyInput = dict(defaultInput)
            faultyInput["mem_per_worker"] = "-110MB"
            slurm_cluster_setup(**faultyInput)
        cluster_cleanup()

        # Ensure additional sbatch parameters are processed correctly
        with pytest.raises(TypeError):
            faultyInput = dict(defaultInput)
            faultyInput["job_extra"] = "--output"
            slurm_cluster_setup(**faultyInput)
        cluster_cleanup()
        with pytest.raises(ValueError):
            faultyInput = dict(defaultInput)
            faultyInput["job_extra"] = ["invalid"]
            slurm_cluster_setup(**faultyInput)
        cluster_cleanup()
        customInput = dict(defaultInput)
        customInput["job_extra"] = ["--job-name='averycustomjobname'"]
        client = slurm_cluster_setup(**customInput)
        assert 'averycustomjobname' in client.cluster.job_header
        cluster_cleanup()

        # Ensure output directory specification is parsed for correctness
        with pytest.raises(ValueError):
            faultyInput = dict(defaultInput)
            faultyInput["job_extra"] = ["--output=/path/to/nowhere"]
            slurm_cluster_setup(**faultyInput)
        cluster_cleanup()
        customInput = dict(defaultInput)
        slurmOut = "/tmp/{}".format(getpass.getuser())
        customInput["job_extra"] = ["--output={}".format(slurmOut), "--job-name='averycustomjobname'"]
        client = slurm_cluster_setup(**customInput)
        assert "--output={}".format(slurmOut) in client.cluster.job_header
        assert 'averycustomjobname' in client.cluster.job_header
        cluster_cleanup(client)

        # Tests specific to the ESI HPC cluster
        if onESI:

            # Over-allocation of memory should default to partition max
            # (this should work on all clusters but we don't know partition
            # names, QoS rules etc.)
            client = esi_cluster_setup(partition="8GBDEV", n_workers=1, mem_per_worker="9000MB", interactive=False)
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert np.round(memory / 1000**3)[0] == 8

            # Invoking `esi_cluster_setup` with existing client must not start a new one
            clnt = esi_cluster_setup(partition="16GBXS", n_workers=2, interactive=False)
            assert clnt == client
            cluster_cleanup(client)

            # Specify CPU count manually
            client = esi_cluster_setup(partition="16GBDEV", n_workers=1, n_cores=1, interactive=False)
            assert "--cpus-per-task=1" in client.cluster.job_header
            cluster_cleanup(client)

def test_local_setup():
    pass

def test_backcompat_cluster():

    deprecatedInput = {"partition" : defaultQ,
                       "n_cores" : 1,
                       "n_jobs": 1,
                       "workers_per_job" : 1,
                       "mem_per_job": "1GB",
                       "n_jobs_startup" : 1,
                       "timeout" : 180,
                       "interactive" : False,
                       "interactive_wait" : 10,
                       "start_client" : True,
                       "job_extra" : []}

    client = slurm_cluster_setup(**deprecatedInput)
    assert len(client.cluster.workers) == 1
    memory = [w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()]
    assert round(memory[0] / 1000**3) == 1
    threads = [w["nthreads"] for w in client.cluster.scheduler_info["workers"].values()]
    assert threads[0] == 1
    cluster_cleanup(client)


#   Accordingly the keywords `n_jobs`, `mem_per_job`, `n_jobs_startup` and
#   `workers_per_job` have been renamed `n_workers`, `mem_per_worker`,
#   `n_workers_startup` and `processes_per_worker`, respectively. To ensure
