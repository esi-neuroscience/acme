#
# Testing module for ACME's dask components
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import pytest
import getpass
import numpy as np
import dask.distributed as dd

# Import main actors here
from acme import cluster_cleanup, esi_cluster_setup, slurm_cluster_setup, local_cluster_setup
from conftest import useSLURM, onESI, onx86, defaultQ


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
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, mem_per_worker="1000XB")
        cluster_cleanup()

        # Ensure invalid worker/process/core count specifications are caught
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, n_workers=-3)
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, n_workers_startup=-3)
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, processes_per_worker=-3)
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, n_cores=-3)
        cluster_cleanup()

        # Ensure invalid timeouts are caught
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, timeout=-3)
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, interactive_wait=-3)
        cluster_cleanup()

        # Test Booleans
        with pytest.raises(TypeError):
            slurm_cluster_setup(partition=defaultQ, interactive=3)
        cluster_cleanup()
        with pytest.raises(TypeError):
            slurm_cluster_setup(partition=defaultQ, start_client=3)
        cluster_cleanup()

        # Ensure additional sbatch parameters are processed correctly
        with pytest.raises(TypeError):
            slurm_cluster_setup(partition=defaultQ, job_extra="--output")
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, job_extra=["invalid"])
        cluster_cleanup()
        with pytest.raises(TypeError):
            slurm_cluster_setup(partition=defaultQ, job_extra=[3])
        cluster_cleanup()
        client = slurm_cluster_setup(partition=defaultQ,
                                     job_extra=["--job-name='averycustomjobname'"],
                                     interactive=False)
        assert 'averycustomjobname' in client.cluster.job_header
        cluster_cleanup()

        # Trigger timeout errors
        with pytest.raises(TimeoutError):
            slurm_cluster_setup(partition=defaultQ, n_workers=1, timeout=1, interactive=False)

        # Ensure output directory specification is parsed for correctness
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, job_extra=["--output=/path/to/nowhere"])
        cluster_cleanup()
        with pytest.raises(ValueError):
            slurm_cluster_setup(partition=defaultQ, job_extra=["--output=invalid=path"])
        cluster_cleanup()
        slurmOut = f"/tmp/{getpass.getuser()}"
        client = slurm_cluster_setup(partition=defaultQ,
                                     n_workers=1,
                                     timeout=120,
                                     job_extra=[f"--output={slurmOut}", "--job-name='averycustomjobname'"],
                                     interactive=False)
        assert f"--output={slurmOut}" in client.cluster.job_header
        assert 'averycustomjobname' in client.cluster.job_header
        cluster_cleanup(client)

        # Tests specific to the ESI HPC cluster
        if onESI:

            # Over-allocation of memory should default to partition max
            # (this should work on all clusters but we don't know partition
            # names, QoS rules etc.)
            if onx86:
                client = esi_cluster_setup(partition=defaultQ,
                                           timeout=120,
                                           n_workers=1,
                                           mem_per_worker="9000MB",
                                           interactive=False)
                memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
                assert memory.size == 1
                assert np.ceil(memory / 1000**3)[0] == 8

                # Invoking `esi_cluster_setup` with existing client must not start a new one
                clnt = esi_cluster_setup(partition="16GBXS", n_workers=2, interactive=False)
                assert clnt == client
                cluster_cleanup(client)

                # Attempt to start a client on ppc64le
                with pytest.raises(ValueError) as valerr:
                    esi_cluster_setup(partition="E880")
                assert "ppc64le from submitting host with architecture x86_64" in str(valerr.value)

                # Define queue for testing CPU allocations below
                tmpQ = "24GBXS"

            else:

                # Attempt to start a client on x86
                with pytest.raises(ValueError) as valerr:
                    esi_cluster_setup(partition="8GBXS")
                assert "x86_64 from submitting host with architecture ppc64le" in str(valerr.value)

                # Define queue for testing CPU allocations below
                tmpq = defaultQ

            # Specify CPU count manually
            n_cores = 3
            client = esi_cluster_setup(partition=tmpQ,
                                       timeout=120,
                                       n_workers=1,
                                       cores_per_worker=n_cores,
                                       interactive=False)
            assert f"--cpus-per-task={n_cores}" in client.cluster.job_script()

            # Kill worker in client to trigger cleanup and startup of new
            # cluster when re-invoking setup routine
            client.retire_workers(list(client.scheduler_info()['workers']), close_workers=True)
            cluster = esi_cluster_setup(defaultQ,
                                        n_workers=1,
                                        mem_per_worker="1GB",
                                        start_client=False,
                                        interactive=False)
            memory = np.unique([w["memory_limit"] for w in cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert np.round(memory / 1000**3)[0] == 1

            # Ensure no client was started w/previous call
            with pytest.raises(ValueError):
                dd.get_client()

            # Manually close cluster
            client = dd.Client(cluster)
            client.retire_workers(list(client.scheduler_info()['workers']), close_workers=True)
            client.close()
            try:
                cluster.close()
            except:
                pass

            # Ensure job-list parsing works
            with pytest.raises(TypeError) as tperr:
                esi_cluster_setup(partition=defaultQ, job_extra="invalid")
            assert "`job_extra` has to be a list, not <class 'str'>" in str(tperr.value)

    else:

        # Trigger an exception by invoking `slurm_cluster_setup`` on non-SLURM node
        with pytest.raises(IOError) as err:
            slurm_cluster_setup()
        assert "Cannot access SLURM queuing system" in str(err)

    # Check if `cluster_cleanup` performs diligent error checking
    with pytest.raises(TypeError):
        cluster_cleanup(3)


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
    assert len(client.cluster.scheduler_info["workers"].keys()) >= 1
    cluster_cleanup()

    # Ensure error handling works
    with pytest.raises(TypeError):
        local_cluster_setup(interactive="invalid")
    cluster_cleanup()


def test_backcompat_cluster():

    if useSLURM:
        if onESI:
            client = esi_cluster_setup(partition=defaultQ,
                                       timeout=120,
                                       n_jobs=1,
                                       workers_per_job=1,
                                       mem_per_job="1GB",
                                       n_jobs_startup=1,
                                       interactive=False)
        else:
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
