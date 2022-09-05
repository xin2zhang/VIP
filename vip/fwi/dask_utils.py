import os
from dask_jobqueue import SGECluster
from dask.distributed import Client, LocalCluster
import time
import shutil


def dask_local(nworkers, ph=1, odask='./dask/'):
    '''
    Create a local dask cluster
    Input
        nworkers: number of dask workers
        ph: number of threads per worker
        odask: directory to store temporary dask file
    Return
        dask cluster and client
    '''
    os.chdir(odask)

    cluster = LocalCluster(n_workers=nworkers, threads_per_worker=ph)
    client = Client(cluster)

    return cluster, client

# submit a dask job
def dask_init(pe, nnodes, nworkers=1, ph=1, odask='./dask/'):
    '''
    Initialise a dask cluster using SGE queue system
    Input
        pe: parallel environment on a hpc
        nnodes: number of nodes required
        nworkers: number of dask workers
        ph: number of processes per worker
        odask: directory to store temporary dask file
    Return
        dask cluster and client
    '''
    os.chdir(odask)

    if pe == 'skylake' or pe == 'skylake-misc':
        cores_per_node = nnodes*36
        memory_per_node = '280 GiB'
    elif pe == 'cascadelake' or pe == 'cascadelake-misc':
        cores_per_node = nnodes*96
        memory_per_node = '500 GiB'
    else:
        cores_per_node = nnodes*24
        memory_per_node = '500 GiB'
    runtime_limit = '800:00:00'
    project_name = 'attributes'

    cluster = SGECluster(
        processes = ph, # number of workers per job
        cores = ph, # make sure nthreads == 1, each dask worker forks one thread
        scheduler_options={"dashboard_address":":0"},
        job_extra = ['-pe {} {}'.format(pe, cores_per_node), '-cwd', '-j y', '-V'],
        memory = memory_per_node,
        project = project_name,
        walltime = runtime_limit
    )

    # get a client
    client = Client(cluster)

    # scale up
    total_workers = nworkers*ph
    cluster.scale(total_workers)
    print(cluster.job_script())

    while client.status == "running" and len(client.scheduler_info()['workers'])/ph < nworkers:
        time.sleep(1.0)

    return cluster, client


# remove a dask job as well as tmp directory
def dask_del(cluster, client, odask='./dask/'):
    client.close()
    cluster.close()
    # delete all dask file
    for file in os.listdir(odask):
        file_path =  os.path.join(odask, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

