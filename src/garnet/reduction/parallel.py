import os
import sys
import traceback

import multiprocess as multiprocessing
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np
from mantid import config

np.seterr(all="ignore", invalid="ignore")
config["Q.convention"] = "Crystallography"
# config.setLogLevel(4, quiet=False)

import warnings
import faulthandler

warnings.filterwarnings("ignore")
faulthandler.enable()

import pickle


def dethread_environment():
    config["MultiThreaded.MaxCores"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TBB_THREAD_ENABLED"] = "0"


def reset_environment():
    config["MultiThreaded.MaxCores"] = "4"
    os.environ.pop("OPENBLAS_NUM_THREADS", None)
    os.environ.pop("MKL_NUM_THREADS", None)
    os.environ.pop("NUMEXPR_NUM_THREADS", None)
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.pop("TBB_THREAD_ENABLED", None)


def _worker_call(func, kv):
    try:
        return func(kv)
    except Exception:
        traceback.print_exc()
        raise


class ParallelTasks:
    def __init__(self, function, combine=None):
        self.function = function
        self.combine = combine
        self.results = None

    def run_tasks(self, plan, n_proc):
        """
        Run parallel tasks with processing pool.

        Parameters
        ----------
        plan : dict
            Data reduction plan split over each process.
        n_proc : int
            Number of processes.

        """

        runs = plan["Runs"]
        split = [split.tolist() for split in np.array_split(runs, n_proc)]

        process_map = {}
        for proc_idx, s in enumerate(split):
            for r in s:
                try:
                    key = int(r)
                except Exception:
                    key = r
                process_map[key] = proc_idx

        plan["ProcessMap"] = process_map
        plan["NProc"] = n_proc

        join_args = [(plan, s, proc) for proc, s in enumerate(split)]

        dethread_environment()

        if n_proc == 1:
            self.results = [
                self.safe_function_wrapper(*args) for args in join_args
            ]
        else:
            pool = multiprocessing.Pool(processes=n_proc)

            def terminate_pool(e):
                print(e)
                pool.terminate()

            try:
                result = pool.starmap_async(
                    self.safe_function_wrapper,
                    join_args,
                    error_callback=terminate_pool,
                )
                self.results = result.get()
            except Exception as e:
                print("Exception in pool: {}".format(e))
                traceback.print_exc()
                pool.terminate()
                sys.exit()

            pool.close()
            pool.join()

        reset_environment()

        if self.combine is not None:
            self.combine(plan, self.results)

    def safe_function_wrapper(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            print("Exception in worker function: {}".format(e))
            traceback.print_exc()
            raise


class ParallelProcessor:
    def __init__(self, n_proc=1):
        self.n_proc = n_proc

    def process_dict(self, data, func):
        use_process = False
        if self.n_proc > 1:
            try:
                pickle.dumps(func)
                use_process = True
            except Exception:
                use_process = False

        dethread_environment()

        results = {}
        if self.n_proc > 1 and use_process:
            print("Process")
            with ProcessPoolExecutor(max_workers=self.n_proc) as executor:
                futures = [
                    executor.submit(_worker_call, func, kv)
                    for kv in data.items()
                ]
                for future in as_completed(futures):
                    try:
                        key, value = future.result()
                        results[key] = value
                    except Exception as e:
                        print("Exception in process pool: {}".format(e))
                        traceback.print_exc()
        else:
            if self.n_proc > 1:
                print("Thread")
                with ThreadPoolExecutor(max_workers=self.n_proc) as executor:
                    futures = [
                        executor.submit(func, (k, v)) for k, v in data.items()
                    ]
                    for future in as_completed(futures):
                        try:
                            key, value = future.result()
                            results[key] = value
                        except Exception as e:
                            print("Exception in thread pool: {}".format(e))
                            traceback.print_exc()
            else:
                print("Serial")
                for k, v in data.items():
                    try:
                        key, value = func((k, v))
                        results[key] = value
                    except Exception as e:
                        print("Exception in serial func call: {}".format(e))
                        traceback.print_exc()

        reset_environment()

        return results
