from uuid import uuid4
from time import sleep
import os
from multiprocessing import Process
import Queue
from functools import partial

from myriad.components import MyriadServer
from myriad.managers import make_client

from RUBRIC.nanonet import stderr_redirected

__timeout__ = 0.5
__worker_startup_sleep__ = 2

class JobQueue(object):

    def __init__(self, jobs, functors):
        """A simple job queue which can be processed by various functors.

        :param jobs: iterable of job items.
        :param functions: tuples of the form (function, n_items), if n_itmes
            is None then the function should accept a single job items and
            process it to produce a single result. if n_items >= 2, then
            function should process a list of items, returning a list of
            results.
        """
        self.jobs = jobs
        self.functors = functors

    def __iter__(self):
        self.start_server()
        workers = [Process(target=partial(worker, f[0], f[1], self.port, self.authkey)) for f in self.functors]

        try:
            for w in workers:
                w.start()

            for result in self.server.imap_unordered(self.jobs, timeout=__timeout__):
                yield result

            for w in workers:
                w.terminate()
        except KeyboardInterrupt:
            for w in workers:
                w.terminate()
            self.server.manager.join()
            self.server.manager.shutdown()

    def start_server(self, ports=(5000,6000)):
        self.authkey = str(uuid4())

        server = None
        for port in xrange(*ports):
            try:
                with stderr_redirected(os.devnull):
                    server = MyriadServer(None, port, self.authkey)
            except EOFError:
                pass
            else:
                break
        if server is None:
            raise RuntimeError("Could not start myriad server.")

        self.server = server
        self.port = port


# On *nix the following could be part of the class above, but not on windows:
#    https://docs.python.org/2/library/multiprocessing.html#windows

def worker(function, take_n, port, authkey, timeout=__timeout__):
    """Worker function for JobQueue. Dispatches to singleton_worker or
    multi_worker as appropriate.

    :param function: function to apply in job items.
    :param take_n: number of items to process, should be None or >=2. Special
        case of None indicates function takes a single item to produce a single
        result.
    """
    sleep(__worker_startup_sleep__) # nasty, allows all workers to come up before iteration begins
    manager = make_client('localhost', port, authkey)
    job_q = manager.get_job_q()
    job_q_closed = manager.q_closed()
    result_q = manager.get_result_q()

    if take_n is None:
        _singleton_worker(function, job_q, job_q_closed, result_q, timeout=timeout)
    else:
        _multi_worker(function, take_n, job_q, job_q_closed, result_q, timeout=timeout)


def _singleton_worker(function, job_q, job_q_closed, result_q, timeout=__timeout__):
    while True:
        try:
            job = job_q.get_nowait()
            result = function(job)
            result_q.put(result)
        except Queue.Empty:
            if job_q_closed._getvalue().value:
                break
        sleep(timeout)


def _multi_worker(function, take_n, job_q, job_q_closed, result_q, timeout=__timeout__):
    while True:
        jobs = []
        try:
            for _ in xrange(take_n):
                job = job_q.get_nowait()
                jobs.append(job)
        except Queue.Empty:
            if job_q_closed._getvalue().value:
                break
        else:
            for i, res in enumerate(function(jobs)):
                result_q.put(res)
        sleep(timeout)
    if len(jobs) > 0:
        for i, res in enumerate(function(jobs)):
            result_q.put(res)

