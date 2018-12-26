import time
from multiprocessing import Process, Queue

try:
    from watchdog.observers import Observer
    from watchdog.events import RegexMatchingEventHandler
except ImportError:
    raise ImportError('Nanonet component error: cannot import optional watchdog module. Install with pip.')


class Fast5Watcher(object):

    def __init__(self, path, timeout=10, regex='.*\.fast5$', initial_jobs=None):
        """Watch a path and yield modified files

        :param path: path to watch for files.
        :param timeout: timeout period for newly modified files.
        :param regex: regex filter for files to consifer.
        :param initial_jobs: pre-existing files to process.
        """
        self.path = path
        self.timeout = timeout
        self.regex = regex
        self.initial_jobs = initial_jobs
        self.q = Queue()
        self.watcher = Process(target=self._watcher)
        self.yielded = set()

    def _watcher(self):
        handler = RegexMatchingEventHandler(regexes=[self.regex], ignore_directories=True)
        handler.on_modified = lambda x: self.q.put(x.src_path)
        observer = Observer()
        observer.schedule(handler, self.path)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def __iter__(self):
        self.watcher.start()

        if self.initial_jobs is not None:
            for item in self.initial_jobs:
                if item not in self.yielded:
                    yield item
                    self.yielded.add(item)

        while True:
            try:
                item = self.q.get(True, self.timeout)
            except:
                break
            else:
                if item not in self.yielded:
                    yield item
                    self.yielded.add(item)
        self.watcher.terminate()
