# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2019 Intelligence Team, R&D Center.
import sys
import logging
import logging.config
import traceback
import threading
import multiprocessing

from logging import FileHandler as fh
from logging import StreamHandler as sh

# ============================================================================
# Define Log Handler
# ============================================================================
class MPLogHandler(logging.Handler):
    """multiprocessing log handler

    This handler makes it possible for several processes
    to log to the same file by using a queue.

    """
    def __init__(self, fname):
        logging.Handler.__init__(self)

        self._f_handler = fh(fname)
        self._f_handler.suffix = '%Y-%m-%d'
        self._s_handler = sh()
        self.queue = multiprocessing.Queue(-1)

        thrd = threading.Thread(target=self.receive)
        thrd.daemon = True
        thrd.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._f_handler.setFormatter(fmt)
        self._s_handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._f_handler.emit(record)
                self._s_handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def setLevel(self, level):
        logging.Handler.setLevel(self, level)
        self._f_handler.setLevel(level)
        self._s_handler.setLevel(level)


    def close(self):
        self._f_handler.close()
        self._s_handler.close()
        logging.Handler.close(self)

# class : MPLogger
class MPLogger(object):
    # Static variables
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __init__(self, log_dir, log_name, log_level):
        # custom logger variables
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_level = MPLogger._get_level(log_level)

        root = logging.getLogger()
        root.setLevel(self.log_level)

        logger = logging.getLogger(self.log_name)

        mpLogHandler = MPLogHandler("%s/%s.log" % (self.log_dir, self.log_name))
        mpLogHandler.setLevel(self.log_level)

        # Formatter
        formatter = logging.Formatter("%(asctime)s %(levelname)-5s - %(processName)-15s - %(filename)-22s:%(lineno)-3s - %(message)s")
        mpLogHandler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(mpLogHandler)
        logger.propagate = False

        self.logger = logger

    def getLogger(self):
        # return logging.getLogger(self.log_name)
        return self.logger

    @staticmethod
    def _get_level(level):
        if level == "DEBUG" :
            return logging.DEBUG
        elif level == "INFO" :
            return logging.INFO
        elif level == "WARN":
            return logging.WARN
        elif level == "ERROR":
            return logging.ERROR
        elif level == "CRITICAL":
            return logging.CRITICAL
        else : #기본값
            return logging.INFO

if __name__ == '__main__':
    from multiprocessing import Process, Queue
    import time
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.Session()
    logger = MPLogger(log_dir=".", log_name="test", log_level="INFO").getLogger()

    def worker_process(q):
        logger = logging.getLogger("test")
        idx = 0
        while True:
            try :
                logger.info("{} - test".format(idx))
                idx += 1
                if idx == 100:
                    idx = 0
                    raise NotImplementedError
            except Exception as e:
                logger.error(e, exc_info=True)
            time.sleep(0.1)
    q = Queue()
    workers = []
    for i in range(20):
        wp = Process(target=worker_process, name='worker %d' % (i + 1), args=(q,))
        workers.append(wp)
        wp.start()
    for wp in workers:
        wp.join()
