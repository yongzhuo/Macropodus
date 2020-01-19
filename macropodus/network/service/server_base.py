# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/7/22 22:11
# @author  : Meteorix
# @function: service_streamer of dl
# @codefrom: https://github.com/ShannonAI/service-streamer


from queue import Queue, Empty
import multiprocessing as mp
from typing import List
import threading
import weakref
import uuid
import time
import os


class ManagedModel(object):
    def __init__(self, pars=None):
        self.model_pars = pars  # 预测模型实例化参数
        self.model = None  # 实例化后的模型，需要在init_model方法里初始化

    def set_gpu_id(self, gpu_id):
        self._set_gpu_id(gpu_id)

    @staticmethod
    def _set_gpu_id(gpu_id=None):
        if gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def init_model(self):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError


class Future(object):
    def __init__(self, task_id, task_size, future_cache_ref):
        self._id = task_id
        self._size = task_size
        self._future_cache_ref = future_cache_ref
        self._outputs = []
        self._finish_event = threading.Event()

    def result(self, timeout=None):
        if self._size == 0:
            self._finish_event.set()
            return []
        finished = self._finish_event.wait(timeout)

        if not finished:
            raise TimeoutError("Task: %d Timeout" % self._id)

        # remove from future_cache
        future_cache = self._future_cache_ref()
        if future_cache is not None:
            del future_cache[self._id]

        # [(request_id, output), ...] sorted by request_id
        self._outputs.sort(key=lambda i: i[0])
        # restore batch result from outputs
        batch_result = [i[1] for i in self._outputs]

        return batch_result

    def done(self):
        if self._finish_event.is_set():
            return True

    def _append_result(self, it_id, it_output):
        self._outputs.append((it_id, it_output))
        if len(self._outputs) >= self._size:
            self._finish_event.set()
            # print("%d task_id:%d size:%d finished" % (os.getpid(), self._id, self._size))


class _FutureCache(dict):
    "Dict for weakref only"
    pass


class _BaseStreamer(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._client_id = str(uuid.uuid4())
        self._task_id = 0
        self._future_cache = _FutureCache()  # {task_id: future}

        self.back_thread = threading.Thread(target=self._loop_collect_result, name="thread_collect_result")
        self.back_thread.daemon = True

    def _delay_setup(self):
        self.back_thread.start()

    def _send_request(self, task_id, request_id, model_input):
        raise NotImplementedError

    def _recv_response(self, timeout=1):
        raise NotImplementedError

    def _input(self, batch: List) -> int:
        """
        input a batch, distribute each item to mq, return task_id
        """
        # task id in one client
        task_id = self._task_id
        self._task_id += 1
        # request id in one task
        request_id = 0

        future = Future(task_id, len(batch), weakref.ref(self._future_cache))
        self._future_cache[task_id] = future

        for model_input in batch:
            # print(f"sending_request: PID:{os.getpid()}  task_id:{task_id}  request_id: {request_id}")
            self._send_request(task_id, request_id, model_input)
            request_id += 1

        return task_id

    def _loop_collect_result(self):
        print(self, "start _loop_collect_result")
        while True:
            message = self._recv_response(timeout=1)
            if message:
                (task_id, request_id, item) = message
                future = self._future_cache[task_id]
                future._append_result(request_id, item)
            else:
                # todo
                time.sleep(0.001)

    def _output(self, task_id: int) -> List:
        future = self._future_cache[task_id]
        batch_result = future.result(20)  # 20s timeout for any requests
        return batch_result

    def submit(self, batch):
        task_id = self._input(batch)
        future = self._future_cache[task_id]
        return future

    def predict(self, batch):
        task_id = self._input(batch)
        ret = self._output(task_id)
        return ret


class _BaseStreamWorker(object):
    def __init__(self, predict_function, batch_size, max_latency, *args, **kwargs):
        super().__init__()
        # assert callable(predict_function)
        self._pid = os.getpid()
        self._predict = predict_function
        self._batch_size = batch_size
        self._max_latency = max_latency

    def run_forever(self):
        self._pid = os.getpid()  # overwrite the pid
        print("[gpu worker %d] %s start working" % (self._pid, self))

        while True:
            handled = self._run_once()
            if not handled:
                # sleep if no data handled last time
                time.sleep(0.001)

    def model_predict(self, batch_input):
        # fairseq (gpu)
        batch_result: List[str] = self._predict(batch_input)
        # batch_result = self._predict(batch_input)
        return batch_result

    def _run_once(self):
        batch = []
        start_time = time.time()
        for i in range(self._batch_size):
            try:
                item = self._recv_request(timeout=self._max_latency)
            except TimeoutError:
                # each item timeout exceed the max latency
                break
            else:
                batch.append(item)
            if (time.time() - start_time) > self._max_latency:
                # total batch time exceeds the max latency
                break
        if not batch:
            return 0

        model_inputs = [i[3] for i in batch]
        model_outputs = self.model_predict(model_inputs)

        # publish results to redis
        for i, item in enumerate(batch):
            client_id, task_id, request_id, _ = item
            self._send_response(client_id, task_id, request_id, model_outputs[i])

        batch_size = len(batch)
        print("[gpu worker %d] run_once batch_size: %d start_at: %s spend: %s" % (
        self._pid, batch_size, start_time, time.time() - start_time))
        return batch_size

    def _recv_request(self, timeout=1):
        raise NotImplementedError

    def _send_response(self, client_id, task_id, request_id, model_input):
        raise NotImplementedError


class ThreadedStreamer(_BaseStreamer):
    def __init__(self, predict_function, batch_size, max_latency=0.1):
        super().__init__()
        self._input_queue = Queue()
        self._output_queue = Queue()
        self._worker = ThreadedWorker(predict_function, batch_size, max_latency, self._input_queue, self._output_queue)
        self._worker_thread = threading.Thread(target=self._worker.run_forever, name="thread_worker")
        self._worker_thread.daemon = True
        self._worker_thread.start()
        self._delay_setup()

    def _send_request(self, task_id, request_id, model_input):
        self._input_queue.put((0, task_id, request_id, model_input))

    def _recv_response(self, timeout=1):
        try:
            message = self._output_queue.get(timeout=1)
        except Empty:
            message = None
        return message


class ThreadedWorker(_BaseStreamWorker):
    def __init__(self, predict_function, batch_size, max_latency, request_queue, response_queue):
        super().__init__(predict_function, batch_size, max_latency)
        self._request_queue = request_queue
        self._response_queue = response_queue

    def _recv_request(self, timeout=6):
        try:
            item = self._request_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, client_id, task_id, request_id, model_output):
        self._response_queue.put((task_id, request_id, model_output))


class Streamer(_BaseStreamer):
    def __init__(self, predict_function_or_model, batch_size, max_latency=0.1, worker_num=1, cuda_devices=None):
        import multiprocessing as mp
        super().__init__()
        self.worker_num = worker_num
        self.cuda_devices = cuda_devices
        self._input_queue = mp.Queue()
        self._output_queue = mp.Queue()
        self._worker = StreamWorker(predict_function_or_model, batch_size, max_latency, self._input_queue,
                                    self._output_queue)
        self._worker_ps = []
        self._worker_ready_events = []
        self._setup_gpu_worker()
        self._delay_setup()

    def _setup_gpu_worker(self):
        for i in range(self.worker_num):
            e = mp.Event()
            if self.cuda_devices is not None:
                gpu_id = self.cuda_devices[i % len(self.cuda_devices)]
                args = (gpu_id, e,)
            else:
                args = (None, e,)
            p = mp.Process(target=self._worker.run_forever, args=args, name="stream_worker", daemon=True)
            p.start()
            self._worker_ps.append(p)
            self._worker_ready_events.append(e)

    def _wait_for_worker_ready(self, timeout=20):
        # wait for all workers finishing init
        for (i, e) in enumerate(self._worker_ready_events):
            # todo: select all events with timeout
            is_ready = e.wait(timeout)
            print("gpu worker:%d ready state: %s" % (i, is_ready))

    def _send_request(self, task_id, request_id, model_input):
        self._input_queue.put((0, task_id, request_id, model_input))

    def _recv_response(self, timeout=1):
        try:
            message = self._output_queue.get(timeout=1)
        except Empty:
            message = None
        return message


class StreamWorker(_BaseStreamWorker):
    def __init__(self, predict_function_or_model, batch_size, max_latency, request_queue, response_queue):
        super().__init__(predict_function_or_model, batch_size, max_latency)
        self._request_queue = request_queue
        self._response_queue = response_queue

    def run_forever(self, gpu_id=None, ready_event=None):
        # if it is a managed model, lazy init model after forked & set CUDA_VISIBLE_DEVICES
        # if isinstance(self._predict, type) and issubclass(self._predict, ManagedModel):
        if isinstance(self._predict, ManagedModel):
            model_class = self._predict
            print("[gpu worker %d] init model on gpu:%s" % (os.getpid(), gpu_id))
            # self._model = model_class(gpu_id)
            model_class.set_gpu_id(gpu_id)
            self._model = model_class
            self._model.init_model()
            print("[gpu worker %d] init model on gpu:%s" % (os.getpid(), gpu_id))
            self._predict = self._model.predict
            if ready_event:
                ready_event.set()  # tell father process that init is finished
        super().run_forever()

    def _recv_request(self, timeout=1):
        try:
            item = self._request_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, client_id, task_id, request_id, model_output):
        self._response_queue.put((task_id, request_id, model_output))


