import asyncio
import logging

try:
    from redis import Redis
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
        TimeoutError as RedisTimeoutError,
    )
except ImportError:
    Redis = None
    RedisConnectionError = RedisTimeoutError = ()


class AsyncRedisPipe:
    """
    A utility class for executing Redis commands concurrently and efficiently using pipelines in the backend.

    This class is designed for high-loaded applications that rely on Redis non-pipeline calls. By using pipelines to batch together multiple Redis commands, it can achieve significantly higher throughput than traditional non-pipeline Redis calls.

    Example:
    >>> redis = Redis()
    >>> pipe = AsyncRedisPipe(redis)
    >>> await pipe.set("key", "value")
    >>> await pipe.get("key")
    >>> redis_calls = [pipe.get("key") for i in range(100000)]
    >>> await asyncio.gather(*redis_calls) # 100K Concurrent redis calls

    Disclaimer:
        This class is intended to be used with basic redis commands, other non redis commands (a.k.a client side commands) are not supported (eg. `redis.lock`) and such call can have unexpected behavior.
    """

    def __init__(
        self,
        redis: Redis,
        pipe_max_size: int = 200,
        workers: int = 8,
        loop: asyncio.AbstractEventLoop = None,
        debug: bool = False,
    ) -> None:
        """
        Parameters:
            redis (``Redis``):
                The Redis client

            pipe_max_size (``int``, *optional*):
                The maximum number of commands in a single pipeline. Defaults to ``200``.

            workers (``int``, *optional*):
                The number of workers processing the pipeline. Defaults to ``8``.

            loop (:py:class:`asyncio.AbstractEventLoop`, *optional*):
                The event loop to use. Defaults to :py:class:`asyncio.get_event_loop()`.

            debug (``bool``, *optional*):
                Whether to enable debug logging. Defaults to ``False``.
        """

        if not Redis:
            raise RuntimeError("Redis is not installed. Try pip install redis")

        self.redis = redis
        self.pipe_max_size = pipe_max_size
        self.debug = debug
        self.logger = logging.getLogger(__name__)

        self.__queue = asyncio.Queue()
        self.__loop = (
            asyncio.get_event_loop()
            if not isinstance(loop, asyncio.AbstractEventLoop)
            else loop
        )
        self.is_running = True
        self.__worker_tasks = [
            self.__loop.create_task(self.__worker()) for _ in range(workers)
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return

    async def __worker(self):
        while self.is_running:
            batch = []
            pipe = self.redis.pipeline(False)

            while len(batch) <= self.pipe_max_size:
                try:
                    if not batch:
                        data = await self.__queue.get()
                    else:
                        data = self.__queue.get_nowait()
                        await asyncio.sleep(0)
                except asyncio.QueueEmpty:
                    break
                else:
                    if self.debug:
                        self.logger.debug(f"Executing command: {data[1]}")

                    try:
                        getattr(pipe, data[1])(*data[2], **data[3])
                    except Exception as e:
                        data[0].set_exception(e)
                        continue

                    batch.append(data)

            if batch:
                await self.__handle_pipe_line(batch, pipe)

    async def __handle_pipe_line(self, batch: list, pipe):
        futures = [data[0] for data in batch]
        results = None
        delay = 0.5

        while True:
            try:
                results = await pipe.execute(False)
            except (RedisConnectionError, RedisTimeoutError) as e:
                # Transient failure (e.g. BusyLoadingError while Redis is
                # loading its dataset after a restart, or a network blip):
                # rebuild the pipeline and retry the same batch instead of
                # letting the worker die and leaving callers hanging forever.
                # pipe.execute() resets its command stack in a finally, so the
                # batch must be replayed onto a fresh pipeline.
                results = None
                if not self.is_running:
                    break
                self.logger.warning(
                    f"Redis pipeline failed ({e.__class__.__name__}), "
                    f"retrying {len(batch)} command(s) in {delay}s"
                )
                await asyncio.sleep(delay)
                if not self.is_running:
                    break
                delay = min(delay * 2, 10)
                pipe = self.redis.pipeline(False)
                for _, name, args, kwargs in batch:
                    getattr(pipe, name)(*args, **kwargs)
            except Exception as e:
                self.__fail_batch(futures, e)
                return
            else:
                break

        if results is None:
            self.__fail_batch(
                futures, RuntimeError("AsyncRedisPipe is closed")
            )
            return

        for x, result in enumerate(results):
            if self.debug:
                self.logger.debug(f"Result: {result}")

            if isinstance(result, BaseException):
                futures[x].set_exception(result)
            else:
                futures[x].set_result(result)

    def __fail_batch(self, futures: list, e: Exception):
        for fut in futures:
            if not fut.done():
                fut.set_exception(e)

    def __getattr__(self, name):
        wrapper = self.__dict__.get(name)
        if wrapper is None:
            def wrapper(*args, **kwargs):
                fut = asyncio.Future()
                self.__queue.put_nowait((fut, name, args, kwargs))
                return fut

            # cache in the instance dict so later calls skip __getattr__
            # and the per-call closure allocation
            self.__dict__[name] = wrapper
        return wrapper

    def close(self):
        """Close the AsyncRedisPipe instance"""
        self.is_running = False
        for worker in self.__worker_tasks:
            worker.cancel()
        while not self.__queue.empty():
            try:
                fut, *_ = self.__queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not fut.done():
                fut.set_exception(RuntimeError("AsyncRedisPipe is closed"))
