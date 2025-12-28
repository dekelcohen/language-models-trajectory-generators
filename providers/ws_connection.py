import json
import threading
import time
from queue import Queue, Empty


class WsJSONConnection:
    """
    Minimal sync wrapper around a WebSocket connection that mirrors
    the Pipe/Subprocess connection API used in this repo.

    - send([CMD, args]) marshals to {"cmd": CMD, "args": args}
    - recv(timeout=None) blocks for next JSON message
    - poll() checks if any message is queued

    Requires the 'websockets' package.
    """

    def __init__(self, url: str, timeout: float | None = 15.0):
        try:
            import asyncio  # noqa: F401
            import websockets  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "WebSocket transport requires the 'websockets' package.\n"
                "Install with: pip install websockets"
            ) from e

        self._url = url
        self._rx_queue: Queue = Queue()
        self._loop = None
        self._ws = None
        # Single override applied to connect/send/close. None => defaults per op; <0 => infinite
        self._timeout_override = timeout
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for connection to establish
        start = time.time()
        # Effective timeout for connect: default 15.0 unless overridden; negative => infinite
        connect_to = None
        if self._timeout_override is None:
            connect_to = 15.0
        elif isinstance(self._timeout_override, (int, float)) and self._timeout_override < 0:
            connect_to = None
        else:
            connect_to = float(self._timeout_override)
        deadline = None if connect_to is None else (start + connect_to)
        while True:
            if self._ws is not None:
                return
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(0.01)
        raise TimeoutError(f"Timed out connecting to WebSocket at {url}")

    def _run_loop(self):
        import asyncio
        import websockets

        async def _connect_and_listen():
            self._ws = await websockets.connect(self._url, max_size=None)

            async def _reader():
                try:
                    async for msg in self._ws:
                        try:
                            obj = json.loads(msg)
                        except Exception:
                            obj = msg
                        self._rx_queue.put(obj)
                finally:
                    # Signal closed connection by putting None
                    try:
                        self._rx_queue.put(None)
                    except Exception:
                        pass

            await _reader()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(_connect_and_listen())
        except Exception:
            # Ensure a closed state if connection fails
            try:
                self._rx_queue.put(None)
            except Exception:
                pass
        finally:
            try:
                if self._loop.is_running():
                    self._loop.stop()
            except Exception:
                pass

    def send(self, payload_list):
        if not isinstance(payload_list, list) or len(payload_list) == 0:
            raise ValueError("Expected a non-empty list payload")
        obj = {"cmd": payload_list[0]}
        if len(payload_list) > 1:
            obj["args"] = payload_list[1]

        # Schedule the send on the loop thread
        import asyncio

        async def _send(ws, data):
            await ws.send(data)

        if self._ws is None or self._loop is None:
            raise RuntimeError("WebSocket not connected")
        data = json.dumps(obj)
        fut = asyncio.run_coroutine_threadsafe(_send(self._ws, data), self._loop)
        # Effective timeout for send: default 10.0 unless overridden; negative => infinite
        if self._timeout_override is None:
            send_to = 10.0
        elif isinstance(self._timeout_override, (int, float)) and self._timeout_override < 0:
            send_to = None
        else:
            send_to = float(self._timeout_override)
        # Wait for send to complete to preserve ordering semantics
        fut.result(timeout=send_to)

    def recv(self, timeout=None):
        try:
            msg = self._rx_queue.get(block=True, timeout=timeout)
        except Empty:
            raise TimeoutError("Timed out waiting for WebSocket message")
        # Treat None as closed connection
        if msg is None:
            raise ConnectionError("WebSocket connection closed")
        return msg

    def poll(self):
        try:
            _ = self._rx_queue.get_nowait()
            self._rx_queue.put(_)  # put it back
            return True
        except Empty:
            return False

    def close(self):
        try:
            import asyncio

            async def _close(ws):
                try:
                    await ws.close()
                except Exception:
                    pass

            if self._ws and self._loop:
                fut = asyncio.run_coroutine_threadsafe(_close(self._ws), self._loop)
                try:
                    # Effective timeout for close: default 5.0 unless overridden; negative => infinite
                    if self._timeout_override is None:
                        close_to = 5.0
                    elif isinstance(self._timeout_override, (int, float)) and self._timeout_override < 0:
                        close_to = None
                    else:
                        close_to = float(self._timeout_override)
                    fut.result(timeout=close_to)
                except Exception:
                    pass
        finally:
            try:
                if self._thread:
                    # Let the loop thread exit on its own
                    self._thread.join(timeout=1.0)
            except Exception:
                pass
