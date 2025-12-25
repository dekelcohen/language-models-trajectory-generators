import json
import subprocess
import threading
from queue import Queue, Empty


class SubprocessJSONConnection:
    """
    A minimal connection-like adapter that talks to a subprocess over
    newline-delimited JSON on stdin/stdout. It exposes send(list) and recv()
    to match the existing Pipe contract used by api.py.
    """

    def __init__(self, cmd, cwd=None, env=None):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,
        )
        self._rx_queue: Queue = Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                # Forward raw line if not JSON
                msg = line
            self._rx_queue.put(msg)

    def send(self, payload_list):
        """
        Accepts a list like [CMD, arg?] and sends as JSON object {"cmd": CMD, "args": arg}
        """
        if not isinstance(payload_list, list) or len(payload_list) == 0:
            raise ValueError("Expected a non-empty list payload")
        obj = {"cmd": payload_list[0]}
        if len(payload_list) > 1:
            obj["args"] = payload_list[1]
        data = json.dumps(obj) + "\n"
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def recv(self, timeout=None):
        msg = self._rx_queue.get(block=True, timeout=timeout)
        # The server returns lists to match existing env.py semantics
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
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
