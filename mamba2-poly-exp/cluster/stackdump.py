"""Run a script with a Python stack dump to stderr every N seconds.

    python cluster/stackdump.py 300 norm/collect_norm_stats.py --args ...

For jobs that hang on a GPU node we cannot log in to: the dumps land in the
job's .err log and show exactly where it is stuck. A CUDA init probe runs first,
timed, so a hung driver shows up before any of our code does.
"""

import faulthandler
import runpy
import sys
import time

every, script, rest = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
faulthandler.dump_traceback_later(every, repeat=True)

t0 = time.time()
print(f"[stackdump] probing CUDA init, dumps every {every}s", flush=True)
import torch  # noqa: E402

print(f"[stackdump] import torch {time.time() - t0:.1f}s", flush=True)
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    print(f"[stackdump] cuda init {time.time() - t0:.1f}s on "
          f"{torch.cuda.get_device_name(0)}", flush=True)

sys.argv = [script] + rest
runpy.run_path(script, run_name="__main__")
