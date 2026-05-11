import time, contextlib, torch

ENABLE_TIMERS = True  # flip off when done
ENABLE_NVTX   = torch.cuda.is_available()

@contextlib.contextmanager
def section(name: str, sink=None):
    if ENABLE_NVTX:
        torch.cuda.nvtx.range_push(name)
    if ENABLE_TIMERS:
        # torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
    with torch.profiler.record_function(name):
        yield
    if ENABLE_TIMERS:
        # torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = (time.perf_counter() - t0) * 1000.0
        if sink is not None:
            sink[name] = sink.get(name, 0.0) + dt
        else:
            print(f"[{name}] {dt:.2f} ms")
    if ENABLE_NVTX:
        torch.cuda.nvtx.range_pop()

def log_trace(p):
    p.export_chrome_trace("/workspace/log/trace_" + str(p.step_num) + ".json")
