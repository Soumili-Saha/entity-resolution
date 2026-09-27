"""Windows only: opt a process out of power throttling (EcoQoS) and raise its priority, so long
unattended runs keep using all cores when the laptop is locked / the console is in the background.
No admin rights needed for processes of the same user.

Usage: python -m entity_matcher.no_throttle <pid> [<pid> ...]      (or call disable_throttling() in-process)
"""
import ctypes
import sys
from ctypes import wintypes

PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
EXECUTION_SPEED = 0x1
ABOVE_NORMAL_PRIORITY_CLASS = 0x8000


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


def disable_throttling(pid=None):
    """Turn off EcoQoS execution-speed throttling and set above-normal priority for pid (default: self)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # declare 64-bit handle types: without them ctypes truncates HANDLE values to 32 bits
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    if pid is None:
        h = k32.GetCurrentProcess()
    else:
        h = k32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            raise OSError(ctypes.get_last_error(), f"OpenProcess failed for pid {pid}")
    state = _PowerThrottlingState(1, EXECUTION_SPEED, 0)  # control speed, state 0 = not throttled
    ok1 = k32.SetProcessInformation(h, PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state))
    ok2 = k32.SetPriorityClass(h, ABOVE_NORMAL_PRIORITY_CLASS)
    if pid is not None:
        k32.CloseHandle(h)
    return bool(ok1), bool(ok2)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, "throttling off / priority raised:", disable_throttling(p))
