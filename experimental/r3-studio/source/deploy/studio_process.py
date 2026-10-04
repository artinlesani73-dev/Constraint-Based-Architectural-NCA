"""Windows worker-tree ownership: kill all descendants on cancellation/owner exit."""
import ctypes
from ctypes import wintypes
import os
import time


class WorkerTree:
    def __init__(self, process):
        self.process, self.handle = process, None
        if os.name != 'nt':
            return
        size = ctypes.c_size_t
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                        ('flags', wintypes.DWORD), ('min_ws', size), ('max_ws', size),
                        ('active_limit', wintypes.DWORD), ('affinity', size),
                        ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('ro','wo','oo','rb','wb','ob')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory',size),
                        ('job_memory',size), ('peak_process',size), ('peak_job',size)]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        handle = self.kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        limits = Extended(); limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error()); self.close(); raise error
        if not self.kernel.AssignProcessToJobObject(handle, process._handle):
            error = ctypes.WinError(ctypes.get_last_error()); self.close(); raise error

    def active_count(self):
        if self.handle is None:
            return int(self.process.poll() is None)
        class Accounting(ctypes.Structure):
            _fields_ = [(name,ctypes.c_longlong) for name in ('user','kernel','period_user','period_kernel')] + [
                (name,wintypes.DWORD) for name in ('faults','total','active','terminated')]
        info = Accounting()
        if not self.kernel.QueryInformationJobObject(self.handle,1,ctypes.byref(info),ctypes.sizeof(info),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.active

    def terminate(self):
        if self.handle is not None:
            if not self.kernel.TerminateJobObject(self.handle,1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic()+5
            while self.active_count():
                if time.monotonic()>deadline:
                    raise RuntimeError('Worker tree did not stop; cancellation is not acknowledged')
                time.sleep(.01)
        elif self.process.poll() is None:
            self.process.terminate()

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
