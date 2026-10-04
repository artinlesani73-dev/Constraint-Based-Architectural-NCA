"""One procedural job. Parent-pipe EOF stops this process after server death."""
import os
import sys
from threading import Thread


def main():
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(10)
    print('Waiting for durable submission acknowledgement', flush=True)
    # Install watchdog before importing torch. No computation if parent already died.
    if sys.stdin.buffer.readline() != b'GO\n':
        return 90
    print('Acknowledged; starting parent watchdog and loading runtime', flush=True)
    def parent_watchdog():
        if os.name == 'nt':
            # CRT reads hold the descriptor lock on Windows and can deadlock
            # library imports that inspect stdin. Read the pipe's native handle.
            import ctypes
            import msvcrt
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                       ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            kernel.ReadFile.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            buffer = ctypes.create_string_buffer(1)
            count = wintypes.DWORD()
            while kernel.ReadFile(handle, buffer, 1, ctypes.byref(count), None) and count.value:
                pass
        else:
            while os.read(sys.stdin.fileno(), 1024):
                pass
        os._exit(90)

    from hashlib import sha256
    from pathlib import Path
    import json
    import torch
    from deploy.studio import ROOT, encode, plan_scene
    from deploy.studio_jobs import write_once
    faulthandler.cancel_dump_traceback_later()
    print('Runtime loaded', flush=True)
    Thread(target=parent_watchdog, daemon=True).start()

    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request.json').read_bytes())
    for name, expected in request['provenance']['code_sha256'].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed after submission; refusing a mixed-version study')
    torch.set_num_threads(2)
    sequence = 0
    def progress(stage):
        nonlocal sequence
        write_once(directory / 'progress' / f'{sequence:06d}.json', {'stage': stage})
        sequence += 1
    from deploy.studio_mass_v2 import generate, replay_import, VERSION
    if request['kind'] != 'mass_result':raise ValueError('MS2 requires a mass request')
    if 'import_sha256' in request:
        payload=json.loads((directory/'import.json').read_bytes())
        if sha256(encode(payload)).hexdigest()!=request['import_sha256']:raise ValueError('Stored import changed')
        computed=replay_import(payload,progress)
    else:
        if request['version']!=VERSION:raise ValueError('Unsupported generation version')
        computed=generate(request['mass_request'],progress)
    record = {**request, **computed}
    progress('Saving candidate')
    write_once(directory / 'candidate.json', {'record': record, 'sha256': sha256(encode(record)).hexdigest()})
    return 0


if __name__ == '__main__':
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    # All persistent writes were fsynced. Do not close stdin during interpreter
    # teardown while the watchdog is blocked on the parent-lifetime pipe.
    os._exit(code)
