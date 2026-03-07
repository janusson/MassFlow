import sys
from matchms.importing.load_from_msp import parse_msp_file

def trace(frame, event, arg):
    if event == 'exception':
        if isinstance(arg, tuple) and len(arg) > 1 and 'num peaks' in str(arg[1]):
            print("CRASH LOCAL VARS:")
            for k, v in frame.f_locals.items():
                print(f"{k}: {v}")
    return trace

sys.settrace(trace)

for _ in parse_msp_file("data/reference/example_library.msp"):
    pass
