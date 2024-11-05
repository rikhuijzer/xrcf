from ctypes import CDLL, c_int
import os
import sys

ll = """
define i32 @calculate(i32 %a, i32 %b) #0 {
    %1 = mul i32 %a, %b
    %2 = add i32 %1, 43
    ret i32 %2
}

attributes #0 = { nounwind }
"""

def load_lib():
    res = os.system(f"echo '{ll}' | clang -x ir - -shared -o tmp.so")
    if res != 0:
        sys.exit(res)

    lib = CDLL("./tmp.so")
    lib.calculate.argtypes = [c_int, c_int]
    lib.calculate.restype = c_int
    return lib


lib = load_lib()
result = lib.calculate(5, 3)
print(f"Result: {result}")
