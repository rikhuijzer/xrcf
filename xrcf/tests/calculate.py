from ctypes import CDLL, c_int, Structure, c_int16, POINTER, byref
import os
import sys
from timeit import timeit

length = 16


class Vec8x16(Structure):
    _fields_ = [("values", c_int16 * length)]

    def __init__(self, value=0):
        super().__init__()
        for i in range(length):
            self.values[i] = value

    def __repr__(self):
        return f"Vec{length}x16({list(self.values)})"


def ll():
    return (
        """
    define i32 @calculate(i32 %a, i32 %b) #0 {
    %1 = mul i32 %a, %b
    %2 = add i32 %1, 43
    ret i32 %2
    }
    """
        + f"""
    define void @calc_vec(<{length} x i16>* %o, <{length} x i16>* %a, <{length} x i16>* %b) #0 {{
    %1 = load <{length} x i16>, <{length} x i16>* %a, align 16
    %2 = load <{length} x i16>, <{length} x i16>* %b, align 16
    %3 = add <{length} x i16> %1, %2
    store <{length} x i16> %3, <{length} x i16>* %o, align 16
    ret void
    }}

    attributes #0 = {{ nounwind "target-features"="neon" }}
    """
    )


def load_lib():
    res = os.system(f"echo '{ll()}' | clang -x ir - -shared -o tmp.so")
    if res != 0:
        sys.exit(res)

    lib = CDLL("./tmp.so")
    lib.calculate.argtypes = [c_int, c_int]
    lib.calculate.restype = c_int
    lib.calc_vec.argtypes = [POINTER(Vec8x16), POINTER(Vec8x16), POINTER(Vec8x16)]
    lib.calc_vec.restype = None
    return lib


lib = load_lib()
result = lib.calculate(1, 2)
print(f"calculate: {result}")

number = 100_000


def calc_vec_py(o, a, b):
    for i in range(length):
        o.values[i] = a.values[i] + b.values[i]
    return o


a = Vec8x16(1)
b = Vec8x16(2)
o = Vec8x16(0)
py_secs = timeit(lambda: calc_vec_py(o, a, b), number=number)/number
print(f"calc_vec_py: {o}")
print(f"calc_vec_py: {py_secs}")

a = Vec8x16(1)
b = Vec8x16(2)
o = Vec8x16()
secs = timeit(lambda: lib.calc_vec(byref(o), byref(a), byref(b)), number=number)/number
print(f"calc_vec: {o}")
print(f"calc_vec: {secs}")

print(f"Difference: {py_secs / secs}")
