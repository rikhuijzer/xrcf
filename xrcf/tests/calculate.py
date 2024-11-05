from ctypes import CDLL, c_int, Structure, c_int16, POINTER, byref
import os
import sys


class Vec16x16(Structure):
    _fields_ = [("values", c_int16 * 16)]

    def __repr__(self):
        return f"Vec16x16({list(self.values)})"


ll = """
define i32 @calculate(i32 %a, i32 %b) #0 {
  %1 = mul i32 %a, %b
  %2 = add i32 %1, 43
  ret i32 %2
}

define void @calc_vec(<16 x i16>* %o, <16 x i16>* %a, <16 x i16>* %b) #0 {
  %1 = load <16 x i16>, <16 x i16>* %a, align 32
  %2 = load <16 x i16>, <16 x i16>* %b, align 32
  %3 = add <16 x i16> %1, %2
  store <16 x i16> %3, <16 x i16>* %o, align 32
  ret void
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
    lib.calc_vec.argtypes = [POINTER(Vec16x16), POINTER(Vec16x16), POINTER(Vec16x16)]
    lib.calc_vec.restype = None
    return lib


lib = load_lib()
result = lib.calculate(1, 2)
print(f"calculate: {result}")

a = Vec16x16()
b = Vec16x16()
for i in range(1, 8):
    a.values[i] = 1
    b.values[i] = 2
for i in range(8, 16):
    a.values[i] = 3
    b.values[i] = 4
o = Vec16x16()
lib.calc_vec(byref(o), byref(a), byref(b))
print(f"calc_vec: {o}")
