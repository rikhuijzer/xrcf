# xrcf

The problem that this project aims to solve is that it should be easy to target multiple backends (like GPUs, TPUs, and CPUs) with a single language.
It should be possible for developers to write code in a high-level syntax,
and for this code to automatically run on various modern backends.

For example, in C++, the code

```cpp
float a[4] = {8.0f, 7.0f, 6.0f, 5.0f};
float b[4] = {16.0f, 15.0f, 14.0f, 13.0f};
float c[4];

for (int i = 0; i < 4; i++) {
    c[i] = a[i] + b[i];
}
```

can be sped up by using AVX instructions.
One way is to manually call the intrinsic:

```cpp
__m256 a = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f);
__m256 b = _mm256_set_ps(16.0f, 15.0f, 14.0f, 13.0f);
__m256 result = _mm256_add_ps(a, b);
```

But this is not very convenient.
What we want is to write the code in a high-level syntax,

```cpp
float a[4] = {8.0f, 7.0f, 6.0f, 5.0f};
float b[4] = {16.0f, 15.0f, 14.0f, 13.0f};
float c[4] = a + b;
```

The compiler should translate this to use the AVX instructions where possible.
To do this, it is important for the compiler to handle the full stack.
A problem with LLVM is that it sometimes receives code that is already too low-level.
At that point, it is not always possible to generate efficient code since some
information was lost during the lowering.

The aim of this project is to create a compiler framework that makes it easy to
build compilers that can run on various backends.
For example, a GPU, TPU, or CPU backend.
The long-term aim is to have these backends available as separate crates.
Compared to MLIR, this project is written in Rust which makes it easier to
combine multiple projects into one.

## Notes

### Name

The meaning of "xrcf" is "xr compiler framework".
What the x and r mean is undefined.
The r could be "rust", "rewrite", "recursive", or something else.
The x could be anything.

### LLVM's codebase

Let's not include any LLVM code.
LLVM is great, but it's also big.
Including the C files would require lots of compilation.
Calling out to an installed LLVM version is better but is hard with package versions and such.
Let's for now just generate LLVM IR and let the client compile it themselves.
At a later point, the plan is to move the LLVM backend to a separate crate.

### `Operation`

MLIR has a `Op` trait where each `struct` that implements it contains a `Operation` field.
This means that `Operation` is very generic and the various `Op` implementations
all access the real data through the `Operation` field.

A downside of the `Operation` field is that it may contain fields that are not necessary.
For example, `arith.constant` does not take any operands,
but the `Operation` field will still contain an empty `operands` field.

The benefit is that transformations do not require the fields to be copied.
They can just call it a different type while pointing to the same `Operation` struct.
This is probably why MLIR uses it.
Otherwise, transforming all ops from one dialect to another would require copying all op fields.

But wait, it's not that much copying.
Many fields are just pointers to other data.
Instead focus on the fact that any `Op` transformation takes ownership of the data.
Then, it's will be harder to mess up the underlying data.
There will be less state to keep track of.

In summary: Do not prematurely optimize!
