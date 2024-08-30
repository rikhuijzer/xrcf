# rrcf

RRCF Compiler Framework

## But you don't know anything about compilers?

Yet.

## Notes

### LLVM's codebase

Let's not include any LLVM code like Zig is moving towards.
LLVM is great, but it's also big.
Including the C files would require lots of compilation.
Calling out to an installed LLVM version is better but is hard with package versions and such.
Let's for now just generate LLVM IR and let the client compile it themselves.

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