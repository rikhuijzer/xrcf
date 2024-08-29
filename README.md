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

The biggest downside is that it's another layer of indirection.
Developers have to go through the `Operation` field to get to the data.
Let's try to go without this for now and see how it goes.