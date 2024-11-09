# Developer Notes

## Verbosity vs. Complexity

In case of doubt, prefer verbosity over complexity.
For example, prefer some duplication over things like macros, declarative code, or DSLs.
The main aim is to keep things easy to understand; for both humans as well as tooling.
I prefer "boilerplate" code over complex things like macros.
Especially with LLMs built into the editor, it's still reasonably easy to refactor boilerplate code.
This can't be said for macros; it's usually very hard to refactor them.

For example, in my experience, declarative code is a beautiful idea, but in practice it's often hard to learn and understand.

## `Arc<RwLock<T>>`

This project uses `Arc<RwLock<T>>` for pretty much everything.
The reason is that the intermediate representation (IR) is essentially a tree with nodes.
This tree is traversed in multiple directions (up, down, and sideways), so we need to store pointers all over the place.
Also, this project uses `Arc` instead of `Rc` because at some point the compiler will be multi-threaded.

To simplify the usage, there are some helpers functions available via `GuardedOp`, `GuardedBlock`, etc.

In some cases, this can simplify code such as
```rust
let x = x.read().unwrap();
let parent = x.parent();
```
to
```rust
let x = x.parent();
```

## LLVM's codebase

This project does not include any LLVM code.
The LLVM codebase is great, but also big.
Including the C files would require lots of compilation.
Calling out to an installed LLVM version is better but is hard with package versions and such.
Let's for now just generate LLVM IR, then the LLVM binary can be used to compile that.
At a later point, the plan is to include the LLVM backend in a separate crate.

## `Operation`

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