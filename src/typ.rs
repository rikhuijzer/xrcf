trait Type {}

/// Represents an integer type such as i32 or i64.
/// This does not include the sign bit like in LLVM since
/// it doesn't matter for 2s complement integer arithmetic.
pub struct IntegerType {
    num_bits: u64,
}

impl Type for IntegerType {}

/// Arbitrary precision integer.
pub struct APInt {
    num_bits: u64,
    value: u64,
    is_signed: bool,
}
