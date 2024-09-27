use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;

trait Type {}

/// Represents an integer type such as i32 or i64.
/// This does not include the sign bit like in LLVM since
/// it doesn't matter for 2s complement integer arithmetic.
pub struct IntegerType {
    num_bits: u64,
}

impl IntegerType {
    pub fn new(num_bits: u64) -> Self {
        Self { num_bits }
    }
}

impl Type for IntegerType {}

impl Display for IntegerType {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "i{}", self.num_bits)
    }
}

/// Arbitrary precision integer.
pub struct APInt {
    num_bits: u64,
    value: u64,
    is_signed: bool,
}

impl APInt {
    pub fn new(num_bits: u64, value: u64, is_signed: bool) -> Self {
        Self {
            num_bits,
            value,
            is_signed,
        }
    }
}

impl Display for APInt {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.value)
    }
}
