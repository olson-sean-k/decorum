//! Hashing.
//!
//! This module provides hashing for primitive floating-point values. Given the
//! set of zero representations $Z$ and set of `NaN` representations $N$,
//! hashing coalesces their representations such that:
//!
//! $$
//! \begin{aligned}
//! h(a)=h(b)&\mid a\in{Z},~b\in{Z}\cr\[1em\]
//! h(a)=h(b)&\mid a\in{N},~b\in{N}
//! \end{aligned}
//! $$
//!
//! The `FloatHash` trait agrees with the ordering and equivalence relations of
//! the `FloatOrd` and `FloatEq` traits.

use core::hash::{Hash, Hasher};

use crate::canonical::ToCanonicalBits;
use crate::primitive::Primitive;
use crate::Float;

/// Hashing for primitive floating-point values.
pub trait FloatHash {
    fn float_hash<H>(&self, state: &mut H)
    where
        H: Hasher;
}

impl<T> FloatHash for T
where
    T: Float + Primitive,
{
    fn float_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.to_canonical_bits().hash(state);
    }
}

impl<T> FloatHash for [T]
where
    T: Float + Primitive,
{
    fn float_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for item in self.iter() {
            item.float_hash(state);
        }
    }
}
