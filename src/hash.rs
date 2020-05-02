//! Hashing.
//!
//! This module provides hashing for primitive floating-point values. Given the
//! set of zero representations $Z$ and set of `NaN` representations $N$,
//! hashing coalesces their representations such that
//! $h(a)=h(b)|a\in{Z},b\in{Z}$ and $h(a)=h(b)|a\in{N},b\in{N}$.
//!
//! The `FloatHash` trait agrees with the ordering and equivalence relations of
//! the `FloatOrd` and `FloatEq` traits.

use core::hash::{Hash, Hasher};

use crate::canonical::ToCanonicalBits;
use crate::primitive::Primitive;
use crate::{Encoding, Nan};

/// Hashing for primitive floating-point values.
pub trait FloatHash {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher;
}

impl<T> FloatHash for T
where
    T: Encoding + Nan + Primitive,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.to_canonical_bits().hash(state);
    }
}

impl<T> FloatHash for [T]
where
    T: Encoding + Nan + Primitive,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for item in self.iter() {
            item.hash(state);
        }
    }
}
