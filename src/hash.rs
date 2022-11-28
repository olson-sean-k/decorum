//! Hashing of IEEE 754 floating-point values.
//!
//! This module provides hashing for primitive floating-point values. Given the set of zero
//! representations $Z$ and set of `NaN` representations $N$, hashing coalesces their
//! representations such that:
//!
//! $$
//! \begin{aligned}
//! h(a)=h(b)&\mid a\in{Z},~b\in{Z}\cr\[1em\]
//! h(a)=h(b)&\mid a\in{N},~b\in{N}
//! \end{aligned}
//! $$
//!
//! The [`FloatHash`] trait agrees with the ordering and equivalence relations of the [`FloatOrd`]
//! and [`FloatEq`] traits.
//!
//! [`FloatEq`]: crate::cmp::FloatEq
//! [`FloatHash`]: crate::hash::FloatHash
//! [`FloatOrd`]: crate::cmp::FloatOrd

use core::hash::{Hash, Hasher};

use crate::{Float, Primitive, ToCanonicalBits};

/// An IEEE 754 encoded type that can be hashed.
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
