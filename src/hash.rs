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
//! The [`CanonicalHash`] trait agrees with the ordering and equivalence relations of the
//! [`CanonicalOrd`] and [`CanonicalEq`] traits.
//!
//! [`CanonicalEq`]: crate::cmp::CanonicalEq
//! [`CanonicalHash`]: crate::hash::CanonicalHash
//! [`CanonicalOrd`]: crate::cmp::CanonicalOrd

use core::hash::{Hash, Hasher};

use crate::ToCanonicalBits;

pub trait CanonicalHash {
    fn hash_canonical_bits<H>(&self, state: &mut H)
    where
        H: Hasher;
}

// TODO: This implementation conflicts with implementations over references to a type `T` where
//       `T: CanonicalHash`. However, this is because `rustc` claims that "sealed" traits can be
//       implemented downstream, which isn't true. Write reference implementations when possible
//       (or consider removing this blanket implementation).
impl<T> CanonicalHash for T
where
    T: ToCanonicalBits,
{
    fn hash_canonical_bits<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.to_canonical_bits().hash(state)
    }
}

impl<T> CanonicalHash for [T]
where
    T: CanonicalHash,
{
    fn hash_canonical_bits<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for item in self {
            item.hash_canonical_bits(state);
        }
    }
}
