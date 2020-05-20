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
    // The name `hash` is ambiguous. `float_hash` is more clear and is
    // consistent with the `FloatEq` and `FloatOrd` traits.
    #[deprecated(since = "0.3.1", note = "use `FloatHash::float_hash` instead")]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher;

    fn float_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        // For compatibility, `float_hash` provides a default implementation.
        // Eventually, `hash` will be removed along with this default
        // implementation.
        #[allow(deprecated)]
        self.hash(state)
    }
}

impl<T> FloatHash for T
where
    T: Float + Primitive,
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
    T: Float + Primitive,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for item in self.iter() {
            item.float_hash(state);
        }
    }
}
