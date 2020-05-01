//! Canonicalization of floating-point values.
//!
//! This module provides canonicalization of floating-point values, converting
//! `NaN` and zero to the canonical forms `CNaN` and `C0` for the following
//! total ordering: `[-INF | ... | C0 | ... | INF | CNaN ]`.
//!
//! This form is used for hashing and comparisons. Functions are provided that
//! operate on primitive floating-point values which can be used by user code
//! and are also used internally by Decorum.

use core::cmp::Ordering;

use crate::canonical::ToCanonicalBits;
use crate::constraint::Constraint;
use crate::primitive::Primitive;
use crate::proxy::ConstrainedFloat;
use crate::{Encoding, Nan};

pub trait FloatEq {
    fn eq(&self, other: &Self) -> bool;
}

impl<T> FloatEq for T
where
    T: Encoding + Nan + Primitive,
{
    fn eq(&self, other: &Self) -> bool {
        self.to_canonical_bits() == other.to_canonical_bits()
    }
}

impl<T> FloatEq for [T]
where
    T: Encoding + Nan + Primitive,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() == other.len() {
            self.iter()
                .zip(other.iter())
                .all(|(a, b)| FloatEq::eq(a, b))
        }
        else {
            false
        }
    }
}

pub trait FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering;
}

impl<T> FloatOrd for T
where
    T: Nan + Primitive,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.partial_cmp(&other) {
            Some(ordering) => ordering,
            None => {
                if self.is_nan() {
                    if other.is_nan() {
                        Ordering::Equal
                    }
                    else {
                        Ordering::Greater
                    }
                }
                else {
                    Ordering::Less
                }
            }
        }
    }
}

impl<T> FloatOrd for [T]
where
    T: Nan + Primitive,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| FloatOrd::cmp(a, b))
            .find(|ordering| *ordering != Ordering::Equal)
        {
            Some(ordering) => ordering,
            None => self.len().cmp(&other.len()),
        }
    }
}

pub trait IntrinsicOrd: Copy + PartialOrd + Sized {
    fn is_undefined(&self) -> bool;

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self);

    fn min_or_undefined(&self, other: &Self) -> Self {
        self.min_max_or_undefined(other).0
    }

    fn max_or_undefined(&self, other: &Self) -> Self {
        self.min_max_or_undefined(other).1
    }
}
macro_rules! impl_intrinsic_ord {
    (no_nan_total => $t:ty) => {
        impl IntrinsicOrd for $t {
            fn is_undefined(&self) -> bool {
                false
            }

            fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
                match self.partial_cmp(other) {
                    Some(ordering) => match ordering {
                        Ordering::Less | Ordering::Equal => (*self, *other),
                        _ => (*other, *self),
                    },
                    _ => unreachable!(),
                }
            }
        }
    };
    (nan => $t:ty) => {
        impl IntrinsicOrd for $t {
            fn is_undefined(&self) -> bool {
                self.is_nan()
            }

            fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
                match self.partial_cmp(other) {
                    Some(ordering) => match ordering {
                        Ordering::Less | Ordering::Equal => (*self, *other),
                        _ => (*other, *self),
                    },
                    _ => (Nan::NAN, Nan::NAN),
                }
            }
        }
    };
}
impl_intrinsic_ord!(no_nan_total => isize);
impl_intrinsic_ord!(no_nan_total => i8);
impl_intrinsic_ord!(no_nan_total => i16);
impl_intrinsic_ord!(no_nan_total => i32);
impl_intrinsic_ord!(no_nan_total => i64);
impl_intrinsic_ord!(no_nan_total => i128);
impl_intrinsic_ord!(no_nan_total => usize);
impl_intrinsic_ord!(no_nan_total => u8);
impl_intrinsic_ord!(no_nan_total => u16);
impl_intrinsic_ord!(no_nan_total => u32);
impl_intrinsic_ord!(no_nan_total => u64);
impl_intrinsic_ord!(no_nan_total => u128);
impl_intrinsic_ord!(nan => f32);
impl_intrinsic_ord!(nan => f64);

// Note that it is not necessary for `NaN` to be a member of the constraint.
// This implementation explicitly detects `NaN`s and emits `NaN` as the
// maximum and minimum (it does not use `FloatOrd`).
impl<T, P> IntrinsicOrd for ConstrainedFloat<T, P>
where
    T: Encoding + IntrinsicOrd + Nan + Primitive,
    P: Constraint<T>,
{
    fn is_undefined(&self) -> bool {
        self.into_inner().is_nan()
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        // This function operates on primitive floating-point values. This
        // avoids the need for implementations for each combination of proxy and
        // constraint (proxy types do not always implement `Nan`, but primitive
        // types do).
        let a = self.into_inner();
        let b = other.into_inner();
        let (min, max) = a.min_max_or_undefined(&b);
        // Both `min` and `max` are `NaN` if `a` and `b` are incomparable.
        if min.is_nan() {
            let nan = T::NAN.into();
            (nan, nan)
        }
        else {
            (min.into(), max.into())
        }
    }
}

impl<T> IntrinsicOrd for Option<T>
where
    T: Copy + PartialOrd,
{
    fn is_undefined(&self) -> bool {
        self.is_none()
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        match (self.as_ref(), other.as_ref()) {
            (Some(a), Some(b)) => match a.partial_cmp(b) {
                Some(ordering) => match ordering {
                    Ordering::Less | Ordering::Equal => (Some(*a), Some(*b)),
                    _ => (Some(*b), Some(*a)),
                },
                _ => (None, None),
            },
            _ => (None, None),
        }
    }
}

pub fn max_or_undefined<T>(a: T, b: T) -> T
where
    T: IntrinsicOrd,
{
    a.max_or_undefined(&b)
}

pub fn min_or_undefined<T>(a: T, b: T) -> T
where
    T: IntrinsicOrd,
{
    a.min_or_undefined(&b)
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::cmp::{self, IntrinsicOrd};
    use crate::{Nan, Total};

    #[test]
    fn intrinsic_ord_option() {
        let zero = Some(0u64);
        let one = Some(1u64);

        assert_eq!(zero, cmp::min_or_undefined(zero, one));
        assert_eq!(one, cmp::max_or_undefined(zero, one));
        assert!(cmp::min_or_undefined(None, zero).is_undefined());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn intrinsic_ord_primitive() {
        let zero = 0.0f64;
        let one = 1.0f64;

        assert_eq!(zero, cmp::min_or_undefined(zero, one));
        assert_eq!(one, cmp::max_or_undefined(zero, one));
        assert!(cmp::min_or_undefined(f64::NAN, zero).is_undefined());
    }

    #[test]
    fn intrinsic_ord_proxy() {
        let nan = Total::<f64>::NAN;
        let zero = Total::zero();
        let one = Total::one();

        assert_eq!((zero, one), zero.min_max_or_undefined(&one));
        assert_eq!((zero, one), one.min_max_or_undefined(&zero));

        assert_eq!((nan, nan), nan.min_max_or_undefined(&zero));
        assert_eq!((nan, nan), zero.min_max_or_undefined(&nan));
        assert_eq!((nan, nan), nan.min_max_or_undefined(&nan));

        assert_eq!(nan, cmp::min_or_undefined(nan, zero));
        assert_eq!(nan, cmp::max_or_undefined(nan, zero));
        assert_eq!(nan, cmp::min_or_undefined(nan, nan));
        assert_eq!(nan, cmp::max_or_undefined(nan, nan));
    }
}
