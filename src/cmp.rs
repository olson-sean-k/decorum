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

pub trait NanOrd: Copy + PartialOrd + Sized {
    fn min_max_or_nan(&self, other: &Self) -> (Self, Self);

    fn min_or_nan(&self, other: &Self) -> Self {
        self.min_max_or_nan(other).0
    }

    fn max_or_nan(&self, other: &Self) -> Self {
        self.min_max_or_nan(other).1
    }
}
macro_rules! impl_nan_ord {
    (total => $t:ty) => {
        impl NanOrd for $t {
            fn min_max_or_nan(&self, other: &Self) -> (Self, Self) {
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
        impl NanOrd for $t {
            fn min_max_or_nan(&self, other: &Self) -> (Self, Self) {
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
impl_nan_ord!(total => isize);
impl_nan_ord!(total => i8);
impl_nan_ord!(total => i16);
impl_nan_ord!(total => i32);
impl_nan_ord!(total => i64);
impl_nan_ord!(total => i128);
impl_nan_ord!(total => usize);
impl_nan_ord!(total => u8);
impl_nan_ord!(total => u16);
impl_nan_ord!(total => u32);
impl_nan_ord!(total => u64);
impl_nan_ord!(total => u128);
impl_nan_ord!(nan => f32);
impl_nan_ord!(nan => f64);

// Note that it is not necessary for `NaN` to be a member of the constraint.
// This implementation explicitly detects `NaN`s and emits `NaN` as the
// maximum and minimum (it does not use `FloatOrd`).
impl<T, P> NanOrd for ConstrainedFloat<T, P>
where
    T: Encoding + Nan + NanOrd + Primitive,
    P: Constraint<T>,
{
    fn min_max_or_nan(&self, other: &Self) -> (Self, Self) {
        // This function operates on primitive floating-point values. This
        // avoids the need for implementations for each combination of proxy and
        // constraint (proxy types do not always implement `Nan`, but primitive
        // types do).
        let a = self.into_inner();
        let b = other.into_inner();
        let (min, max) = a.min_max_or_nan(&b);
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

impl<T> NanOrd for Option<T>
where
    T: Copy + PartialOrd,
{
    fn min_max_or_nan(&self, other: &Self) -> (Self, Self) {
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

pub fn max_or_nan<T>(a: T, b: T) -> T
where
    T: NanOrd,
{
    a.max_or_nan(&b)
}

pub fn min_or_nan<T>(a: T, b: T) -> T
where
    T: NanOrd,
{
    a.min_or_nan(&b)
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::cmp::{self, FloatEq, NanOrd};
    use crate::{Nan, Total};

    #[test]
    fn nan_ord_option() {
        let zero = Some(0u64);
        let one = Some(1u64);

        assert_eq!(zero, cmp::min_or_nan(zero, one));
        assert_eq!(one, cmp::max_or_nan(zero, one));
        assert_eq!(None, cmp::min_or_nan(None, zero));
    }

    #[test]
    fn nan_ord_primitive() {
        assert_eq!(0.0f64, cmp::min_or_nan(0.0, 1.0));
        assert_eq!(1.0f64, cmp::max_or_nan(0.0, 1.0));
        assert!(FloatEq::eq(&f64::NAN, &cmp::min_or_nan(f64::NAN, 0.0)));
    }

    #[test]
    fn nan_ord_proxy() {
        let nan = Total::<f64>::NAN;
        let zero = Total::zero();
        let one = Total::one();

        assert_eq!((zero, one), zero.min_max_or_nan(&one));
        assert_eq!((zero, one), one.min_max_or_nan(&zero));

        assert_eq!((nan, nan), nan.min_max_or_nan(&zero));
        assert_eq!((nan, nan), zero.min_max_or_nan(&nan));
        assert_eq!((nan, nan), nan.min_max_or_nan(&nan));

        assert_eq!(nan, cmp::min_or_nan(nan, zero));
        assert_eq!(nan, cmp::max_or_nan(nan, zero));
        assert_eq!(nan, cmp::min_or_nan(nan, nan));
        assert_eq!(nan, cmp::max_or_nan(nan, nan));
    }
}
