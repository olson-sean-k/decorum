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
    T: Encoding + Nan + Primitive,
    P: Constraint<T>,
{
    fn min_max_or_nan(&self, other: &Self) -> (Self, Self) {
        // This function operates on primitive floating-point values. This
        // avoids the need for implementations for each combination of proxy and
        // constraint (proxy types do not always implement `Nan`, but primitive
        // types do).
        let a = self.into_inner();
        let b = other.into_inner();
        match a.partial_cmp(&b) {
            Some(ordering) => match ordering {
                Ordering::Less | Ordering::Equal => (a.into(), b.into()),
                _ => (b.into(), a.into()),
            },
            _ => {
                let nan = T::NAN.into();
                (nan, nan)
            }
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
