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
use crate::primitive::Primitive;
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

// TODO: Implement for as many types as possible, including types behind
//       feature flags.
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
                    None => unreachable!(),
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
                    None => (Nan::nan(), Nan::nan()),
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
