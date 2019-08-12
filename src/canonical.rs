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
use core::hash::{Hash, Hasher};
use core::mem;

#[cfg(not(feature = "std"))]
use num_traits::float::FloatCore as Float;
#[cfg(feature = "std")]
use num_traits::Float;

use crate::{Encoding, Primitive};

const SIGN_MASK: u64 = 0x8000_0000_0000_0000u64;
const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000u64;
const MANTISSA_MASK: u64 = 0x000f_ffff_ffff_ffffu64;

const CANONICAL_NAN: u64 = 0x7ff8_0000_0000_0000u64;
const CANONICAL_ZERO: u64 = 0x0u64;

pub trait FloatArray: Sized {
    type Item: Float + Primitive;

    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher;

    fn cmp<T>(&self, other: &T) -> Ordering
    where
        T: FloatArray<Item = Self::Item>;

    fn eq<T>(&self, other: &T) -> bool
    where
        T: FloatArray<Item = Self::Item>;

    fn as_slice(&self) -> &[Self::Item];
}

// TODO: Is there a better way to implement this macro? See `hash_float_array`.
macro_rules! float_array {
    (lengths => $($N:expr),*) => {$(
        impl<T> FloatArray for [T; $N]
        where
            T: Float + Primitive,
        {
            type Item = T;

            fn hash<H>(&self, state: &mut H)
            where
                H: Hasher
            {
                hash_float_slice(self, state)
            }

            fn cmp<U>(&self, other: &U) -> Ordering
            where
                U: FloatArray<Item = Self::Item>,
            {
                cmp_float_slice(self, other.as_slice())
            }

            fn eq<U>(&self, other: &U) -> bool
            where
                U: FloatArray<Item = Self::Item>,
            {
                eq_float_slice(self, other.as_slice())
            }

            fn as_slice(&self) -> &[Self::Item] {
                &self[..]
            }
        }
    )*};
}
float_array!(lengths => 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

/// Compares primitive floating-point values.
///
/// To perform the comparison, the floating-point values are interpretted in
/// canonicalized form. All `NaN`s and zeroes (positive and negative) are
/// considered equal to each other.
///
/// The total ordering is: `[-INF | ... | C0 | ... | INF | CNaN ]`.
pub fn cmp_float<T>(lhs: T, rhs: T) -> Ordering
where
    T: Float + Primitive,
{
    // Using `canonicalize_float` here would be difficult, because comparing the
    // `u64` encoding would not always have the expected results. `+0` and `-0`
    // already compare as equal, so only `NaN`s must be handled explicitly.
    match lhs.partial_cmp(&rhs) {
        Some(ordering) => ordering,
        None => {
            if lhs.is_nan() {
                if rhs.is_nan() {
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

/// Compares primitive floating-point slices.
///
/// See `cmp_float` for details of scalar comparisons. The ordering of slices
/// is determined by the first instance of non-equal corresponding elements. If
/// no such instance exists, then the length of the slices are compared; longer
/// slices are considered greater than shorter slices.  Naturally, this means
/// that slices of the same length with equivalent elements are considered
/// equal.
pub fn cmp_float_slice<T>(lhs: &[T], rhs: &[T]) -> Ordering
where
    T: Float + Primitive,
{
    match lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| cmp_float(*lhs, *rhs))
        .find(|ordering| *ordering != Ordering::Equal)
    {
        Some(ordering) => ordering,
        None => lhs.len().cmp(&rhs.len()),
    }
}

/// Compares primitive floating-point arrays.
///
/// See `cmp_float` for details of scalar comparisons. The ordering of arrays
/// is determined by the first instance of non-equal corresponding elements. If
/// no such instance exists, then the arrays are equal.
pub fn cmp_float_array<T>(lhs: &T, rhs: &T) -> Ordering
where
    T: FloatArray,
{
    lhs.cmp(rhs)
}

/// Determines if primitive floating-point values are equal.
///
/// To perform the comparison, the floating-point value is canonicalized. If
/// `NaN` or zero, a canonical form is used so that all `NaN`s and zeroes
/// (positive and negative) are considered equal.
pub fn eq_float<T>(lhs: T, rhs: T) -> bool
where
    T: Float + Primitive,
{
    canonicalize_float(lhs) == canonicalize_float(rhs)
}

/// Determines if primitive floating-point slices are equal.
///
/// See `eq_float` for details of scalar comparisons. Slices are equal if all
/// of their corresponding elements are equal. Slices of different lengths are
/// never considered equal.
pub fn eq_float_slice<T>(lhs: &[T], rhs: &[T]) -> bool
where
    T: Float + Primitive,
{
    if lhs.len() == rhs.len() {
        lhs.iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| eq_float(*lhs, *rhs))
    }
    else {
        false
    }
}

/// Determines if primitive floating-point arrays are equal.
///
/// See `eq_float` for details of scalar comparisons. Arrays are equal if all
/// of their corresponding elements are equal.
pub fn eq_float_array<T>(lhs: &T, rhs: &T) -> bool
where
    T: FloatArray,
{
    lhs.eq(rhs)
}

/// Hashes a primitive floating-point value.
///
/// To perform the hash, the floating-point value is canonicalized. If `NaN` or
/// zero, a canonical form is used so that all `NaN`s result in the same hash
/// and all zeroes (positive and negative) result in the same hash.
pub fn hash_float<T, H>(value: T, state: &mut H)
where
    T: Float + Primitive,
    H: Hasher,
{
    canonicalize_float(value).hash(state);
}

/// Hashes a slice of primitive floating-point values.
///
/// See `hash_float` for details.
pub fn hash_float_slice<T, H>(values: &[T], state: &mut H)
where
    T: Float + Primitive,
    H: Hasher,
{
    for value in values {
        hash_float(*value, state);
    }
}

// TODO: Use integer generics to implement hashing over arrays.
/// Hashes an array of primitive floating-point values.
///
/// Supports arrays up to length 16. See `hash_float` for details.
pub fn hash_float_array<T, H>(array: &T, state: &mut H)
where
    T: FloatArray,
    H: Hasher,
{
    array.hash(state);
}

fn canonicalize_float<T>(value: T) -> u64
where
    T: Float + Primitive,
{
    if value.is_nan() {
        CANONICAL_NAN
    }
    else {
        canonicalize_not_nan(value)
    }
}

fn canonicalize_not_nan<T>(value: T) -> u64
where
    T: Encoding + Primitive,
{
    let (mantissa, exponent, sign) = value.integer_decode();
    if mantissa == 0 {
        CANONICAL_ZERO
    }
    else {
        let exponent = u64::from(unsafe { mem::transmute::<i16, u16>(exponent) });
        let sign = if sign > 0 { 1u64 } else { 0u64 };

        (mantissa & MANTISSA_MASK) | ((exponent << 52) & EXPONENT_MASK) | ((sign << 63) & SIGN_MASK)
    }
}
