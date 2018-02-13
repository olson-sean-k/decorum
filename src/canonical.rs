use num_traits::Float;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use {Primitive, Real};

const SIGN_MASK: u64 = 0x8000000000000000u64;
const EXPONENT_MASK: u64 = 0x7ff0000000000000u64;
const MANTISSA_MASK: u64 = 0x000fffffffffffffu64;

const CANONICAL_NAN: u64 = 0x7ff8000000000000u64;
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

pub fn cmp_float<T>(lhs: T, rhs: T) -> Ordering
where
    T: Float + Primitive,
{
    match lhs.partial_cmp(&rhs) {
        Some(ordering) => ordering,
        None => if lhs.is_nan() {
            if rhs.is_nan() {
                Ordering::Equal
            }
            else {
                Ordering::Greater
            }
        }
        else {
            Ordering::Less
        },
    }
}

pub fn cmp_float_slice<T>(lhs: &[T], rhs: &[T]) -> Ordering
where
    T: Float + Primitive,
{
    match lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| cmp_float(*lhs, *rhs))
        .find(|ordering| *ordering != Ordering::Equal)
    {
        Some(ordering) => ordering,
        None => lhs.len().cmp(&rhs.len()),
    }
}

pub fn cmp_float_array<T>(lhs: &T, rhs: &T) -> Ordering
where
    T: FloatArray,
{
    lhs.cmp(rhs)
}

// TODO: Consider comparing the output of `canonicalize` here.
pub fn eq_float<T>(lhs: T, rhs: T) -> bool
where
    T: Float + Primitive,
{
    if lhs.is_nan() {
        if rhs.is_nan() {
            true
        }
        else {
            false
        }
    }
    else {
        if rhs.is_nan() {
            false
        }
        else {
            lhs == rhs
        }
    }
}

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

pub fn eq_float_array<T>(lhs: &T, rhs: &T) -> bool
where
    T: FloatArray,
{
    lhs.eq(rhs)
}

/// Hashes a raw floating point value.
///
/// To perform the hash, the floating point value is normalized. If `NaN` or
/// zero, a canonical form is used, so all `NaN`s result in the same hash and
/// all zeroes (positive and negative) result in the same hash.
pub fn hash_float<T, H>(value: T, state: &mut H)
where
    T: Float + Primitive,
    H: Hasher,
{
    canonicalize_float(value).hash(state);
}

/// Hashes a slice of raw floating point values.
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
/// Hashes an array of raw floating point values.
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
    T: Primitive + Real,
{
    use std::mem;

    let (mantissa, exponent, sign) = value.integer_decode();
    if mantissa == 0 {
        CANONICAL_ZERO
    }
    else {
        let exponent = unsafe { mem::transmute::<i16, u16>(exponent) } as u64;
        let sign = if sign > 0 { 1u64 } else { 0u64 };

        (mantissa & MANTISSA_MASK) | ((exponent << 52) & EXPONENT_MASK) | ((sign << 63) & SIGN_MASK)
    }
}
