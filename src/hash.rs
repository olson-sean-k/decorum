use num_traits::Float;
use std::hash::{Hash, Hasher};

use Real;

const SIGN_MASK: u64 = 0x8000000000000000u64;
const EXPONENT_MASK: u64 = 0x7ff0000000000000u64;
const MANTISSA_MASK: u64 = 0x000fffffffffffffu64;

const CANONICAL_NAN: u64 = 0x7ff8000000000000u64;
const CANONICAL_ZERO: u64 = 0x0u64;

pub trait FloatArray {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher;
}

// TODO: Is there a better way to implement this macro? See `hash_float_array`.
macro_rules! float_array {
    (lengths => $($N:expr),*) => {$(
        impl<T> FloatArray for [T; $N]
        where
            T: Float,
        {
            fn hash<H>(&self, state: &mut H)
            where
                H: Hasher
            {
                hash_float_slice(self, state)
            }
        }
    )*};
}
float_array!(lengths => 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

#[inline(always)]
pub fn hash_float<T, H>(value: T, state: &mut H)
where
    T: Float,
    H: Hasher,
{
    canonicalize_float(value).hash(state);
}

pub fn hash_float_slice<T, H>(values: &[T], state: &mut H)
where
    T: Float,
    H: Hasher,
{
    for value in values {
        hash_float(*value, state);
    }
}

// TODO: Use integer generics to implement hashing over arrays.
pub fn hash_float_array<T, H>(array: &T, state: &mut H)
where
    T: FloatArray,
    H: Hasher,
{
    array.hash(state);
}

fn canonicalize_float<T>(value: T) -> u64
where
    T: Float,
{
    if value.is_nan() {
        CANONICAL_NAN
    }
    else {
        canonicalize_real(value)
    }
}

fn canonicalize_real<T>(value: T) -> u64
where
    T: Real,
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
