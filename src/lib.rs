extern crate num_traits;
#[cfg(feature = "serialize-serde")]
extern crate serde;

// TODO: Emit useful errors and use the error_chain crate.

mod hash;
mod notnan;
mod ordered;

pub use hash::{hash_float, hash_float_array, hash_float_slice};
pub use notnan::NotNan;

use num_traits::Float;
use std::num::FpCategory;

// This is essentially `num_traits::Float` without its NaN functions. Until
// such a distinction is made upstream, this can be used to be generic over
// floats, including `NotNan`.
pub trait Real: Copy + Sized {
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn min_positive_value() -> Self;

    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;

    fn neg_zero() -> Self;

    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;

    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;

    fn integer_decode(self) -> (u64, i16, i8);

    fn hypot(self, other: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;

    // TODO: Provide the remaining functions from `Float`.
}

pub trait Nan: Copy + Sized {
    fn nan() -> Self;
    fn is_nan(self) -> bool;
}

impl<T> Real for T
where
    T: Float,
{
    #[inline(always)]
    fn max_value() -> Self {
        Float::max_value()
    }

    #[inline(always)]
    fn min_value() -> Self {
        Float::min_value()
    }

    #[inline(always)]
    fn min_positive_value() -> Self {
        Float::min_positive_value()
    }

    #[inline(always)]
    fn infinity() -> Self {
        Float::infinity()
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        Float::neg_infinity()
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        Float::is_infinite(self)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        Float::is_finite(self)
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        Float::neg_zero()
    }

    #[inline(always)]
    fn is_sign_positive(self) -> bool {
        Float::is_sign_positive(self)
    }

    #[inline(always)]
    fn is_sign_negative(self) -> bool {
        Float::is_sign_negative(self)
    }

    #[inline(always)]
    fn classify(self) -> FpCategory {
        Float::classify(self)
    }

    #[inline(always)]
    fn is_normal(self) -> bool {
        Float::is_normal(self)
    }

    #[inline(always)]
    fn integer_decode(self) -> (u64, i16, i8) {
        Float::integer_decode(self)
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        Float::hypot(self, other)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Float::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Float::cos(self)
    }

    #[inline(always)]
    fn tan(self) -> Self {
        Float::tan(self)
    }

    #[inline(always)]
    fn asin(self) -> Self {
        Float::asin(self)
    }

    #[inline(always)]
    fn acos(self) -> Self {
        Float::acos(self)
    }

    #[inline(always)]
    fn atan(self) -> Self {
        Float::atan(self)
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        Float::atan2(self, other)
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        Float::sin_cos(self)
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        Float::sinh(self)
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        Float::cosh(self)
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        Float::tanh(self)
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        Float::asinh(self)
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        Float::acosh(self)
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        Float::atanh(self)
    }
}

impl<T> Nan for T
where
    T: Float,
{
    #[inline(always)]
    fn nan() -> Self {
        Float::nan()
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}
