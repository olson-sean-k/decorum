//! Making floating-point values behave: ordering, equality, hashing, and
//! constraints for floating-point types.

#[macro_use]
extern crate derivative;
extern crate num_traits;
#[cfg(feature = "serialize-serde")]
extern crate serde;
#[cfg(feature = "serialize-serde")]
#[macro_use]
extern crate serde_derive;

use num_traits::{Float, Num, NumCast};
use std::num::FpCategory;
use std::ops::Neg;

// TODO: Support `f128`.

mod canonical;
mod constraint;
mod primitive;
mod proxy;

use constraint::{FiniteConstraint, NotNanConstraint};
use primitive::Primitive;
use proxy::ConstrainedFloat;

pub use canonical::{cmp_float, cmp_float_array, cmp_float_slice, eq_float, eq_float_array,
                    eq_float_slice, hash_float, hash_float_array, hash_float_slice};
pub use proxy::FloatProxy;

/// An ordered and canonicalized floating-point value that does not constraint
/// its values.
pub type Ordered<T> = ConstrainedFloat<T, ()>;

/// An ordered and canonicalized floating-point value that cannot be `NaN`.
pub type NotNan<T> = ConstrainedFloat<T, NotNanConstraint<T>>;

/// An alias for a floating-point value that cannot be `NaN`.
pub type N32 = NotNan<f32>;
/// An alias for a floating-point value that cannot be `NaN`.
pub type N64 = NotNan<f64>;

/// An ordered and canonicalized floating-point value that must represent a
/// real number.
///
/// `NaN`, `INF`, etc. are not allowed. This is sometimes referred to simply as
/// a "real" as seen in the `R32` and `R64` aliases.
pub type Finite<T> = ConstrainedFloat<T, FiniteConstraint<T>>;

/// An alias for a floating-point value that represents a real number.
///
/// The prefix "R" for "real" is used instead of "F" for "finite", because if
/// "F" were used, then this name would be very similar to `f32`,
/// differentiated only by capitalization.
pub type R32 = Finite<f32>;
/// An alias for a floating-point value that represents a real number.
///
/// The prefix "R" for "real" is used instead of "F" for "finite", because if
/// "F" were used, then this name would be very similar to `f64`,
/// differentiated only by capitalization.
pub type R64 = Finite<f64>;

/// A floating-point value that can be infinite (`-INF` or `INF`).
pub trait Infinite: Copy + NumCast {
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

/// A floating-point value that can be `NaN`.
pub trait Nan: Copy + NumCast {
    fn nan() -> Self;
    fn is_nan(self) -> bool;
}

/// Floating-point encoding.
pub trait Encoding: Copy + NumCast {
    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;
    fn integer_decode(self) -> (u64, i16, i8);
}

/// A floating-point representation of a real number.
pub trait Real: Copy + Neg<Output = Self> + Num + NumCast + PartialOrd {
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn min_positive_value() -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;

    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn signum(self) -> Self;
    fn abs(self) -> Self;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn recip(self) -> Self;

    fn mul_add(self, a: Self, b: Self) -> Self;
    fn abs_sub(self, other: Self) -> Self;

    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn exp_m1(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn ln_1p(self) -> Self;

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
}

impl<T> Infinite for T
where
    T: Float + Primitive,
{
    fn infinity() -> Self {
        Float::infinity()
    }

    fn neg_infinity() -> Self {
        Float::neg_infinity()
    }

    fn is_infinite(self) -> bool {
        Float::is_infinite(self)
    }

    fn is_finite(self) -> bool {
        Float::is_finite(self)
    }
}

impl<T> Nan for T
where
    T: Float + Primitive,
{
    fn nan() -> Self {
        Float::nan()
    }

    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl<T> Encoding for T
where
    T: Float + Primitive,
{
    fn classify(self) -> FpCategory {
        Float::classify(self)
    }

    fn is_normal(self) -> bool {
        Float::is_normal(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        Float::integer_decode(self)
    }
}

impl<T> Real for T
where
    T: Float + Primitive,
{
    fn max_value() -> Self {
        Float::max_value()
    }

    fn min_value() -> Self {
        Float::min_value()
    }

    fn min_positive_value() -> Self {
        Float::min_positive_value()
    }

    fn min(self, other: Self) -> Self {
        Float::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Float::max(self, other)
    }

    fn is_sign_positive(self) -> bool {
        Float::is_sign_positive(self)
    }

    fn is_sign_negative(self) -> bool {
        Float::is_sign_negative(self)
    }

    fn signum(self) -> Self {
        Float::signum(self)
    }

    fn abs(self) -> Self {
        Float::abs(self)
    }

    fn floor(self) -> Self {
        Float::floor(self)
    }

    fn ceil(self) -> Self {
        Float::ceil(self)
    }

    fn round(self) -> Self {
        Float::round(self)
    }

    fn trunc(self) -> Self {
        Float::trunc(self)
    }

    fn fract(self) -> Self {
        Float::fract(self)
    }

    fn recip(self) -> Self {
        Float::recip(self)
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Float::mul_add(self, a, b)
    }

    fn abs_sub(self, other: Self) -> Self {
        Float::abs_sub(self, other)
    }

    fn powi(self, n: i32) -> Self {
        Float::powi(self, n)
    }

    fn powf(self, n: Self) -> Self {
        Float::powf(self, n)
    }

    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }

    fn cbrt(self) -> Self {
        Float::cbrt(self)
    }

    fn exp(self) -> Self {
        Float::exp(self)
    }

    fn exp2(self) -> Self {
        Float::exp2(self)
    }

    fn exp_m1(self) -> Self {
        Float::exp_m1(self)
    }

    fn log(self, base: Self) -> Self {
        Float::log(self, base)
    }

    fn ln(self) -> Self {
        Float::ln(self)
    }

    fn log2(self) -> Self {
        Float::log2(self)
    }

    fn log10(self) -> Self {
        Float::log10(self)
    }

    fn ln_1p(self) -> Self {
        Float::ln_1p(self)
    }

    fn hypot(self, other: Self) -> Self {
        Float::hypot(self, other)
    }

    fn sin(self) -> Self {
        Float::sin(self)
    }

    fn cos(self) -> Self {
        Float::cos(self)
    }

    fn tan(self) -> Self {
        Float::tan(self)
    }

    fn asin(self) -> Self {
        Float::asin(self)
    }

    fn acos(self) -> Self {
        Float::acos(self)
    }

    fn atan(self) -> Self {
        Float::atan(self)
    }

    fn atan2(self, other: Self) -> Self {
        Float::atan2(self, other)
    }

    fn sin_cos(self) -> (Self, Self) {
        Float::sin_cos(self)
    }

    fn sinh(self) -> Self {
        Float::sinh(self)
    }

    fn cosh(self) -> Self {
        Float::cosh(self)
    }

    fn tanh(self) -> Self {
        Float::tanh(self)
    }

    fn asinh(self) -> Self {
        Float::asinh(self)
    }

    fn acosh(self) -> Self {
        Float::acosh(self)
    }

    fn atanh(self) -> Self {
        Float::atanh(self)
    }
}
