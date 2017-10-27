//! Constrained, ordered, hashable floating point values.

#[macro_use]
extern crate derivative;
extern crate num_traits;
#[cfg(feature = "serialize-serde")]
extern crate serde;

use num_traits::Float;
use std::num::FpCategory;

// TODO: Emit useful errors and use the error_chain crate.

mod constrain;
mod hash;

use constrain::{ConstrainedFloat, FiniteConstraint, NotNanConstraint};

pub use hash::{hash_float, hash_float_array, hash_float_slice};

/// A floating point value that can have any IEEE-754 value, but is ordered and
/// normalized.
pub type Ordered<T> = ConstrainedFloat<T, ()>;

/// A floating point value that cannot be `NaN`.
pub type NotNan<T> = ConstrainedFloat<T, NotNanConstraint<T>>;

/// A floating point value that cannot be `NaN`, `INF`, or `-INF` (i.e., a
/// represnetation of a real value).
pub type Finite<T> = ConstrainedFloat<T, FiniteConstraint<T>>;

pub type N32 = NotNan<f32>;
pub type N64 = NotNan<f64>;

// Use "R" for "real" instead of "F" for "finite", because then this name would
// be very similar to `f32` and `f64`, differentiated only be capitalization.
pub type R32 = Finite<f32>;
pub type R64 = Finite<f64>;

pub trait Primitive {}

impl Primitive for f32 {}
impl Primitive for f64 {}

// This is essentially `num_traits::Float` without its NaN or INF functions.
// Until such a distinction is made upstream, this can be used to be generic
// over both raw and constrained floats.
//
// Implementations for `Real`, `Infinity` and `Nan` are provided for all types
// implementing `num_traits::Float`.
pub trait Real: Copy + Sized {
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn min_positive_value() -> Self;

    fn neg_zero() -> Self;

    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn signum(self) -> Self;
    fn abs(self) -> Self;

    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;

    fn integer_decode(self) -> (u64, i16, i8);

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

/// A value that can be infinite.
pub trait Infinite: Copy + Sized {
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

/// A value that can be `NaN`.
pub trait Nan: Copy + Sized {
    fn nan() -> Self;
    fn is_nan(self) -> bool;
}

// TODO: These blanket implementations over types implementing `Float` make it
//       difficult to implement `Float` for `ConstrainedFloat` types (because
//       they would provide conflicting implementations for the more targeted
//       traits like `Real`).
//
//       It may still be possible to implement `Float` for the `Ordered` type,
//       but it will likely require some refactoring and constraints on
//       `ConstrainedFloat`'s implementation of `Real`, `Infinite`, and `NaN`.
//       In the meantime, code can use `Real + Infinite + Nan` to capture
//       `Ordered` and raw floating point values (anything that is `Float` and
//       anything that is "logically" `Float`).

impl<T> Real for T
where
    T: Float + Primitive,
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
    fn signum(self) -> Self {
        Float::signum(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Float::abs(self)
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
    fn floor(self) -> Self {
        Float::floor(self)
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Float::ceil(self)
    }

    #[inline(always)]
    fn round(self) -> Self {
        Float::round(self)
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        Float::trunc(self)
    }

    #[inline(always)]
    fn fract(self) -> Self {
        Float::fract(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Float::recip(self)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Float::mul_add(self, a, b)
    }

    #[inline(always)]
    fn abs_sub(self, other: Self) -> Self {
        Float::abs_sub(self, other)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        Float::powi(self, n)
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        Float::powf(self, n)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }

    #[inline(always)]
    fn cbrt(self) -> Self {
        Float::cbrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        Float::exp(self)
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        Float::exp2(self)
    }

    #[inline(always)]
    fn exp_m1(self) -> Self {
        Float::exp_m1(self)
    }

    #[inline(always)]
    fn log(self, base: Self) -> Self {
        Float::log(self, base)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        Float::ln(self)
    }

    #[inline(always)]
    fn log2(self) -> Self {
        Float::log2(self)
    }

    #[inline(always)]
    fn log10(self) -> Self {
        Float::log10(self)
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        Float::ln_1p(self)
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

impl<T> Infinite for T
where
    T: Float + Primitive,
{
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
}

impl<T> Nan for T
where
    T: Float + Primitive,
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
