//! Making floating-point values behave: traits, ordering, equality, hashing,
//! and constraints for floating-point types.
//!
//! Decorum provides proxy (wrapper) types and functions that canonicalize
//! floating-point values and provide a total ordering. This allows
//! floating-point values to be hashed and compared. Proxy types also provide
//! contraints on the values that may be represented and will panic if those
//! constraints are violated. See the [README](https://docs.rs/crate/decorum/).
//!
//! # Ordering
//!
//! `NaN` and zero are canonicalized to a single representation (called `CNaN`
//! and `C0` respectively) to provide the following total ordering for all
//! proxy types and ordering functions:
//!
//! `[ -INF | ... | C0 | ... | +INF | CNaN ]`
//!
//! Note that `NaN` is canonicalized to `CNaN`, which has a single
//! representation and supports the relations `CNaN = CNaN` and `CNaN > x | x ≠
//! CNaN`. `+0` and `-0` are also canonicalized to `C0`, which is equivalent to
//! `+0`.
//!
//! # Constraints
//!
//! The `NotNan` and `Finite` types wrap raw floating-point values and disallow
//! certain values like `NaN`, `INF`, and `-INF`. They will panic if an
//! operation or conversion invalidates these constraints. The `Ordered` type
//! allows any valid IEEE-754 value (there are no constraints). For most use
//! cases, either `Ordered` or `NotNan` are appropriate.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::num::FpCategory;
use core::ops::Neg;
#[allow(unused_imports)]
use num_traits::{Num, NumCast};

mod canonical;
pub mod cmp;
mod constraint;
pub mod hash;
mod primitive;
mod proxy;

use crate::constraint::{FiniteConstraint, NotNanConstraint, UnitConstraint};

pub use crate::canonical::ToCanonicalBits;
pub use crate::primitive::Primitive;
pub use crate::proxy::ConstrainedFloat;

/// An ordered and canonicalized floating-point value.
pub type Total<T> = ConstrainedFloat<T, UnitConstraint<T>>;

/// An ordered and canonicalized floating-point value that cannot be `NaN`.
///
/// If any operation results in a `NaN` value, then a panic will occur.
pub type NotNan<T> = ConstrainedFloat<T, NotNanConstraint<T>>;

/// An alias for a floating-point value that cannot be `NaN`.
pub type N32 = NotNan<f32>;
/// An alias for a floating-point value that cannot be `NaN`.
pub type N64 = NotNan<f64>;

/// An ordered and canonicalized floating-point value that must represent a
/// real number.
///
/// `NaN`, `INF`, etc. are not allowed and a panic will occur if any operation
/// results in such a value. This is sometimes referred to simply as a "real" as
/// seen in the `R32` and `R64` aliases.
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
pub trait Infinite: Copy {
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

/// A floating-point value that can be `NaN`.
pub trait Nan: Copy {
    const NAN: Self;

    fn is_nan(self) -> bool;
}

/// Floating-point encoding.
///
/// Provides values and operations that directly relate to the encoding of an
/// IEEE-754 floating-point value with the exception of `-INF`, `INF`, and
/// `NaN`. See the `Infinite` and `Nan` traits.
pub trait Encoding: Copy {
    const MAX: Self;
    const MIN: Self;
    const MIN_POSITIVE: Self;
    const EPSILON: Self;

    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;
    fn integer_decode(self) -> (u64, i16, i8);
}

/// A value that can represent a real number.
///
/// Provides values and operations that generally apply to real numbers.
pub trait Real: Copy + Neg<Output = Self> + Num + NumCast + PartialOrd {
    const E: Self;
    const PI: Self;
    const FRAC_1_PI: Self;
    const FRAC_2_PI: Self;
    const FRAC_2_SQRT_PI: Self;
    const FRAC_PI_2: Self;
    const FRAC_PI_3: Self;
    const FRAC_PI_4: Self;
    const FRAC_PI_6: Self;
    const FRAC_PI_8: Self;
    const SQRT_2: Self;
    const FRAC_1_SQRT_2: Self;
    const LN_2: Self;
    const LN_10: Self;
    const LOG2_E: Self;
    const LOG10_E: Self;

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

    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self;

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self;
    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self;
    #[cfg(feature = "std")]
    fn sqrt(self) -> Self;
    #[cfg(feature = "std")]
    fn cbrt(self) -> Self;
    #[cfg(feature = "std")]
    fn exp(self) -> Self;
    #[cfg(feature = "std")]
    fn exp2(self) -> Self;
    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self;
    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self;
    #[cfg(feature = "std")]
    fn ln(self) -> Self;
    #[cfg(feature = "std")]
    fn log2(self) -> Self;
    #[cfg(feature = "std")]
    fn log10(self) -> Self;
    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self;

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self;
    #[cfg(feature = "std")]
    fn sin(self) -> Self;
    #[cfg(feature = "std")]
    fn cos(self) -> Self;
    #[cfg(feature = "std")]
    fn tan(self) -> Self;
    #[cfg(feature = "std")]
    fn asin(self) -> Self;
    #[cfg(feature = "std")]
    fn acos(self) -> Self;
    #[cfg(feature = "std")]
    fn atan(self) -> Self;
    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self;
    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self);
    #[cfg(feature = "std")]
    fn sinh(self) -> Self;
    #[cfg(feature = "std")]
    fn cosh(self) -> Self;
    #[cfg(feature = "std")]
    fn tanh(self) -> Self;
    #[cfg(feature = "std")]
    fn asinh(self) -> Self;
    #[cfg(feature = "std")]
    fn acosh(self) -> Self;
    #[cfg(feature = "std")]
    fn atanh(self) -> Self;
}
