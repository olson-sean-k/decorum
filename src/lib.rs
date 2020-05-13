//! Making floating-point behave: ordering, equivalence, hashing, and
//! constraints for floating-point types.
//!
//! Decorum provides traits that describe types using floating-point
//! representations and provides proxy types that wrap primitive floating-point
//! types. Proxy types implement a total ordering and constraints on the classes
//! of values that they may represent.
//!
//! # Floating-Point Classes
//!
//! Traits, proxy types, and constraints are based on three classes or subsets
//! of floating-point values:
//!
//! | Class        | Trait      | Proxy Types                 |
//! |--------------|------------|-----------------------------|
//! | real number  | `Real`     | `Total`, `NotNan`, `Finite` |
//! | infinity     | `Infinite` | `Total`, `NotNan`           |
//! | not-a-number | `Nan`      | `Total`, `Finite`           |
//!
//! Primitive floating-point values directly expose IEEE-754 and therefore the
//! complete set of values (and traits). Proxy types like `NotNan` and `Finite`
//! use a subset of values that disallow `NaN`s and/or infinities. Traits are
//! implemented by types that expose the corresponding class of values, so
//! `NotNan` implements the `Real` and `Infinite` traits but not the `Nan`
//! trait, for example.
//!
//! # Constraints
//!
//! The `NotNan` and `Finite` types wrap primitive floating-point values and
//! disallow values that represent `NaN`, $\infin$, and $-\infin$. **Operations
//! that emit values that violate these constraints will panic**. The `Total`
//! type applies no constraints and exposes all classes of values.
//!
//! # Total Ordering
//!
//! The following total ordering is implemented by all proxy types and is
//! provided by traits in the `cmp` module:
//!
//! $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
//!
//! Note that all zero and `NaN` representations are considered equivalent. See
//! the `cmp` module documentation for more details.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum-favicon.ico"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.svg?sanitize=true"
)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::num::FpCategory;
use core::ops::Neg;
#[allow(unused_imports)]
use num_traits::{Num, NumCast, Signed};

#[cfg(not(feature = "std"))]
pub(in crate) use num_traits::float::FloatCore as ForeignFloat;
#[cfg(feature = "std")]
pub(in crate) use num_traits::real::Real as ForeignReal;
#[cfg(feature = "std")]
pub(in crate) use num_traits::Float as ForeignFloat;

mod canonical;
pub mod cmp;
mod constraint;
pub mod hash;
mod primitive;
mod proxy;

use crate::cmp::IntrinsicOrd;
use crate::constraint::{FiniteConstraint, NotNanConstraint, UnitConstraint};

pub use crate::canonical::ToCanonicalBits;
pub use crate::primitive::Primitive;
pub use crate::proxy::ConstrainedFloat;

/// Floating-point representation with total ordering.
pub type Total<T> = ConstrainedFloat<T, UnitConstraint<T>>;

/// Floating-point representation that cannot be `NaN`.
///
/// If an operation emits `NaN`, then a panic will occur. Like `Total`, this
/// type implements a total ordering.
pub type NotNan<T> = ConstrainedFloat<T, NotNanConstraint<T>>;

/// 32-bit floating-point representation that cannot be `NaN`.
pub type N32 = NotNan<f32>;
/// 64-bit floating-point representation that cannot be `NaN`.
pub type N64 = NotNan<f64>;

/// Floating-point representation that must be a real number.
///
/// If an operation emits `NaN` or infinities, then a panic will occur. Like
/// `Total`, this type implements a total ordering.
pub type Finite<T> = ConstrainedFloat<T, FiniteConstraint<T>>;

/// 32-bit floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if
/// "F" were used, then this name would be very similar to `f32`, differentiated
/// only by capitalization.
pub type R32 = Finite<f32>;
/// 64-bit floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if
/// "F" were used, then this name would be very similar to `f64`, differentiated
/// only by capitalization.
pub type R64 = Finite<f64>;

/// Floating-point representations that expose infinities.
pub trait Infinite: Encoding {
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

/// Floating-point representations that expose `NaN`s.
pub trait Nan: Encoding {
    /// A representation of `NaN`.
    ///
    /// For primitive floating-point types, `NaN` is incomparable. Therefore,
    /// prefer the `is_nan` predicate over direct comparisons with `NaN`.
    const NAN: Self;

    fn is_nan(self) -> bool;
}

/// Floating-point encoding.
///
/// Provides values and operations that describe the encoding of an IEEE-754
/// floating-point value. Infinities and `NaN`s are described by the `Infinite`
/// and `NaN` sub-traits.
pub trait Encoding: Copy {
    const MAX: Self;
    const MIN: Self;
    const MIN_POSITIVE: Self;
    const EPSILON: Self;

    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;

    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;

    fn integer_decode(self) -> (u64, i16, i8);
}

/// Types that can represent real numbers.
///
/// Provides values and operations that generally apply to real numbers. As
/// such, this trait is implemented by types using floating-point
/// representations, but this trait is a general numeric trait and can apply to
/// other numeric types as well.
///
/// Some members of this trait depend on the standard library and the `std`
/// feature.
pub trait Real: Copy + Neg<Output = Self> + Num + PartialOrd + Signed {
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

/// Floating-point representations.
///
/// Types that implement this trait are represented using IEEE-754 encoding and
/// expose the details of that encoding, including infinities, `NaN`, and
/// operations on real numbers. This trait is implemented by primitive
/// floating-point types and the `Total` proxy type.
pub trait Float: Encoding + Infinite + IntrinsicOrd + Nan + Real {}

impl<T> Float for T where T: Encoding + Infinite + IntrinsicOrd + Nan + Real {}
