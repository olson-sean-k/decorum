//! APIs describing real numbers and interactions with IEEE 754 floating-point types.

use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::cmp::IntrinsicOrd;
use crate::{Float, Infinite, Primitive};

pub trait Function {
    type Codomain;
}

pub trait Endofunction: Function<Codomain = Self> {}

impl<T> Endofunction for T where T: Function<Codomain = T> {}

// This trait is implemented by trivial `Copy` types.
#[allow(clippy::wrong_self_convention)]
pub trait UnaryReal:
    Function + IntrinsicOrd + Neg<Output = Self> + PartialEq + PartialOrd + Sized
{
    const ZERO: Self;
    const ONE: Self;
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

    fn is_zero(self) -> bool;
    fn is_one(self) -> bool;

    fn sign(self) -> Sign;
    #[cfg(feature = "std")]
    fn abs(self) -> Self;

    #[cfg(feature = "std")]
    fn floor(self) -> Self;
    #[cfg(feature = "std")]
    fn ceil(self) -> Self;
    #[cfg(feature = "std")]
    fn round(self) -> Self;
    #[cfg(feature = "std")]
    fn trunc(self) -> Self;
    #[cfg(feature = "std")]
    fn fract(self) -> Self;
    fn recip(self) -> Self::Codomain; // Undefined.

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Codomain; // Floating-point exception or undefined.
    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn cbrt(self) -> Self;
    #[cfg(feature = "std")]
    fn exp(self) -> Self::Codomain; // Floating-point exception.
    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Codomain; // Floating-point exception.
    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Codomain; // Floating-point exception.
    #[cfg(feature = "std")]
    fn ln(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn log2(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn log10(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self::Codomain; // Undefined.

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Codomain; // Floating-point exception.
    #[cfg(feature = "std")]
    fn to_radians(self) -> Self;
    #[cfg(feature = "std")]
    fn sin(self) -> Self;
    #[cfg(feature = "std")]
    fn cos(self) -> Self;
    #[cfg(feature = "std")]
    fn tan(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn asin(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn acos(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn atan(self) -> Self;
    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self);
    #[cfg(feature = "std")]
    fn sinh(self) -> Self;
    #[cfg(feature = "std")]
    fn cosh(self) -> Self;
    #[cfg(feature = "std")]
    fn tanh(self) -> Self;
    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Codomain; // Undefined.
}

// NOTE: Because `T` is not constrained, it isn't possible for functions that always map reals to
//       reals to express their output as `Self`. The `T` input may not be real and that may result
//       in a non-real output.
pub trait BinaryReal<T = Self>:
    Add<T, Output = Self::Codomain>
    + Div<T, Output = Self::Codomain>
    + Mul<T, Output = Self::Codomain>
    + Rem<T, Output = Self::Codomain>
    + Sub<T, Output = Self::Codomain>
    + UnaryReal
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: T) -> Self::Codomain; // Undefined.
    #[cfg(feature = "std")]
    fn rem_euclid(self, n: T) -> Self::Codomain; // Undefined.

    #[cfg(feature = "std")]
    fn pow(self, n: T) -> Self::Codomain; // Floating-point exception or undefined.
    #[cfg(feature = "std")]
    fn log(self, base: T) -> Self::Codomain; // Undefined.

    #[cfg(feature = "std")]
    fn hypot(self, other: T) -> Self::Codomain; // Floating-point exception.
    #[cfg(feature = "std")]
    fn atan2(self, other: T) -> Self::Codomain;
}

pub trait Real: BinaryReal<Self> {}

impl<T> Real for T where T: BinaryReal<T> {}

pub trait ExtendedReal: Infinite + Real {}

impl<T> ExtendedReal for T where T: Infinite + Real {}

pub trait Endoreal: Endofunction + Real {}

impl<T> Endoreal for T where T: Endofunction + Real {}

pub trait FloatReal<T>: BinaryReal<T> + Into<T> + Real + TryFrom<T>
where
    T: Float + Primitive,
{
}

impl<T, U> FloatReal<T> for U
where
    T: Float + Primitive,
    U: BinaryReal<T> + Into<T> + Real + TryFrom<T>,
{
}

pub trait FloatEndoreal<T>: Endoreal + FloatReal<T>
where
    T: Float + Primitive,
{
}

impl<T, U> FloatEndoreal<T> for U
where
    T: Float + Primitive,
    U: Endofunction + FloatReal<T>,
{
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Sign {
    Positive,
    Negative,
    Zero,
}

impl Sign {
    pub fn is_positive(&self) -> bool {
        matches!(self, Sign::Positive)
    }

    pub fn is_negative(&self) -> bool {
        matches!(self, Sign::Negative)
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Sign::Zero)
    }
}
