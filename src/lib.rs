//! Making floating-point behave: ordering, equivalence, hashing, and
//! constraints for floating-point types.
//!
//! Decorum provides traits that describe types using floating-point
//! representations and provides proxy types that wrap primitive floating-point
//! types. Proxy types implement a total ordering and constraints on the sets of
//! values that they may represent.
//!
//! Decorum requires Rust 1.43.0 or higher.
//!
//! # Floating-Point Classes
//!
//! Traits, proxy types, and constraints are based on three classes or subsets
//! of floating-point values:
//!
//! | Set          | Trait        |
//! |--------------|--------------|
//! | real number  | [`Real`]     |
//! | infinity     | [`Infinite`] |
//! | not-a-number | [`Nan`]      |
//!
//! Primitive floating-point values directly expose IEEE-754 and therefore the
//! complete set of values (and traits). Proxy types implement traits that are
//! compatible with their constraints, so types that disallow `NaN`s do not
//! implement the `Nan` trait, for example.
//!
//! # Proxy Types
//!
//! Proxy types wrap primitive floating-point types and constrain the sets of
//! values that they can represent:
//!
//! | Type       | Aliases      | Trait Implementations                      | Disallowed Values     |
//! |------------|--------------|--------------------------------------------|-----------------------|
//! | [`Total`]  |              | `Encoding + Real + Infinite + Nan + Float` |                       |
//! | [`NotNan`] | `N32`, `N64` | `Encoding + Real + Infinite`               | `NaN`                 |
//! | [`Finite`] | `R32`, `R64` | `Encoding + Real`                          | `NaN`, `-INF`, `+INF` |
//!
//! The [`NotNan`] and [`Finite`] types disallow values that represent `NaN`,
//! $\infin$, and $-\infin$. **Operations that emit values that violate these
//! constraints will panic**. The [`Total`] type applies no constraints and
//! exposes all classes of floating-point values.
//!
//! # Total Ordering
//!
//! The following total ordering is implemented by all proxy types and is
//! provided by traits in the [`cmp`] module:
//!
//! $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
//!
//! Note that all zero and `NaN` representations are considered equivalent. See
//! the [`cmp`] module documentation for more details.
//!
//! # Equivalence
//!
//! Floating-point `NaN`s have numerous representations and are incomparable.
//! Decorum considers all `NaN` representations equal to all other `NaN`
//! representations and any and all `NaN` representations are unequal to
//! non-`NaN` values.
//!
//! See the [`cmp`] module documentation for more details.
//!
//! [`cmp`]: crate::cmp
//! [`Finite`]: crate::Finite
//! [`Infinite`]: crate::Infinite
//! [`Nan`]: crate::Nan
//! [`NotNan`]: crate::NotNan
//! [`Real`]: crate::Real
//! [`Total`]: crate::Total

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum-favicon.ico"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.svg?sanitize=true"
)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::mem;
use core::num::FpCategory;
use core::ops::Neg;
use num_traits::{Num, PrimInt, Signed, Unsigned};

#[cfg(not(feature = "std"))]
pub(in crate) use num_traits::float::FloatCore as ForeignFloat;
#[cfg(feature = "std")]
pub(in crate) use num_traits::real::Real as ForeignReal;
#[cfg(feature = "std")]
pub(in crate) use num_traits::Float as ForeignFloat;

pub mod cmp;
mod constraint;
pub mod hash;
mod proxy;

use crate::cmp::IntrinsicOrd;
use crate::constraint::{FiniteConstraint, NotNanConstraint, UnitConstraint};

pub use crate::proxy::Proxy;

/// Floating-point representation with total ordering.
pub type Total<T> = Proxy<T, UnitConstraint<T>>;

/// Floating-point representation that cannot be `NaN`.
///
/// If an operation emits `NaN`, then a panic will occur. Like [`Total`], this
/// type implements a total ordering.
///
/// [`Total`]: crate::Total
pub type NotNan<T> = Proxy<T, NotNanConstraint<T>>;

/// 32-bit floating-point representation that cannot be `NaN`.
pub type N32 = NotNan<f32>;
/// 64-bit floating-point representation that cannot be `NaN`.
pub type N64 = NotNan<f64>;

/// Floating-point representation that must be a real number.
///
/// If an operation emits `NaN` or infinities, then a panic will occur. Like
/// [`Total`], this type implements a total ordering.
///
/// [`Total`]: crate::Total
pub type Finite<T> = Proxy<T, FiniteConstraint<T>>;

/// 32-bit floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if
/// "F" were used, then this name would be very similar to `f32`.
pub type R32 = Finite<f32>;
/// 64-bit floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if
/// "F" were used, then this name would be very similar to `f64`.
pub type R64 = Finite<f64>;

// TODO: Inverse the relationship between `Encoding` and `ToCanonicalBits` such
//       that `Encoding` requires `ToCanonicalBits`.
/// Converts floating-point values into a canonicalized form.
pub trait ToCanonicalBits: Encoding {
    type Bits: PrimInt + Unsigned;

    /// Conversion to a canonical representation.
    ///
    /// Unlike the `to_bits` function provided by `f32` and `f64`, this function
    /// collapses representations for real numbers, infinities, and `NaN`s into
    /// a canonical form such that every semantic value has a unique
    /// representation as canonical bits.
    fn to_canonical_bits(self) -> Self::Bits;
}

// TODO: Implement this differently for differently sized types.
impl<T> ToCanonicalBits for T
where
    T: Encoding + Nan + Primitive,
{
    type Bits = u64;

    fn to_canonical_bits(self) -> Self::Bits {
        const SIGN_MASK: u64 = 0x8000_0000_0000_0000;
        const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;
        const MANTISSA_MASK: u64 = 0x000f_ffff_ffff_ffff;

        const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;
        const CANONICAL_ZERO_BITS: u64 = 0x0;

        if self.is_nan() {
            CANONICAL_NAN_BITS
        }
        else {
            let (mantissa, exponent, sign) = self.integer_decode();
            if mantissa == 0 {
                CANONICAL_ZERO_BITS
            }
            else {
                let exponent = u64::from(unsafe { mem::transmute::<i16, u16>(exponent) });
                let sign = if sign > 0 { 1u64 } else { 0u64 };
                (mantissa & MANTISSA_MASK)
                    | ((exponent << 52) & EXPONENT_MASK)
                    | ((sign << 63) & SIGN_MASK)
            }
        }
    }
}

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
    const MAX_FINITE: Self;
    const MIN_FINITE: Self;
    const MIN_POSITIVE_NORMAL: Self;
    const EPSILON: Self;

    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;

    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;

    fn integer_decode(self) -> (u64, i16, i8);
}

impl Encoding for f32 {
    const MAX_FINITE: Self = f32::MAX;
    const MIN_FINITE: Self = f32::MIN;
    const MIN_POSITIVE_NORMAL: Self = f32::MIN_POSITIVE;
    const EPSILON: Self = f32::EPSILON;

    fn classify(self) -> FpCategory {
        self.classify()
    }

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn is_sign_positive(self) -> bool {
        Self::is_sign_positive(self)
    }

    fn is_sign_negative(self) -> bool {
        Self::is_sign_negative(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.to_bits();
        let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
        let exponent: i16 = ((bits >> 23) & 0xff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x7f_ffff) << 1
        }
        else {
            (bits & 0x7f_ffff) | 0x80_0000
        };
        (mantissa as u64, exponent - (127 + 23), sign)
    }
}

impl Encoding for f64 {
    const MAX_FINITE: Self = f64::MAX;
    const MIN_FINITE: Self = f64::MIN;
    const MIN_POSITIVE_NORMAL: Self = f64::MIN_POSITIVE;
    const EPSILON: Self = f64::EPSILON;

    fn classify(self) -> FpCategory {
        self.classify()
    }

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn is_sign_positive(self) -> bool {
        Self::is_sign_positive(self)
    }

    fn is_sign_negative(self) -> bool {
        Self::is_sign_negative(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xf_ffff_ffff_ffff) << 1
        }
        else {
            (bits & 0xf_ffff_ffff_ffff) | 0x10_0000_0000_0000
        };
        (mantissa, exponent - (1023 + 52), sign)
    }
}

/// Types that can represent real numbers.
///
/// Provides values and operations that generally apply to real numbers. As
/// such, this trait is implemented by types using floating-point
/// representations, but this trait is a general numeric trait and can be
/// implemented by other numeric types as well.
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

/// Primitive floating-point types.
pub trait Primitive {}

/// Implements floating-point traits for primitive types.
macro_rules! impl_primitive {
    (primitive => $t:ident) => {
        impl Infinite for $t {
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;

            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }

        impl Nan for $t {
            const NAN: Self = <$t>::NAN;

            fn is_nan(self) -> bool {
                self.is_nan()
            }
        }

        impl Primitive for $t {}

        impl Real for $t {
            // TODO: The propagation from a constant in a module requires that
            //       this macro accept an `ident` token rather than a `ty`
            //       token. Use `ty` if these constants become associated
            //       constants of the primitive types.
            const E: Self = core::$t::consts::E;
            const PI: Self = core::$t::consts::PI;
            const FRAC_1_PI: Self = core::$t::consts::FRAC_1_PI;
            const FRAC_2_PI: Self = core::$t::consts::FRAC_2_PI;
            const FRAC_2_SQRT_PI: Self = core::$t::consts::FRAC_2_SQRT_PI;
            const FRAC_PI_2: Self = core::$t::consts::FRAC_PI_2;
            const FRAC_PI_3: Self = core::$t::consts::FRAC_PI_3;
            const FRAC_PI_4: Self = core::$t::consts::FRAC_PI_4;
            const FRAC_PI_6: Self = core::$t::consts::FRAC_PI_6;
            const FRAC_PI_8: Self = core::$t::consts::FRAC_PI_8;
            const SQRT_2: Self = core::$t::consts::SQRT_2;
            const FRAC_1_SQRT_2: Self = core::$t::consts::FRAC_1_SQRT_2;
            const LN_2: Self = core::$t::consts::LN_2;
            const LN_10: Self = core::$t::consts::LN_10;
            const LOG2_E: Self = core::$t::consts::LOG2_E;
            const LOG10_E: Self = core::$t::consts::LOG10_E;

            fn floor(self) -> Self {
                <$t>::floor(self)
            }

            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }

            fn round(self) -> Self {
                <$t>::round(self)
            }

            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }

            fn fract(self) -> Self {
                <$t>::fract(self)
            }

            fn recip(self) -> Self {
                <$t>::recip(self)
            }

            #[cfg(feature = "std")]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$t>::mul_add(self, a, b)
            }

            #[cfg(feature = "std")]
            fn powi(self, n: i32) -> Self {
                <$t>::powi(self, n)
            }

            #[cfg(feature = "std")]
            fn powf(self, n: Self) -> Self {
                <$t>::powf(self, n)
            }

            #[cfg(feature = "std")]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }

            #[cfg(feature = "std")]
            fn cbrt(self) -> Self {
                <$t>::cbrt(self)
            }

            #[cfg(feature = "std")]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }

            #[cfg(feature = "std")]
            fn exp2(self) -> Self {
                <$t>::exp2(self)
            }

            #[cfg(feature = "std")]
            fn exp_m1(self) -> Self {
                <$t>::exp_m1(self)
            }

            #[cfg(feature = "std")]
            fn log(self, base: Self) -> Self {
                <$t>::log(self, base)
            }

            #[cfg(feature = "std")]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            #[cfg(feature = "std")]
            fn log2(self) -> Self {
                <$t>::log2(self)
            }

            #[cfg(feature = "std")]
            fn log10(self) -> Self {
                <$t>::log10(self)
            }

            #[cfg(feature = "std")]
            fn ln_1p(self) -> Self {
                <$t>::ln_1p(self)
            }

            #[cfg(feature = "std")]
            fn hypot(self, other: Self) -> Self {
                <$t>::hypot(self, other)
            }

            #[cfg(feature = "std")]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }

            #[cfg(feature = "std")]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }

            #[cfg(feature = "std")]
            fn tan(self) -> Self {
                <$t>::tan(self)
            }

            #[cfg(feature = "std")]
            fn asin(self) -> Self {
                <$t>::asin(self)
            }

            #[cfg(feature = "std")]
            fn acos(self) -> Self {
                <$t>::acos(self)
            }

            #[cfg(feature = "std")]
            fn atan(self) -> Self {
                <$t>::atan(self)
            }

            #[cfg(feature = "std")]
            fn atan2(self, other: Self) -> Self {
                <$t>::atan2(self, other)
            }

            #[cfg(feature = "std")]
            fn sin_cos(self) -> (Self, Self) {
                <$t>::sin_cos(self)
            }

            #[cfg(feature = "std")]
            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }

            #[cfg(feature = "std")]
            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }

            #[cfg(feature = "std")]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }

            #[cfg(feature = "std")]
            fn asinh(self) -> Self {
                <$t>::asinh(self)
            }

            #[cfg(feature = "std")]
            fn acosh(self) -> Self {
                <$t>::acosh(self)
            }

            #[cfg(feature = "std")]
            fn atanh(self) -> Self {
                <$t>::atanh(self)
            }
        }
    };
}
impl_primitive!(primitive => f32);
impl_primitive!(primitive => f64);
