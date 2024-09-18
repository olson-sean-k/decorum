//! Making floating-point behave: total ordering, equivalence, hashing, constraints, error
//! handling, and more for IEEE 754 floating-point representations.
//!
//! Decorum provides APIs for extending IEEE 754 floating-point. This is primarily accomplished
//! with [proxy types][`proxy`] that wrap primitive floating-point types and compose
//! [constraints][`constraint`] and [divergence] to configure behavior. Decorum also provides
//! numerous traits describing real numerics and IEEE 754 encoding.
//!
//! Decorum requires Rust 1.65.0 or higher.
//!
//! # Proxy Types
//!
//! [`Proxy`] types wrap primitive floating-point types and constrain the set of values that they
//! can represent. These types use the same representation as primitives and in many cases can be
//! used as drop-in replacements, in particular the [`Total`] type. [`Proxy`] supports numerous
//! traits and APIs (including third-party integrations) and always provides a complete API for
//! real numbers.
//!
//! Proxy types and their [constraints][`constraint`] operate on three subsets of IEEE 754
//! floating-point values:
//!
//! | Subset       | Example Member |
//! |--------------|----------------|
//! | real numbers | `3.1459`       |
//! | infinities   | `+Inf`         |
//! | not-a-number | `NaN`          |
//!
//! These subsets are reflected throughout APIs, in particular in traits concerning IEEE 754
//! encoding and constraints. [`Constraint`]s describe which subsets are members of a proxy type
//! and are composed with a [divergence], which further describes the outputs and error behavior of
//! operations.
//!
//! These types can be configured to, for example, cause a panic in debugging builds whenever a
//! `NaN` is encountered or enable structured error handling of extended real numbers where any
//! `NaN` is interpreted as undefined and yields an explicit error value.
//!
//! Proxy types and their components are provided by the [`proxy`], [`constraint`], and
//! [`divergence`] modules. Numerous type definitions are also provided in the crate root:
//!
//! | Type Definition | Subsets                                |
//! |-----------------|----------------------------------------|
//! | [`Total`]       | real numbers, infinities, not-a-number |
//! | [`NotNan`]      | real numbers, infinities               |
//! | [`Finite`]      | real numbers                           |
//!
//! # Equivalence and Ordering
//!
//! The [`cmp`] module provides APIs for comparing floating-point representations as well as other
//! partially ordered types. For example, it provides traits for intrinic comparisons of partially
//! ordered types that propagate `NaN`s when used with floating-point representations. It also
//! defines a non-standard total ordering for complete floating-point types:
//!
//! $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
//!
//! Note that all `NaN` representations are considered equivalent in this relation. The [`Total`]
//! proxy type uses this ordering.
//!
//! # Hashing
//!
//! The [`hash`] module provides traits for hashing floating-point representations. Hashing is
//! consistent with the total ordering defined by the [`cmp`] module. Proxy types implement the
//! standard [`Hash`] trait via this module and it also provides functions for hashing primitive
//! floating-point types.
//!
//! # Numeric Traits
//!
//! The [`real`] module provides traits that describe real numbers and their approximation via
//! floating-point representations. These traits describe the codomain of operations and respect
//! the branching behavior of such functions. For example, many functions over real numbers have a
//! range that includes non-reals (such as undefined). Additionally, these traits feature ergonomic
//! improvements on similar traits in the crate ecosystem.
//!
//! # Expressions
//!
//! [`Expression`] types represent the output of computations using constrained [`Proxy`] types.
//! They provide structured types that directly encode divergence (errors) as values. Unlike other
//! branch types, [`Expression`] also supports the same numeric operations as [`Proxy`] types, so
//! they can be used fluently in numeric expressions without matching or trying.
//!
//! ```rust
//! use decorum::constraint::IsReal;
//! use decorum::divergence::OrError;
//! use decorum::proxy::{OutputOf, Proxy};
//!
//! type Real = Proxy<f64, IsReal<OrError>>;
//! type Expr = OutputOf<Real>;
//!
//! fn f(x: Real, y: Real) -> Expr {
//!     let z = x + y;
//!     z / x
//! }
//!
//! let z = f(Real::assert(3.0), Real::assert(4.0));
//! assert!(z.is_defined());
//! ```
//!
//! For finer control, the [`try_expression`] macro can be used to differentiate between
//! expressions and defined results. When using a nightly Rust toolchain, the `unstable` Cargo
//! feature also implements the unstable (at time of writing) [`Try`] trait for [`Expression`] so
//! that the try operator `?` may be used instead.
//!
//! ```rust,ignore
//! use decorum::constraint::IsReal;
//! use decorum::divergence::OrError;
//! use decorum::proxy::{OutputOf, Proxy};
//! use decorum::real::UnaryReal;
//!
//! type Real = Proxy<f64, IsReal<OrError>>;
//! type Expr = OutputOf<Real>;
//!
//! # fn fallible() -> Expr {
//! fn f(x: Real, y: Real) -> Expr {
//!     x / y
//! }
//!
//! let z = f(Real::PI, Real::ONE)?; // OK: `z` is `Real`.
//! let w = f(Real::PI, Real::ZERO)?; // Error: this returns `Expression::Undefined`.
//! // ...
//! # f(Real::PI, Real::ONE)
//! # }
//! ```
//!
//! [`cmp`]: crate::cmp
//! [`constraint`]: crate::constraint
//! [`Constraint`]: crate::constraint::Constraint
//! [`divergence`]: crate::divergence
//! [`Expression`]: crate::expression::Expression
//! [`Finite`]: crate::Finite
//! [`hash`]: crate::hash
//! [`Hash`]: core::hash::Hash
//! [`NotNan`]: crate::NotNan
//! [`proxy`]: crate::proxy
//! [`Proxy`]: crate::proxy::Proxy
//! [`real`]: crate::real
//! [`Total`]: crate::Total
//! [`Try`]: core::ops::Try
//! [`try_expression`]: crate::try_expression

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum-favicon.ico"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.svg?sanitize=true"
)]
#![no_std]
#![cfg_attr(all(nightly, feature = "unstable"), feature(try_trait_v2))]

#[cfg(feature = "std")]
extern crate std;

pub mod cmp;
pub mod constraint;
pub mod divergence;
pub mod expression;
pub mod hash;
pub mod proxy;
pub mod real;

#[cfg(not(feature = "std"))]
pub(crate) use num_traits::float::FloatCore as ForeignFloat;
#[cfg(feature = "std")]
pub(crate) use num_traits::real::Real as ForeignReal;
#[cfg(feature = "std")]
pub(crate) use num_traits::Float as ForeignFloat;

use core::mem;
use core::num::FpCategory;
use num_traits::{PrimInt, Unsigned};

use crate::cmp::IntrinsicOrd;
use crate::constraint::{IsExtendedReal, IsFloat, IsReal};
use crate::divergence::OrPanic;
use crate::proxy::Proxy;
use crate::real::{BinaryReal, Function, Real, Sign, UnaryReal};

mod sealed {
    use core::convert::Infallible;

    pub trait Sealed {}

    impl Sealed for Infallible {}
}
use crate::sealed::Sealed;

/// IEEE 754 floating-point representation with non-standard total ordering and hashing.
///
/// This [`Proxy`] type applies no constraints and no divergence. It can trivially replace
/// primitive floating point types and implements the standard [`Eq`] and [`Ord`] traits. See the
/// [`cmp`] module for more details about these relations.
///
/// [`cmp`]: crate::cmp
/// [`Eq`]: core::cmp::Eq
/// [`Ord`]: core::cmp::Ord
/// [`Proxy`]: crate::proxy::Proxy
pub type Total<T> = Proxy<T, IsFloat>;

/// IEEE 754 floating-point representation that cannot be `NaN`.
///
/// This [`Proxy`] type applies the [`IsExtendedReal`] constraint and [diverges][`divergence`] if a
/// `NaN` value is encountered. **The default divergence of this definition is [`OrPanic`], which
/// panics when the constraint is violated.**
///
/// Like [`Total`], `NotNan` defines equivalence and total ordering, but need not consider `NaN`
/// and so uses only standard IEEE 754 floating-point semantics.
///
/// [`divergence`]: crate::divergence
/// [`IsExtendedReal`]: crate::constraint::IsExtendedReal
/// [`OrPanic`]: crate::divergence::OrPanic
/// [`Proxy`]: crate::proxy::Proxy
/// [`Total`]: crate::Total
pub type NotNan<T, D = OrPanic> = Proxy<T, IsExtendedReal<D>>;

/// 32-bit IEEE 754 floating-point representation that cannot be `NaN`.
pub type N32<D = OrPanic> = NotNan<f32, D>;
/// 64-bit IEEE 754 floating-point representation that cannot be `NaN`.
pub type N64<D = OrPanic> = NotNan<f64, D>;

/// IEEE 754 floating-point representation that must be a real number.
pub type Finite<T, D = OrPanic> = Proxy<T, IsReal<D>>;

/// 32-bit IEEE 754 floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if "F" were used, then
/// this name would be too similar to `f32`.
pub type R32<D = OrPanic> = Finite<f32, D>;
/// 64-bit IEEE 754 floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if "F" were used, then
/// this name would be too similar to `f64`.
pub type R64<D = OrPanic> = Finite<f64, D>;

// TODO: Inverse the relationship between `Encoding` and `ToCanonicalBits` such that `Encoding`
//       requires `ToCanonicalBits`.
/// Converts IEEE 754 floating-point values to a canonicalized form.
pub trait ToCanonicalBits: Encoding {
    type Bits: PrimInt + Unsigned;

    /// Conversion to a canonical representation.
    ///
    /// This function collapses representations for real numbers, zeroes, infinities, and `NaN`s
    /// into a canonical form such that every semantic value has a unique representation as
    /// canonical bits.
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
                let sign = u64::from(sign > 0);
                (mantissa & MANTISSA_MASK)
                    | ((exponent << 52) & EXPONENT_MASK)
                    | ((sign << 63) & SIGN_MASK)
            }
        }
    }
}

/// IEEE 754 floating-point representations that expose infinities (`-INF` and `+INF`).
// This trait is implemented by trivial `Copy` types.
#[allow(clippy::wrong_self_convention)]
pub trait Infinite: Sized {
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

/// IEEE 754 floating-point representations that expose `NaN`s.
// This trait is implemented by trivial `Copy` types.
#[allow(clippy::wrong_self_convention)]
pub trait Nan: Sized {
    /// A representation of `NaN`.
    ///
    /// For primitive floating-point types, `NaN` is incomparable. Therefore, prefer the `is_nan`
    /// predicate over direct comparisons with `NaN`.
    const NAN: Self;

    fn is_nan(self) -> bool;
}

/// IEEE 754 floating-point representations that expose general encoding.
///
/// Provides values and operations that describe the encoding of an IEEE 754 floating-point value.
/// The specific semantic values for infinities and `NaN`s are described by independent traits.
// This trait is implemented by trivial `Copy` types.
#[allow(clippy::wrong_self_convention)]
pub trait Encoding: Sized {
    const MAX_FINITE: Self;
    const MIN_FINITE: Self;
    const MIN_POSITIVE_NORMAL: Self;
    const EPSILON: Self;

    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;

    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;
    #[cfg(feature = "std")]
    fn signum(self) -> Self;

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

    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        Self::signum(self)
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

    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        Self::signum(self)
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

/// IEEE 754 floating-point representations.
///
/// Types that implement this trait are represented using IEEE 754 encoding **and directly expose
/// the complete details of that encoding**, including infinities, `NaN`s, and operations on real
/// numbers.
pub trait Float: Encoding + Infinite + IntrinsicOrd + Nan + Real<Codomain = Self> {}

impl<T> Float for T where T: Encoding + Infinite + IntrinsicOrd + Nan + Real<Codomain = T> {}

/// A primitive IEEE 754 floating-point type.
pub trait Primitive: Copy + Sealed {}

// TODO: Remove this. Of course.
fn _sanity() {
    use crate::real::FloatEndoreal;

    type Real = Proxy<f64, IsReal<OrPanic>>;

    fn f<T>(x: T) -> T
    where
        T: FloatEndoreal<f64>,
    {
        -x
    }

    fn g<T, U>(x: T, y: U) -> T
    where
        T: BinaryReal<U> + FloatEndoreal<f64>,
    {
        (x + T::ONE) * y
    }

    fn h<T>(x: T, y: T) -> T
    where
        T: FloatEndoreal<f64>,
    {
        x + y
    }

    let x = Real::ONE;
    let y = g(f(x), 2.0);
    let z = h(y, Real::assert(1.0));
    let _ = f(y + z);
}

macro_rules! with_primitives {
    ($f:ident) => {
        $f!(primitive => f32);
        $f!(primitive => f64);
    }
}
pub(crate) use with_primitives;

macro_rules! with_binary_operations {
    ($f:ident) => {
        $f!(operation => Add::add);
        $f!(operation => Div::div);
        $f!(operation => Mul::mul);
        $f!(operation => Rem::rem);
        $f!(operation => Sub::sub);
    };
}
pub(crate) use with_binary_operations;

/// Implements floating-point traits for primitive types.
macro_rules! impl_primitive {
    () => {
        with_primitives!(impl_primitive);
    };
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

        impl Function for $t {
            type Codomain = $t;
        }

        impl Sealed for $t {}

        impl UnaryReal for $t {
            // TODO: The propagation from a constant in a module requires that this macro accept an
            //       `ident` token rather than a `ty` token. Use `ty` if these constants become
            //       associated constants of the primitive types.
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
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

            fn is_zero(self) -> bool {
                self == Self::ZERO
            }

            fn is_one(self) -> bool {
                self == Self::ONE
            }

            fn sign(self) -> Sign {
                if self.is_nan() || self.is_zero() {
                    Sign::Zero
                }
                else if self > 0.0 {
                    Sign::Positive
                }
                else {
                    Sign::Negative
                }
            }

            #[cfg(feature = "std")]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }

            #[cfg(feature = "std")]
            fn floor(self) -> Self {
                <$t>::floor(self)
            }

            #[cfg(feature = "std")]
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }

            #[cfg(feature = "std")]
            fn round(self) -> Self {
                <$t>::round(self)
            }

            #[cfg(feature = "std")]
            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }

            #[cfg(feature = "std")]
            fn fract(self) -> Self {
                <$t>::fract(self)
            }

            fn recip(self) -> Self::Codomain {
                <$t>::recip(self)
            }

            #[cfg(feature = "std")]
            fn powi(self, n: i32) -> Self::Codomain {
                <$t>::powi(self, n)
            }

            #[cfg(feature = "std")]
            fn sqrt(self) -> Self::Codomain {
                <$t>::sqrt(self)
            }

            #[cfg(feature = "std")]
            fn cbrt(self) -> Self {
                <$t>::cbrt(self)
            }

            #[cfg(feature = "std")]
            fn exp(self) -> Self::Codomain {
                <$t>::exp(self)
            }

            #[cfg(feature = "std")]
            fn exp2(self) -> Self::Codomain {
                <$t>::exp2(self)
            }

            #[cfg(feature = "std")]
            fn exp_m1(self) -> Self::Codomain {
                <$t>::exp_m1(self)
            }

            #[cfg(feature = "std")]
            fn ln(self) -> Self::Codomain {
                <$t>::ln(self)
            }

            #[cfg(feature = "std")]
            fn log2(self) -> Self::Codomain {
                <$t>::log2(self)
            }

            #[cfg(feature = "std")]
            fn log10(self) -> Self::Codomain {
                <$t>::log10(self)
            }

            #[cfg(feature = "std")]
            fn ln_1p(self) -> Self::Codomain {
                <$t>::ln_1p(self)
            }

            #[cfg(feature = "std")]
            fn to_degrees(self) -> Self::Codomain {
                <$t>::to_degrees(self)
            }

            #[cfg(feature = "std")]
            fn to_radians(self) -> Self {
                <$t>::to_radians(self)
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
            fn tan(self) -> Self::Codomain {
                <$t>::tan(self)
            }

            #[cfg(feature = "std")]
            fn asin(self) -> Self::Codomain {
                <$t>::asin(self)
            }

            #[cfg(feature = "std")]
            fn acos(self) -> Self::Codomain {
                <$t>::acos(self)
            }

            #[cfg(feature = "std")]
            fn atan(self) -> Self {
                <$t>::atan(self)
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
            fn asinh(self) -> Self::Codomain {
                <$t>::asinh(self)
            }

            #[cfg(feature = "std")]
            fn acosh(self) -> Self::Codomain {
                <$t>::acosh(self)
            }

            #[cfg(feature = "std")]
            fn atanh(self) -> Self::Codomain {
                <$t>::atanh(self)
            }
        }

        impl BinaryReal<$t> for $t {
            #[cfg(feature = "std")]
            fn div_euclid(self, n: Self) -> Self::Codomain {
                <$t>::div_euclid(self, n)
            }

            #[cfg(feature = "std")]
            fn rem_euclid(self, n: Self) -> Self::Codomain {
                <$t>::rem_euclid(self, n)
            }

            #[cfg(feature = "std")]
            fn pow(self, n: Self) -> Self::Codomain {
                <$t>::powf(self, n)
            }

            #[cfg(feature = "std")]
            fn log(self, base: Self) -> Self::Codomain {
                <$t>::log(self, base)
            }

            #[cfg(feature = "std")]
            fn hypot(self, other: Self) -> Self::Codomain {
                <$t>::hypot(self, other)
            }

            #[cfg(feature = "std")]
            fn atan2(self, other: Self) -> Self {
                <$t>::atan2(self, other)
            }
        }
    };
}
impl_primitive!();
