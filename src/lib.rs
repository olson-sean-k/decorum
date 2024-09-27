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
//! | Type Definition  | Subsets                                |
//! |------------------|----------------------------------------|
//! | [`Total`]        | real numbers, infinities, not-a-number |
//! | [`ExtendedReal`] | real numbers, infinities               |
//! | [`Real`]         | real numbers                           |
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
//! use decorum::real::UnaryRealFunction;
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
//! [`ExtendedReal`]: crate::ExtendedReal
//! [`hash`]: crate::hash
//! [`Hash`]: core::hash::Hash
//! [`proxy`]: crate::proxy
//! [`Proxy`]: crate::proxy::Proxy
//! [`real`]: crate::real
//! [`Real`]: crate::Real
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

use core::hash::Hash;
use core::num::FpCategory;

use crate::cmp::IntrinsicOrd;
use crate::constraint::{IsExtendedReal, IsFloat, IsReal};
use crate::divergence::OrPanic;
use crate::proxy::Proxy;
use crate::real::{
    BinaryRealFunction, Endofunction, Function, RealFunction, Sign, UnaryRealFunction,
};

mod sealed {
    use core::convert::Infallible;
    use core::fmt::{self, Formatter};

    pub trait Sealed {}

    impl Sealed for Infallible {}

    pub trait StaticDebug {
        fn fmt(formatter: &mut Formatter<'_>) -> fmt::Result;
    }
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

/// IEEE 754 floating-point representation that must be an extended real.
pub type ExtendedReal<T, D = OrPanic> = Proxy<T, IsExtendedReal<D>>;

/// IEEE 754 floating-point representation that must not be `NaN`.
pub type NotNan<T, D = OrPanic> = ExtendedReal<T, D>;

/// 32-bit IEEE 754 floating-point representation that must be an extended real (not `NaN`).
pub type E32<D = OrPanic> = ExtendedReal<f32, D>;
/// 64-bit IEEE 754 floating-point representation that must be an extended real (not `NaN`).
pub type E64<D = OrPanic> = ExtendedReal<f64, D>;

/// IEEE 754 floating-point representation that must be a real number.
pub type Real<T, D = OrPanic> = Proxy<T, IsReal<D>>;

/// 32-bit IEEE 754 floating-point representation that must be a real number.
pub type R32<D = OrPanic> = Real<f32, D>;
/// 64-bit IEEE 754 floating-point representation that must be a real number.
pub type R64<D = OrPanic> = Real<f64, D>;

/// Converts IEEE 754 floating-point values to a canonicalized form.
pub trait ToCanonical: BaseEncoding + Copy {
    type Canonical: Copy + Eq + Hash;

    /// Conversion to a canonical representation.
    ///
    /// This function collapses real numbers, zeroes, infinities, and `NaN`s into a canonical form
    /// such that every semantic value has a unique representation with an equivalence relation.
    fn to_canonical(self) -> Self::Canonical;
}

// TODO: Implement this differently for differently sized primitive types.
impl<T> ToCanonical for T
where
    T: Primitive,
{
    type Canonical = u64;

    fn to_canonical(self) -> Self::Canonical {
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
                let exponent = u64::from(exponent as u16);
                let sign = u64::from(sign > 0);
                (mantissa & MANTISSA_MASK)
                    | ((exponent << 52) & EXPONENT_MASK)
                    | ((sign << 63) & SIGN_MASK)
            }
        }
    }
}

/// A type with an IEEE 754 floating-point representation that exposes its basic encoding.
///
/// `BaseEncoding` types have a floating-point representation ([`binaryN`]), **but may not support
/// nor expose other elements of the specification**. This trait describes the most basic
/// non-computational elements of the encoding and does not specify the inhabitants of a type.
///
/// [`binaryN`]: https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
pub trait BaseEncoding: Copy {
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

impl BaseEncoding for f32 {
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

impl BaseEncoding for f64 {
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

/// A type with an IEEE 754 floating-point representation that supports infinities.
///
/// `InfinityEncoding` types have `-INF` and `+INF` inhabitants.
pub trait InfinityEncoding: Copy {
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

/// A type with an IEEE 754 floating-point representation that supports `NaN`s.
///
/// `NanEncoding` types have `NaN` inhabitants.
pub trait NanEncoding: Copy {
    /// The type of the arbitrary `Nan` representation [`NAN`].
    ///
    /// This may be an intermediate type other than `Self`. In particular, primitive IEEE 754
    /// floating-point types represent `NaN` with the [`Nan`] type, which is incomparable and must
    /// first be converted into its primitive types.
    ///
    /// For proxy types, which are totally ordered, this type satisfies the bound `Eq + Ord`.
    ///
    /// [`Nan`]: crate::Nan
    /// [`NAN`]: crate::NanEncoding::NAN
    type Nan;

    /// An arbitrary representation of `NaN`.
    const NAN: Self::Nan;

    fn is_nan(self) -> bool;
}

/// A primitive IEEE 754 floating-point type.
pub trait Primitive:
    BaseEncoding
    + Copy
    + Endofunction
    + InfinityEncoding
    + IntrinsicOrd<Undefined = Self>
    + NanEncoding<Nan = Nan<Self>>
    + PartialEq
    + PartialOrd
    + RealFunction
    + Sealed
{
}

/// An incomparable primitive IEEE 754 floating-point `NaN`.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Nan<T>
where
    T: Primitive,
{
    inner: T,
}

impl<T> Nan<T>
where
    T: Primitive,
{
    pub const fn into_inner(self) -> T {
        self.inner
    }
}

impl From<Nan<f32>> for f32 {
    fn from(nan: Nan<f32>) -> Self {
        nan.into_inner()
    }
}

impl From<Nan<f64>> for f64 {
    fn from(nan: Nan<f64>) -> Self {
        nan.into_inner()
    }
}

// TODO: Remove this. Of course.
fn _sanity() {
    use crate::real::FloatFunction;

    type Real = Proxy<f64, IsReal<OrPanic>>;

    fn f<T>(x: T) -> T
    where
        T: FloatFunction<f64>,
    {
        -x
    }

    fn g<T, U>(x: T, y: U) -> T
    where
        T: BinaryRealFunction<U> + Endofunction + FloatFunction<f64>,
    {
        (x + T::ONE) * y
    }

    fn h<T>(x: T, y: T) -> T
    where
        T: Endofunction + FloatFunction<f64>,
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

/// Implements real number and floating-point traits for primitive types.
macro_rules! impl_primitive {
    () => {
        with_primitives!(impl_primitive);
    };
    (primitive => $t:ident) => {
        impl BinaryRealFunction<$t> for $t {
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

        impl Function for $t {
            type Codomain = $t;
        }

        impl InfinityEncoding for $t {
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;

            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }

        impl NanEncoding for $t {
            type Nan = Nan<$t>;

            const NAN: Self::Nan = Nan { inner: <$t>::NAN };

            fn is_nan(self) -> bool {
                self.is_nan()
            }
        }

        impl Primitive for $t {}

        impl Sealed for $t {}

        impl UnaryRealFunction for $t {
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
    };
}
impl_primitive!();
