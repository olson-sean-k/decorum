//! Proxy types that wrap primitive floating-point types and apply constraints
//! and a total ordering.

#[cfg(feature = "approx")]
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display, Formatter, LowerExp, UpperExp};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::num::FpCategory;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::str::FromStr;
use num_traits::{
    Bounded, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};
#[cfg(feature = "serialize-serde")]
use serde_derive::{Deserialize, Serialize};

use crate::canonical::ToCanonicalBits;
use crate::cmp::{self, FloatEq, FloatOrd, IntrinsicOrd};
use crate::constraint::{Constraint, InfiniteClass, Member, NanClass, SubsetOf, SupersetOf};
use crate::hash::FloatHash;
use crate::primitive::Primitive;
use crate::{Encoding, Finite, Float, ForeignFloat, Infinite, Nan, NotNan, Real, Total};
#[cfg(feature = "std")]
use crate::{ForeignReal, N32, N64, R32, R64};

/// Floating-point proxy that provides a total ordering, equivalence, hashing,
/// and constraints.
///
/// `ConstrainedFloat` wraps primitive floating-point types and provides
/// implementations for numeric traits using a total ordering, including `Ord`,
/// `Eq`, and `Hash`. `ConstrainedFloat` supports various constraints on the
/// class of values that may be represented and panics if these constraints are
/// violated.
///
/// This type is re-exported but should not (and cannot) be used directly. Use
/// the type aliases `Total`, `NotNan`, and `Finite` instead.
///
/// # Total Ordering
///
/// All proxy types use the following total ordering:
///
/// $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
///
/// See the `cmp` module for a description of the total ordering used to
/// implement `Ord` and `Eq`.
///
/// # Constraints
///
/// Constraints restrict the set of values that a proxy may take by disallowing
/// certain classes or subsets of those values. If a constraint is violated
/// (because a proxy type would need to take a value it disallows), the
/// operation panics.
///
/// Constraints may disallow two broad classes of floating-point values:
/// infinities and `NaN`s. Constraints are exposed by the `Total`, `NotNan`, and
/// `Finite` type definitions. Note that `Total` uses a unit constraint, which
/// enforces no constraints at all and never panics.
#[cfg_attr(feature = "serialize-serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ConstrainedFloat<T, P> {
    value: T,
    phantom: PhantomData<P>,
}

impl<T, P> ConstrainedFloat<T, P> {
    const fn from_inner_unchecked(value: T) -> Self {
        ConstrainedFloat {
            value,
            phantom: PhantomData,
        }
    }
}

impl<T, P> ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    /// Converts a primitive floating-point value into a proxy.
    ///
    /// The same behavior is provided by an implementation of the `From` trait.
    ///
    /// # Panics
    ///
    /// This conversion and the implementation of the `From` trait will panic
    /// if the primitive floating-point value violates the constraints of the
    /// proxy.
    ///
    /// # Examples
    ///
    /// Converting primitive floating-point values into proxies:
    ///
    /// ```rust
    /// use decorum::R64;
    ///
    /// fn f(x: R64) -> R64 {
    ///     x * 2.0
    /// }
    ///
    /// // Conversion using `from_inner`.
    /// let y = f(R64::from_inner(2.0));
    /// // Conversion using `From`/`Into`.
    /// let z = f(2.0.into());
    /// ```
    ///
    /// Performing a conversion that panics:
    ///
    /// ```rust,should_panic
    /// use decorum::R64;
    ///
    /// // `R64` does not allow `NaN`s, but `0.0 / 0.0` produces a `NaN`.
    /// let x = R64::from_inner(0.0 / 0.0); // Panics.
    /// ```
    pub fn from_inner(value: T) -> Self {
        Self::try_from_inner(value).expect("floating-point constraint violated")
    }

    /// Converts a proxy into a primitive floating-point value.
    ///
    /// # Examples
    ///
    /// Converting a proxy into a primitive floating-point value:
    ///
    /// ```rust
    /// use decorum::R64;
    ///
    /// fn f() -> R64 {
    /// #    0.0.into()
    ///     // ...
    /// }
    ///
    /// let x: f64 = f().into_inner();
    /// ```
    pub fn into_inner(self) -> T {
        let ConstrainedFloat { value, .. } = self;
        value
    }

    /// Converts a proxy into another proxy that is capable of representing a
    /// superset of the values that are members of its constraint.
    ///
    /// # Examples
    ///
    /// Converting between compatible proxy types:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate num;
    /// use decorum::{N64, R64};
    /// use num::Zero;
    ///
    /// let x = R64::zero();
    /// let y = N64::from_subset(x);
    /// ```
    pub fn from_subset<Q>(other: ConstrainedFloat<T, Q>) -> Self
    where
        Q: Constraint<T> + SubsetOf<P>,
    {
        Self::from_inner_unchecked(other.into_inner())
    }

    /// Converts a proxy into another proxy that is capable of representing a
    /// superset of the values that are members of its constraint.
    ///
    /// # Examples
    ///
    /// Converting between compatible proxy types:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate num;
    /// use decorum::{N64, R64};
    /// use num::Zero;
    ///
    /// let x = R64::zero();
    /// let y: N64 = x.into_superset();
    /// ```
    pub fn into_superset<Q>(self) -> ConstrainedFloat<T, Q>
    where
        Q: Constraint<T> + SupersetOf<P>,
    {
        ConstrainedFloat::from_inner_unchecked(self.into_inner())
    }

    fn try_from_inner(value: T) -> Result<Self, ()> {
        P::filter(value)
            .map(|value| ConstrainedFloat {
                value,
                phantom: PhantomData,
            })
            .ok_or(())
    }

    fn map<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        ConstrainedFloat::from_inner(f(self.into_inner()))
    }

    fn map_unchecked<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        ConstrainedFloat::from_inner_unchecked(f(self.into_inner()))
    }

    fn zip_map<F>(self, other: Self, f: F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        ConstrainedFloat::from_inner(f(self.into_inner(), other.into_inner()))
    }
}

impl<T, P> AsRef<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> From<NotNan<T>> for Total<T>
where
    T: Float + Primitive,
{
    fn from(other: NotNan<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<T> From<Finite<T>> for Total<T>
where
    T: Float + Primitive,
{
    fn from(other: Finite<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<T> From<Finite<T>> for NotNan<T>
where
    T: Float + Primitive,
{
    fn from(other: Finite<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<T, P> From<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn from(value: T) -> Self {
        Self::from_inner(value)
    }
}

impl<P> From<ConstrainedFloat<f32, P>> for f32
where
    P: Constraint<f32>,
{
    fn from(value: ConstrainedFloat<f32, P>) -> Self {
        value.into_inner()
    }
}

impl<P> From<ConstrainedFloat<f64, P>> for f64
where
    P: Constraint<f64>,
{
    fn from(value: ConstrainedFloat<f64, P>) -> Self {
        value.into_inner()
    }
}

#[cfg(feature = "approx")]
impl<T, P> AbsDiffEq for ConstrainedFloat<T, P>
where
    T: AbsDiffEq<Epsilon = T> + Float + Primitive,
    P: Constraint<T>,
{
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon().into()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.into_inner()
            .abs_diff_eq(&other.into_inner(), epsilon.into_inner())
    }
}

impl<T, P> Add for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.zip_map(other, Add::add)
    }
}

impl<T, P> Add<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        self.map(|inner| inner + other)
    }
}

impl<T, P> AddAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<T, P> AddAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn add_assign(&mut self, other: T) {
        *self = self.map(|inner| inner + other);
    }
}

impl<T, P> Bounded for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn min_value() -> Self {
        Encoding::MIN
    }

    fn max_value() -> Self {
        Encoding::MAX
    }
}

impl<T> Debug for Finite<T>
where
    T: Debug + Float + Primitive,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Finite").field(self.as_ref()).finish()
    }
}

impl<T> Debug for NotNan<T>
where
    T: Debug + Float + Primitive,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NotNan").field(self.as_ref()).finish()
    }
}

impl<T> Debug for Total<T>
where
    T: Debug + Float + Primitive,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Total").field(self.as_ref()).finish()
    }
}

impl<T, P> Default for ConstrainedFloat<T, P>
where
    T: Default + Float + Primitive,
    P: Constraint<T>,
{
    fn default() -> Self {
        T::default().into()
    }
}

impl<T, P> Display for ConstrainedFloat<T, P>
where
    T: Display + Float + Primitive,
    P: Constraint<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, P> Div for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        self.zip_map(other, Div::div)
    }
}

impl<T, P> Div<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        self.map(|inner| inner / other)
    }
}

impl<T, P> DivAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn div_assign(&mut self, other: Self) {
        *self = *self / other
    }
}

impl<T, P> DivAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn div_assign(&mut self, other: T) {
        *self = self.map(|inner| inner / other);
    }
}

impl<T, P> Encoding for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    const MAX: Self = ConstrainedFloat::from_inner_unchecked(T::MAX);
    const MIN: Self = ConstrainedFloat::from_inner_unchecked(T::MIN);
    const MIN_POSITIVE: Self = ConstrainedFloat::from_inner_unchecked(T::MIN_POSITIVE);
    const EPSILON: Self = ConstrainedFloat::from_inner_unchecked(T::EPSILON);

    fn classify(self) -> FpCategory {
        T::classify(self.into_inner())
    }

    fn is_normal(self) -> bool {
        T::is_normal(self.into_inner())
    }

    fn is_sign_positive(self) -> bool {
        self.into_inner().is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.into_inner().is_sign_negative()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        T::integer_decode(self.into_inner())
    }
}

impl<T, P> Eq for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
}

impl<T, P> ForeignFloat for ConstrainedFloat<T, P>
where
    T: Float + ForeignFloat + IntrinsicOrd + Primitive,
    P: Constraint<T> + Member<InfiniteClass> + Member<NanClass>,
{
    fn infinity() -> Self {
        Infinite::INFINITY
    }

    fn neg_infinity() -> Self {
        Infinite::NEG_INFINITY
    }

    fn is_infinite(self) -> bool {
        Infinite::is_infinite(self)
    }

    fn is_finite(self) -> bool {
        Infinite::is_finite(self)
    }

    fn nan() -> Self {
        Nan::NAN
    }

    fn is_nan(self) -> bool {
        Nan::is_nan(self)
    }

    fn max_value() -> Self {
        Encoding::MAX
    }

    fn min_value() -> Self {
        Encoding::MIN
    }

    fn min_positive_value() -> Self {
        Encoding::MIN_POSITIVE
    }

    fn epsilon() -> Self {
        Encoding::EPSILON
    }

    fn min(self, other: Self) -> Self {
        // Avoid panics by propagating `NaN`s for incomparable values.
        self.zip_map(other, cmp::min_or_undefined)
    }

    fn max(self, other: Self) -> Self {
        // Avoid panics by propagating `NaN`s for incomparable values.
        self.zip_map(other, cmp::max_or_undefined)
    }

    fn neg_zero() -> Self {
        Self::from_inner(T::neg_zero())
    }

    fn is_sign_positive(self) -> bool {
        Encoding::is_sign_positive(self.into_inner())
    }

    fn is_sign_negative(self) -> bool {
        Encoding::is_sign_negative(self.into_inner())
    }

    fn signum(self) -> Self {
        self.map(|inner| inner.signum())
    }

    fn abs(self) -> Self {
        self.map(|inner| inner.abs())
    }

    fn classify(self) -> FpCategory {
        Encoding::classify(self)
    }

    fn is_normal(self) -> bool {
        Encoding::is_normal(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        Encoding::integer_decode(self)
    }

    fn floor(self) -> Self {
        self.map(Real::floor)
    }

    fn ceil(self) -> Self {
        self.map(Real::ceil)
    }

    fn round(self) -> Self {
        self.map(Real::round)
    }

    fn trunc(self) -> Self {
        self.map(Real::trunc)
    }

    fn fract(self) -> Self {
        self.map(Real::fract)
    }

    fn recip(self) -> Self {
        self.map(Real::recip)
    }

    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Real::mul_add(self, a, b)
    }

    #[cfg(feature = "std")]
    fn abs_sub(self, other: Self) -> Self {
        self.zip_map(other, ForeignFloat::abs_sub)
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self {
        Real::powi(self, n)
    }

    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self {
        Real::powf(self, n)
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self {
        Real::sqrt(self)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        Real::cbrt(self)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self {
        Real::exp(self)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self {
        Real::exp2(self)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self {
        Real::exp_m1(self)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self {
        Real::log(self, base)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self {
        Real::ln(self)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self {
        Real::log2(self)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self {
        Real::log10(self)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self {
        Real::ln_1p(self)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self {
        Real::hypot(self, other)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        Real::sin(self)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        Real::cos(self)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self {
        Real::tan(self)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self {
        Real::asin(self)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self {
        Real::acos(self)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        Real::atan(self)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self {
        Real::atan2(self, other)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        Real::sin_cos(self)
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        Real::sinh(self)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        Real::cosh(self)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        Real::tanh(self)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self {
        Real::asinh(self)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self {
        Real::acosh(self)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self {
        Real::atanh(self)
    }

    #[cfg(not(feature = "std"))]
    fn to_degrees(self) -> Self {
        self.map(ForeignFloat::to_degrees)
    }

    #[cfg(not(feature = "std"))]
    fn to_radians(self) -> Self {
        self.map(ForeignFloat::to_radians)
    }
}

impl<T, P> FloatConst for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn E() -> Self {
        <Self as Real>::E
    }

    fn PI() -> Self {
        <Self as Real>::PI
    }

    fn SQRT_2() -> Self {
        <Self as Real>::SQRT_2
    }

    fn FRAC_1_PI() -> Self {
        <Self as Real>::FRAC_1_PI
    }

    fn FRAC_2_PI() -> Self {
        <Self as Real>::FRAC_2_PI
    }

    fn FRAC_1_SQRT_2() -> Self {
        <Self as Real>::FRAC_1_SQRT_2
    }

    fn FRAC_2_SQRT_PI() -> Self {
        <Self as Real>::FRAC_2_SQRT_PI
    }

    fn FRAC_PI_2() -> Self {
        <Self as Real>::FRAC_PI_2
    }

    fn FRAC_PI_3() -> Self {
        <Self as Real>::FRAC_PI_3
    }

    fn FRAC_PI_4() -> Self {
        <Self as Real>::FRAC_PI_4
    }

    fn FRAC_PI_6() -> Self {
        <Self as Real>::FRAC_PI_6
    }

    fn FRAC_PI_8() -> Self {
        <Self as Real>::FRAC_PI_8
    }

    fn LN_10() -> Self {
        <Self as Real>::LN_10
    }

    fn LN_2() -> Self {
        <Self as Real>::LN_2
    }

    fn LOG10_E() -> Self {
        <Self as Real>::LOG10_E
    }

    fn LOG2_E() -> Self {
        <Self as Real>::LOG2_E
    }
}

// TODO: Should constraint violations panic here?
impl<T, P> FromPrimitive for ConstrainedFloat<T, P>
where
    T: Float + FromPrimitive + Primitive,
    P: Constraint<T>,
{
    fn from_i8(value: i8) -> Option<Self> {
        T::from_i8(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_u8(value: u8) -> Option<Self> {
        T::from_u8(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_i16(value: i16) -> Option<Self> {
        T::from_i16(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_u16(value: u16) -> Option<Self> {
        T::from_u16(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_i32(value: i32) -> Option<Self> {
        T::from_i32(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_u32(value: u32) -> Option<Self> {
        T::from_u32(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_i64(value: i64) -> Option<Self> {
        T::from_i64(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_u64(value: u64) -> Option<Self> {
        T::from_u64(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_isize(value: isize) -> Option<Self> {
        T::from_isize(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_usize(value: usize) -> Option<Self> {
        T::from_usize(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_f32(value: f32) -> Option<Self> {
        T::from_f32(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }

    fn from_f64(value: f64) -> Option<Self> {
        T::from_f64(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }
}

impl<T, P> FromStr for ConstrainedFloat<T, P>
where
    T: Float + FromStr + Primitive,
    P: Constraint<T>,
{
    type Err = <T as FromStr>::Err;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        T::from_str(string).map(Self::from_inner)
    }
}

impl<T, P> Hash for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        FloatHash::float_hash(self.as_ref(), state);
    }
}

impl<T, P> Infinite for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T> + Member<InfiniteClass>,
{
    const INFINITY: Self = ConstrainedFloat::from_inner_unchecked(T::INFINITY);
    const NEG_INFINITY: Self = ConstrainedFloat::from_inner_unchecked(T::NEG_INFINITY);

    fn is_infinite(self) -> bool {
        self.into_inner().is_infinite()
    }

    fn is_finite(self) -> bool {
        self.into_inner().is_finite()
    }
}

impl<T, P> LowerExp for ConstrainedFloat<T, P>
where
    T: Float + LowerExp + Primitive,
    P: Constraint<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, P> Mul for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self.zip_map(other, Mul::mul)
    }
}

impl<T, P> Mul<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        self.map(|a| a * other)
    }
}

impl<T, P> MulAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<T, P> MulAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T, P> Nan for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T> + Member<NanClass>,
{
    const NAN: Self = ConstrainedFloat::from_inner_unchecked(T::NAN);

    fn is_nan(self) -> bool {
        self.into_inner().is_nan()
    }
}

impl<T, P> Neg for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        ConstrainedFloat::from_inner_unchecked(-self.into_inner())
    }
}

impl<T, P> Num for ConstrainedFloat<T, P>
where
    Self: PartialEq,
    T: Float + Primitive,
    P: Constraint<T>,
{
    type FromStrRadixErr = ();

    fn from_str_radix(source: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(source, radix)
            .map_err(|_| ())
            .and_then(ConstrainedFloat::try_from_inner)
    }
}

impl<T, P> NumCast for ConstrainedFloat<T, P>
where
    T: Float + NumCast + Primitive + ToPrimitive,
    P: Constraint<T>,
{
    fn from<U>(value: U) -> Option<Self>
    where
        U: ToPrimitive,
    {
        T::from(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }
}

impl<T, P> One for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn one() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::one())
    }
}

impl<T, P> Ord for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        FloatOrd::float_cmp(self.as_ref(), other.as_ref())
    }
}

impl<T, P> PartialEq for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn eq(&self, other: &Self) -> bool {
        FloatEq::float_eq(self.as_ref(), other.as_ref())
    }
}

impl<T, P> PartialEq<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn eq(&self, other: &T) -> bool {
        if let Ok(other) = Self::try_from_inner(*other) {
            Self::eq(self, &other)
        }
        else {
            false
        }
    }
}

impl<T, P> PartialOrd for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(FloatOrd::float_cmp(self.as_ref(), other.as_ref()))
    }
}

impl<T, P> PartialOrd<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Self::try_from_inner(*other)
            .ok()
            .and_then(|other| Self::partial_cmp(self, &other))
    }
}

impl<T, P> Product for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn product<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(One::one(), |a, b| a * b)
    }
}

impl<T, P> Real for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    const E: Self = ConstrainedFloat::from_inner_unchecked(Real::E);
    const PI: Self = ConstrainedFloat::from_inner_unchecked(Real::PI);
    const FRAC_1_PI: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_1_PI);
    const FRAC_2_PI: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_2_PI);
    const FRAC_2_SQRT_PI: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_2_SQRT_PI);
    const FRAC_PI_2: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_PI_2);
    const FRAC_PI_3: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_PI_3);
    const FRAC_PI_4: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_PI_4);
    const FRAC_PI_6: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_PI_6);
    const FRAC_PI_8: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_PI_8);
    const SQRT_2: Self = ConstrainedFloat::from_inner_unchecked(Real::SQRT_2);
    const FRAC_1_SQRT_2: Self = ConstrainedFloat::from_inner_unchecked(Real::FRAC_1_SQRT_2);
    const LN_2: Self = ConstrainedFloat::from_inner_unchecked(Real::LN_2);
    const LN_10: Self = ConstrainedFloat::from_inner_unchecked(Real::LN_10);
    const LOG2_E: Self = ConstrainedFloat::from_inner_unchecked(Real::LOG2_E);
    const LOG10_E: Self = ConstrainedFloat::from_inner_unchecked(Real::LOG10_E);

    fn floor(self) -> Self {
        self.map(Real::floor)
    }

    fn ceil(self) -> Self {
        self.map(Real::ceil)
    }

    fn round(self) -> Self {
        self.map(Real::round)
    }

    fn trunc(self) -> Self {
        self.map(Real::trunc)
    }

    fn fract(self) -> Self {
        self.map(Real::fract)
    }

    fn recip(self) -> Self {
        self.map(Real::recip)
    }

    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self {
        ConstrainedFloat::from_inner(<T as Real>::mul_add(
            self.into_inner(),
            a.into_inner(),
            b.into_inner(),
        ))
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self {
        self.map(|inner| Real::powi(inner, n))
    }

    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self {
        self.zip_map(n, Real::powf)
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self {
        self.map(Real::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map(Real::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self {
        self.map(Real::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self {
        self.map(Real::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self {
        self.map(Real::exp_m1)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self {
        self.zip_map(base, Real::log)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self {
        self.map(Real::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self {
        self.map(Real::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self {
        self.map(Real::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self {
        self.map(Real::ln_1p)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self {
        self.zip_map(other, Real::hypot)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map(Real::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map(Real::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self {
        self.map(Real::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self {
        self.map(Real::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self {
        self.map(Real::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map(Real::atan)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self {
        self.zip_map(other, Real::atan2)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.into_inner().sin_cos();
        (
            ConstrainedFloat::from_inner_unchecked(sin),
            ConstrainedFloat::from_inner_unchecked(cos),
        )
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        self.map(Real::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map(Real::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map(Real::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self {
        self.map(Real::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self {
        self.map(Real::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self {
        self.map(Real::atanh)
    }
}

#[cfg(feature = "approx")]
impl<T, P> RelativeEq for ConstrainedFloat<T, P>
where
    T: Float + Primitive + RelativeEq<Epsilon = T>,
    P: Constraint<T>,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative().into()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.into_inner().relative_eq(
            &other.into_inner(),
            epsilon.into_inner(),
            max_relative.into_inner(),
        )
    }
}

impl<T, P> Rem for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        self.zip_map(other, Rem::rem)
    }
}

impl<T, P> Rem<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn rem(self, other: T) -> Self::Output {
        self.map(|inner| inner % other)
    }
}

impl<T, P> RemAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    }
}

impl<T, P> RemAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn rem_assign(&mut self, other: T) {
        *self = self.map(|inner| inner % other);
    }
}

impl<T, P> Signed for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn abs(&self) -> Self {
        self.map_unchecked(|inner| inner.abs())
    }

    #[cfg(feature = "std")]
    fn abs_sub(&self, other: &Self) -> Self {
        self.zip_map(*other, |a, b| a.abs_sub(&b))
    }

    #[cfg(not(feature = "std"))]
    fn abs_sub(&self, other: &Self) -> Self {
        self.zip_map(*other, |a, b| {
            if a <= b {
                Zero::zero()
            }
            else {
                a - b
            }
        })
    }

    fn signum(&self) -> Self {
        self.map(|inner| inner.signum())
    }

    fn is_positive(&self) -> bool {
        self.into_inner().is_positive()
    }

    fn is_negative(&self) -> bool {
        self.into_inner().is_negative()
    }
}

impl<T, P> Sub for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.zip_map(other, Sub::sub)
    }
}

impl<T, P> Sub<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        self.map(|inner| inner - other)
    }
}

impl<T, P> SubAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other
    }
}

impl<T, P> SubAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn sub_assign(&mut self, other: T) {
        *self = self.map(|inner| inner - other)
    }
}

impl<T, P> Sum for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn sum<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(Zero::zero(), |a, b| a + b)
    }
}

impl<T, P> ToCanonicalBits for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    type Bits = <T as ToCanonicalBits>::Bits;

    fn to_canonical_bits(self) -> Self::Bits {
        self.value.to_canonical_bits()
    }
}

impl<T, P> ToPrimitive for ConstrainedFloat<T, P>
where
    T: Float + Primitive + ToPrimitive,
    P: Constraint<T>,
{
    fn to_i8(&self) -> Option<i8> {
        self.into_inner().to_i8()
    }

    fn to_u8(&self) -> Option<u8> {
        self.into_inner().to_u8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.into_inner().to_i16()
    }

    fn to_u16(&self) -> Option<u16> {
        self.into_inner().to_u16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.into_inner().to_i32()
    }

    fn to_u32(&self) -> Option<u32> {
        self.into_inner().to_u32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.into_inner().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.into_inner().to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self.into_inner().to_isize()
    }

    fn to_usize(&self) -> Option<usize> {
        self.into_inner().to_usize()
    }

    fn to_f32(&self) -> Option<f32> {
        self.into_inner().to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.into_inner().to_f64()
    }
}

#[cfg(feature = "approx")]
impl<T, P> UlpsEq for ConstrainedFloat<T, P>
where
    T: Float + Primitive + UlpsEq<Epsilon = T>,
    P: Constraint<T>,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.into_inner()
            .ulps_eq(&other.into_inner(), epsilon.into_inner(), max_ulps)
    }
}

impl<T, P> UpperExp for ConstrainedFloat<T, P>
where
    T: Float + Primitive + UpperExp,
    P: Constraint<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, P> Zero for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: Constraint<T>,
{
    fn zero() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.into_inner().is_zero()
    }
}

/// Implements the `Real` trait from
/// [`num-traits`](https://crates.io/crates/num-traits) in terms of Decorum's
/// numeric traits. Does nothing if the `std` feature is disabled.
///
/// This is not generic, because the blanket implementation provided by
/// `num-traits` prevents a constraint-based implementation. Instead, this macro
/// must be applied manually to each proxy type exported by Decorum that is
/// `Real` but not `Float`.
///
/// See the following issues:
///
///   https://github.com/olson-sean-k/decorum/issues/10
///   https://github.com/rust-num/num-traits/issues/49
macro_rules! impl_foreign_real {
    (proxy => $t:ty) => {
        #[cfg(feature = "std")]
        impl ForeignReal for $t {
            fn max_value() -> Self {
                Encoding::MAX
            }

            fn min_value() -> Self {
                Encoding::MIN
            }

            fn min_positive_value() -> Self {
                Encoding::MIN_POSITIVE
            }

            fn epsilon() -> Self {
                Encoding::EPSILON
            }

            fn min(self, other: Self) -> Self {
                // Avoid panics by propagating `NaN`s for incomparable values.
                self.zip_map(other, cmp::min_or_undefined)
            }

            fn max(self, other: Self) -> Self {
                // Avoid panics by propagating `NaN`s for incomparable values.
                self.zip_map(other, cmp::max_or_undefined)
            }

            fn is_sign_positive(self) -> bool {
                Encoding::is_sign_positive(self)
            }

            fn is_sign_negative(self) -> bool {
                Encoding::is_sign_negative(self)
            }

            fn signum(self) -> Self {
                Signed::signum(&self)
            }

            fn abs(self) -> Self {
                Signed::abs(&self)
            }

            fn floor(self) -> Self {
                Real::floor(self)
            }

            fn ceil(self) -> Self {
                Real::ceil(self)
            }

            fn round(self) -> Self {
                Real::round(self)
            }

            fn trunc(self) -> Self {
                Real::trunc(self)
            }

            fn fract(self) -> Self {
                Real::fract(self)
            }

            fn recip(self) -> Self {
                Real::recip(self)
            }

            fn mul_add(self, a: Self, b: Self) -> Self {
                Real::mul_add(self, a, b)
            }

            fn abs_sub(self, other: Self) -> Self {
                self.zip_map(other, ForeignFloat::abs_sub)
            }

            fn powi(self, n: i32) -> Self {
                Real::powi(self, n)
            }

            fn powf(self, n: Self) -> Self {
                Real::powf(self, n)
            }

            fn sqrt(self) -> Self {
                Real::sqrt(self)
            }

            fn cbrt(self) -> Self {
                Real::cbrt(self)
            }

            fn exp(self) -> Self {
                Real::exp(self)
            }

            fn exp2(self) -> Self {
                Real::exp2(self)
            }

            fn exp_m1(self) -> Self {
                Real::exp_m1(self)
            }

            fn log(self, base: Self) -> Self {
                Real::log(self, base)
            }

            fn ln(self) -> Self {
                Real::ln(self)
            }

            fn log2(self) -> Self {
                Real::log2(self)
            }

            fn log10(self) -> Self {
                Real::log10(self)
            }

            fn to_degrees(self) -> Self {
                self.map(ForeignFloat::to_degrees)
            }

            fn to_radians(self) -> Self {
                self.map(ForeignFloat::to_radians)
            }

            fn ln_1p(self) -> Self {
                Real::ln_1p(self)
            }

            fn hypot(self, other: Self) -> Self {
                Real::hypot(self, other)
            }

            fn sin(self) -> Self {
                Real::sin(self)
            }

            fn cos(self) -> Self {
                Real::cos(self)
            }

            fn tan(self) -> Self {
                Real::tan(self)
            }

            fn asin(self) -> Self {
                Real::asin(self)
            }

            fn acos(self) -> Self {
                Real::acos(self)
            }

            fn atan(self) -> Self {
                Real::atan(self)
            }

            fn atan2(self, other: Self) -> Self {
                Real::atan2(self, other)
            }

            fn sin_cos(self) -> (Self, Self) {
                Real::sin_cos(self)
            }

            fn sinh(self) -> Self {
                Real::sinh(self)
            }

            fn cosh(self) -> Self {
                Real::cosh(self)
            }

            fn tanh(self) -> Self {
                Real::tanh(self)
            }

            fn asinh(self) -> Self {
                Real::asinh(self)
            }

            fn acosh(self) -> Self {
                Real::acosh(self)
            }

            fn atanh(self) -> Self {
                Real::atanh(self)
            }
        }
    };
}
impl_foreign_real!(proxy => N32);
impl_foreign_real!(proxy => N64);
impl_foreign_real!(proxy => R32);
impl_foreign_real!(proxy => R64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Finite, NotNan, Total, N32, R32};

    #[test]
    fn total_no_panic_on_inf() {
        let x: Total<f32> = 1.0.into();
        let y = x / 0.0;
        assert!(Infinite::is_infinite(y));
    }

    #[test]
    fn total_no_panic_on_nan() {
        let x: Total<f32> = 0.0.into();
        let y = x / 0.0;
        assert!(Nan::is_nan(y));
    }

    #[test]
    fn notnan_no_panic_on_inf() {
        let x: N32 = 1.0.into();
        let y = x / 0.0;
        assert!(Infinite::is_infinite(y));
    }

    #[test]
    #[should_panic]
    fn notnan_panic_on_nan() {
        let x: N32 = 0.0.into();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_nan() {
        let x: R32 = 0.0.into();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_inf() {
        let x: R32 = 1.0.into();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_neg_inf() {
        let x: R32 = (-1.0).into();
        let _ = x / 0.0;
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    fn total_nan_eq() {
        let x: Total<f32> = (0.0 / 0.0).into();
        let y: Total<f32> = (0.0 / 0.0).into();
        assert_eq!(x, y);

        let z: Total<f32> = (<f32 as Infinite>::INFINITY + <f32 as Infinite>::NEG_INFINITY).into();
        assert_eq!(x, z);

        #[cfg(feature = "std")]
        {
            let w: Total<f32> = (Real::sqrt(-1.0)).into();
            assert_eq!(x, w);
        }
    }

    #[test]
    #[allow(clippy::cmp_nan)]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    fn cmp_proxy_primitive() {
        // Compare a canonicalized `NaN` with a primitive `NaN` with a
        // different representation.
        let x: Total<f32> = (0.0 / 0.0).into();
        assert_eq!(x, f32::sqrt(-1.0));

        // Compare a canonicalized `INF` with a primitive `NaN`.
        let y: Total<f32> = (1.0 / 0.0).into();
        assert!(y < (0.0 / 0.0));

        // Compare a proxy that disallows `INF` to a primitive `INF`.
        let z: R32 = 0.0.into();
        assert_eq!(z.partial_cmp(&(1.0 / 0.0)), None);
    }

    #[test]
    fn sum() {
        let xs = [1.0.into(), 2.0.into(), 3.0.into()];
        assert_eq!(xs.iter().cloned().sum::<R32>(), R32::from_inner(6.0));
    }

    #[test]
    fn product() {
        let xs = [1.0.into(), 2.0.into(), 3.0.into()];
        assert_eq!(xs.iter().cloned().product::<R32>(), R32::from_inner(6.0));
    }

    // TODO: This test is questionable.
    #[test]
    fn impl_traits() {
        fn as_float<T>(_: T)
        where
            T: Float,
        {
        }

        fn as_infinite<T>(_: T)
        where
            T: Infinite,
        {
        }

        fn as_nan<T>(_: T)
        where
            T: Nan,
        {
        }

        fn as_real<T>(_: T)
        where
            T: Real,
        {
        }

        let finite = Finite::<f32>::default();
        as_real(finite);

        let notnan = NotNan::<f32>::default();
        as_infinite(notnan);
        as_real(notnan);

        let ordered = Total::<f32>::default();
        as_float(ordered);
        as_infinite(ordered);
        as_nan(ordered);
        as_real(ordered);
    }

    #[test]
    fn fmt() {
        let x: Total<f32> = 1.0.into();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", x);
        let y: NotNan<f32> = 1.0.into();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", y);
        let z: Finite<f32> = 1.0.into();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", z);
    }
}
