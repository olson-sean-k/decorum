//! Constrained, ordered, hashable wrapper for floating point values.

use num_traits::{Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, Signed,
                 ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use {Infinite, Nan, Primitive, Real};
use hash;

pub trait FloatEq<T>
where
    T: Float + Primitive,
{
    // TODO: Consider comparing the output of `canonicalize` here.
    fn eq(lhs: T, rhs: T) -> bool {
        if lhs.is_nan() {
            if rhs.is_nan() {
                true
            }
            else {
                false
            }
        }
        else {
            if rhs.is_nan() {
                false
            }
            else {
                lhs == rhs
            }
        }
    }
}

pub trait FloatPartialOrd<T>
where
    T: Float + Primitive,
{
    fn partial_cmp(lhs: T, rhs: T) -> Option<Ordering> {
        lhs.partial_cmp(&rhs)
    }
}

impl<T, U> FloatPartialOrd<T> for U
where
    T: Float + Primitive,
    U: FloatOrd<T>,
{
    fn partial_cmp(lhs: T, rhs: T) -> Option<Ordering> {
        Some(U::cmp(lhs, rhs))
    }
}

pub trait FloatOrd<T>
where
    T: Float + Primitive,
{
    fn cmp(lhs: T, rhs: T) -> Ordering {
        match lhs.partial_cmp(&rhs) {
            Some(ordering) => ordering,
            None => if lhs.is_nan() {
                if rhs.is_nan() {
                    Ordering::Equal
                }
                else {
                    Ordering::Greater
                }
            }
            else {
                Ordering::Less
            },
        }
    }
}

pub trait FloatInfinity<T>
where
    T: Float + Primitive,
{
    fn infinity() -> T {
        T::infinity()
    }

    fn neg_infinity() -> T {
        T::neg_infinity()
    }

    fn is_infinite(value: T) -> bool {
        value.is_infinite()
    }

    fn is_finite(value: T) -> bool {
        value.is_finite()
    }
}

pub trait FloatNan<T>
where
    T: Float + Primitive,
{
    fn nan() -> T {
        T::nan()
    }

    fn is_nan(value: T) -> bool {
        value.is_nan()
    }
}

/// Constraint on floating point values.
pub trait FloatConstraint<T>: Copy + PartialEq + PartialOrd + Sized
where
    T: Float + Primitive,
{
    /// Filters a floating point value based on some constraints.
    ///
    /// Returns `Some` for values that satisfy constraints and `None` for
    /// values that do not.
    fn evaluate(value: T) -> Option<T>;
}

impl<T> FloatConstraint<T> for ()
where
    T: Float + Primitive,
{
    fn evaluate(value: T) -> Option<T> {
        Some(value)
    }
}

impl<T> FloatEq<T> for ()
where
    T: Float + Primitive,
{
}

impl<T> FloatOrd<T> for ()
where
    T: Float + Primitive,
{
}

impl<T> FloatInfinity<T> for ()
where
    T: Float + Primitive,
{
}

impl<T> FloatNan<T> for ()
where
    T: Float + Primitive,
{
}

/// Disallows `NaN` floating point values.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct NotNanConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> FloatConstraint<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
    fn evaluate(value: T) -> Option<T> {
        if value.is_nan() {
            None
        }
        else {
            Some(value)
        }
    }
}

impl<T> FloatEq<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
    fn eq(lhs: T, rhs: T) -> bool {
        // The input values should never be `NaN`, so just compare the raw
        // floating point values.
        lhs == rhs
    }
}

impl<T> FloatOrd<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
    fn cmp(lhs: T, rhs: T) -> Ordering {
        // The input values should never be `NaN`, so just compare the raw
        // floating point values.
        lhs.partial_cmp(&rhs).unwrap()
    }
}

impl<T> FloatInfinity<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
}

/// Disallows `NaN`, `INF`, and `-INF` floating point values.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> FloatConstraint<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    fn evaluate(value: T) -> Option<T> {
        if value.is_nan() | value.is_infinite() {
            None
        }
        else {
            Some(value)
        }
    }
}

impl<T> FloatEq<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    fn eq(lhs: T, rhs: T) -> bool {
        // The input values should never be `NaN`, so just compare the raw
        // floating point values.
        lhs == rhs
    }
}

impl<T> FloatOrd<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    fn cmp(lhs: T, rhs: T) -> Ordering {
        // The input values should never be `NaN`, so just compare the raw
        // floating point values.
        lhs.partial_cmp(&rhs).unwrap()
    }
}

/// Constrained, ordered, hashable floating point proxy.
///
/// Wraps floating point values and provides a proxy that implements operation
/// and numerical traits, including `Hash`, `Ord`, and `Eq`.
#[derivative(Clone, Copy, Debug, Default)]
#[derive(Derivative)]
pub struct ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    value: T,
    #[derivative(Debug = "ignore")] phantom: PhantomData<P>,
}

impl<T, P> ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    pub fn try_from_raw_float(value: T) -> Result<Self, ()> {
        P::evaluate(value)
            .map(|value| {
                ConstrainedFloat {
                    value,
                    phantom: PhantomData,
                }
            })
            .ok_or(())
    }

    pub fn into_raw_float(self) -> T {
        let ConstrainedFloat { value, .. } = self;
        value
    }

    fn from_raw_float_unchecked(value: T) -> Self {
        ConstrainedFloat {
            value,
            phantom: PhantomData,
        }
    }
}

#[cfg(feature = "enforce-constraints")]
impl<T, P> ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    pub fn from_raw_float(value: T) -> Self {
        P::evaluate(value)
            .map(|value| {
                ConstrainedFloat {
                    value,
                    phantom: PhantomData,
                }
            })
            .unwrap()
    }
}

#[cfg(not(feature = "enforce-constraints"))]
impl<T, P> ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    pub fn from_raw_float(value: T) -> Self {
        Self::from_raw_float_unchecked(value)
    }
}

impl<T, P> AsRef<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn as_ref(&self) -> &T {
        &self.value
    }
}

// Because of the reflexive implementation in core, this `Into` cannot be
// implemented over a type `T`.
impl<P> Into<f32> for ConstrainedFloat<f32, P>
where
    P: FloatConstraint<f32>,
{
    fn into(self) -> f32 {
        self.into_raw_float()
    }
}

// Because of the reflexive implementation in core, this `Into` cannot be
// implemented over a type `T`.
impl<P> Into<f64> for ConstrainedFloat<f64, P>
where
    P: FloatConstraint<f64>,
{
    fn into(self) -> f64 {
        self.into_raw_float()
    }
}

impl<T, P> Add for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() + other.into_raw_float())
    }
}

impl<T, P> Add<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() + other)
    }
}

impl<T, P> AddAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn add_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() + other.into_raw_float())
    }
}

impl<T, P> AddAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn add_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() + other)
    }
}

impl<T, P> Bounded for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    fn min_value() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::min_value())
    }

    #[inline(always)]
    fn max_value() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::max_value())
    }
}

impl<T, P> Display for ConstrainedFloat<T, P>
where
    T: Display + Float + Primitive,
    P: FloatConstraint<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, P> Div for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() / other.into_raw_float())
    }
}

impl<T, P> Div<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() / other)
    }
}

impl<T, P> DivAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn div_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() / other.into_raw_float())
    }
}

impl<T, P> DivAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn div_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() / other)
    }
}

impl<T, P> Eq for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + FloatEq<T>,
{
}

impl<T, P> Float for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + FloatEq<T> + FloatInfinity<T> + FloatNan<T> + FloatPartialOrd<T>,
{
    #[inline(always)]
    fn infinity() -> Self {
        Infinite::infinity()
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        Infinite::neg_infinity()
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        Infinite::is_infinite(self)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        Infinite::is_finite(self)
    }

    #[inline(always)]
    fn nan() -> Self {
        Nan::nan()
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        Nan::is_nan(self)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Real::max_value()
    }

    #[inline(always)]
    fn min_value() -> Self {
        Real::min_value()
    }

    #[inline(always)]
    fn min_positive_value() -> Self {
        Real::min_positive_value()
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Real::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Real::max(self, other)
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        Real::neg_zero()
    }

    #[inline(always)]
    fn is_sign_positive(self) -> bool {
        Real::is_sign_positive(self)
    }

    #[inline(always)]
    fn is_sign_negative(self) -> bool {
        Real::is_sign_negative(self)
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Real::signum(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Real::abs(self)
    }

    #[inline(always)]
    fn classify(self) -> FpCategory {
        Real::classify(self)
    }

    #[inline(always)]
    fn is_normal(self) -> bool {
        Real::is_normal(self)
    }

    #[inline(always)]
    fn integer_decode(self) -> (u64, i16, i8) {
        Real::integer_decode(self)
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Real::floor(self)
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Real::ceil(self)
    }

    #[inline(always)]
    fn round(self) -> Self {
        Real::round(self)
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        Real::trunc(self)
    }

    #[inline(always)]
    fn fract(self) -> Self {
        Real::fract(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Real::recip(self)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Real::mul_add(self, a, b)
    }

    #[inline(always)]
    fn abs_sub(self, other: Self) -> Self {
        Real::abs_sub(self, other)
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        Real::powi(self, n)
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        Real::powf(self, n)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Real::sqrt(self)
    }

    #[inline(always)]
    fn cbrt(self) -> Self {
        Real::cbrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        Real::exp(self)
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        Real::exp2(self)
    }

    #[inline(always)]
    fn exp_m1(self) -> Self {
        Real::exp_m1(self)
    }

    #[inline(always)]
    fn log(self, base: Self) -> Self {
        Real::log(self, base)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        Real::ln(self)
    }

    #[inline(always)]
    fn log2(self) -> Self {
        Real::log2(self)
    }

    #[inline(always)]
    fn log10(self) -> Self {
        Real::log10(self)
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        Real::ln_1p(self)
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        Real::hypot(self, other)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Real::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Real::cos(self)
    }

    #[inline(always)]
    fn tan(self) -> Self {
        Real::tan(self)
    }

    #[inline(always)]
    fn asin(self) -> Self {
        Real::asin(self)
    }

    #[inline(always)]
    fn acos(self) -> Self {
        Real::acos(self)
    }

    #[inline(always)]
    fn atan(self) -> Self {
        Real::atan(self)
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        Real::atan2(self, other)
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        Real::sin_cos(self)
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        Real::sinh(self)
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        Real::cosh(self)
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        Real::tanh(self)
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        Real::asinh(self)
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        Real::acosh(self)
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        Real::atanh(self)
    }
}

impl<T, P> FloatConst for ConstrainedFloat<T, P>
where
    T: Float + FloatConst + Primitive,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    fn E() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::E())
    }

    #[inline(always)]
    fn PI() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::PI())
    }

    #[inline(always)]
    fn SQRT_2() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::SQRT_2())
    }

    #[inline(always)]
    fn FRAC_1_PI() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_1_PI())
    }

    #[inline(always)]
    fn FRAC_2_PI() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_2_PI())
    }

    #[inline(always)]
    fn FRAC_1_SQRT_2() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_1_SQRT_2())
    }

    #[inline(always)]
    fn FRAC_2_SQRT_PI() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_2_SQRT_PI())
    }

    #[inline(always)]
    fn FRAC_PI_2() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_PI_2())
    }

    #[inline(always)]
    fn FRAC_PI_3() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_PI_3())
    }

    #[inline(always)]
    fn FRAC_PI_4() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_PI_4())
    }

    #[inline(always)]
    fn FRAC_PI_6() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_PI_6())
    }

    #[inline(always)]
    fn FRAC_PI_8() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::FRAC_PI_8())
    }

    #[inline(always)]
    fn LN_10() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::LN_10())
    }

    #[inline(always)]
    fn LN_2() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::LN_2())
    }

    #[inline(always)]
    fn LOG10_E() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::LOG10_E())
    }

    #[inline(always)]
    fn LOG2_E() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::LOG2_E())
    }
}

impl<T, P> FromPrimitive for ConstrainedFloat<T, P>
where
    T: Float + FromPrimitive + Primitive,
    P: FloatConstraint<T>,
{
    fn from_i8(value: i8) -> Option<Self> {
        T::from_i8(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_u8(value: u8) -> Option<Self> {
        T::from_u8(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_i16(value: i16) -> Option<Self> {
        T::from_i16(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_u16(value: u16) -> Option<Self> {
        T::from_u16(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_i32(value: i32) -> Option<Self> {
        T::from_i32(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_u32(value: u32) -> Option<Self> {
        T::from_u32(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_i64(value: i64) -> Option<Self> {
        T::from_i64(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_u64(value: u64) -> Option<Self> {
        T::from_u64(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_isize(value: isize) -> Option<Self> {
        T::from_isize(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_usize(value: usize) -> Option<Self> {
        T::from_usize(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_f32(value: f32) -> Option<Self> {
        T::from_f32(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }

    fn from_f64(value: f64) -> Option<Self> {
        T::from_f64(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }
}

impl<T, P> Hash for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        hash::hash_float(self.into_raw_float(), state);
    }
}

impl<T, P> Infinite for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + FloatInfinity<T>,
{
    #[inline(always)]
    fn infinity() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(P::infinity())
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(P::neg_infinity())
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        P::is_infinite(self.into_raw_float())
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        P::is_finite(self.into_raw_float())
    }
}

impl<T, P> Mul for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() * other.into_raw_float())
    }
}

impl<T, P> Mul<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() * other)
    }
}

impl<T, P> MulAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() * other.into_raw_float())
    }
}

impl<T, P> MulAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn mul_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() * other)
    }
}

impl<T, P> Nan for ConstrainedFloat<T, P>
where
    T: Float + Num + Primitive,
    P: FloatConstraint<T> + FloatNan<T>,
{
    #[inline(always)]
    fn nan() -> Self {
        Self::from_raw_float_unchecked(P::nan())
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        P::is_nan(self.into_raw_float())
    }
}

impl<T, P> Neg for ConstrainedFloat<T, P>
where
    T: Float + Num + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        ConstrainedFloat::from_raw_float_unchecked(-self.into_raw_float())
    }
}

impl<T, P> Num for ConstrainedFloat<T, P>
where
    Self: PartialEq,
    T: Float + Num + Primitive,
    P: FloatConstraint<T>,
{
    type FromStrRadixErr = ();

    fn from_str_radix(source: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(source, radix)
            .map_err(|_| ())
            .and_then(|value| {
                ConstrainedFloat::try_from_raw_float(value).map_err(|_| ())
            })
    }
}

impl<T, P> NumCast for ConstrainedFloat<T, P>
where
    T: Float + Num + Primitive,
    P: FloatConstraint<T>,
{
    fn from<U>(value: U) -> Option<Self>
    where
        U: ToPrimitive,
    {
        T::from(value).and_then(|value| ConstrainedFloat::try_from_raw_float(value).ok())
    }
}

impl<T, P> One for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    fn one() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::one())
    }
}

impl<T, P> Ord for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + FloatEq<T> + FloatOrd<T>,
{
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        <P as FloatOrd<T>>::cmp(self.into_raw_float(), other.into_raw_float())
    }
}

impl<T, P> PartialEq for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + FloatEq<T>,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        <P as FloatEq<T>>::eq(self.into_raw_float(), other.into_raw_float())
    }
}

impl<T, P> PartialOrd for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + FloatEq<T> + FloatPartialOrd<T>,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        <P as FloatPartialOrd<T>>::partial_cmp(self.into_raw_float(), other.into_raw_float())
    }
}

impl<T, P> Real for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    fn max_value() -> Self {
        <Self as Bounded>::max_value()
    }

    #[inline(always)]
    fn min_value() -> Self {
        <Self as Bounded>::min_value()
    }

    #[inline(always)]
    fn min_positive_value() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::min_positive_value())
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(
            T::min(self.into_raw_float(), other.into_raw_float()),
        )
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(
            T::max(self.into_raw_float(), other.into_raw_float()),
        )
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::neg_zero())
    }

    #[inline(always)]
    fn is_sign_positive(self) -> bool {
        T::is_sign_positive(self.into_raw_float())
    }

    #[inline(always)]
    fn is_sign_negative(self) -> bool {
        T::is_sign_negative(self.into_raw_float())
    }

    #[inline(always)]
    fn signum(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::signum(self.into_raw_float()))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::abs(self.into_raw_float()))
    }

    #[inline(always)]
    fn classify(self) -> FpCategory {
        T::classify(self.into_raw_float())
    }

    #[inline(always)]
    fn is_normal(self) -> bool {
        T::is_normal(self.into_raw_float())
    }

    #[inline(always)]
    fn integer_decode(self) -> (u64, i16, i8) {
        T::integer_decode(self.into_raw_float())
    }

    #[inline(always)]
    fn floor(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::floor(self.into_raw_float()))
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::ceil(self.into_raw_float()))
    }

    #[inline(always)]
    fn round(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::round(self.into_raw_float()))
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::trunc(self.into_raw_float()))
    }

    #[inline(always)]
    fn fract(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::fract(self.into_raw_float()))
    }

    #[inline(always)]
    fn recip(self) -> Self {
        ConstrainedFloat::from_raw_float(T::recip(self.into_raw_float()))
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::mul_add(
            self.into_raw_float(),
            a.into_raw_float(),
            b.into_raw_float(),
        ))
    }

    #[inline(always)]
    fn abs_sub(self, other: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(
            T::abs_sub(self.into_raw_float(), other.into_raw_float()),
        )
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::powi(self.into_raw_float(), n))
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(
            T::powf(self.into_raw_float(), n.into_raw_float()),
        )
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::sqrt(self.into_raw_float()))
    }

    #[inline(always)]
    fn cbrt(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::cbrt(self.into_raw_float()))
    }

    #[inline(always)]
    fn exp(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::exp(self.into_raw_float()))
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::exp2(self.into_raw_float()))
    }

    #[inline(always)]
    fn exp_m1(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::exp_m1(self.into_raw_float()))
    }

    #[inline(always)]
    fn log(self, base: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(
            T::log(self.into_raw_float(), base.into_raw_float()),
        )
    }

    #[inline(always)]
    fn ln(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::ln(self.into_raw_float()))
    }

    #[inline(always)]
    fn log2(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::log2(self.into_raw_float()))
    }

    #[inline(always)]
    fn log10(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::log10(self.into_raw_float()))
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::ln_1p(self.into_raw_float()))
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(
            self.into_raw_float().hypot(other.into_raw_float()),
        )
    }

    #[inline(always)]
    fn sin(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().sin())
    }

    #[inline(always)]
    fn cos(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().cos())
    }

    #[inline(always)]
    fn tan(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().tan())
    }

    #[inline(always)]
    fn asin(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().asin())
    }

    #[inline(always)]
    fn acos(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().acos())
    }

    #[inline(always)]
    fn atan(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().atan())
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(
            self.into_raw_float().atan2(other.into_raw_float()),
        )
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.into_raw_float().sin_cos();
        (
            ConstrainedFloat::from_raw_float_unchecked(sin),
            ConstrainedFloat::from_raw_float_unchecked(cos),
        )
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().sinh())
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().cosh())
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().tanh())
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().asinh())
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().acosh())
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().atanh())
    }
}

impl<T, P> Rem for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() % other.into_raw_float())
    }
}

impl<T, P> Rem<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn rem(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() % other)
    }
}

impl<T, P> RemAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn rem_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() % other.into_raw_float())
    }
}

impl<T, P> RemAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn rem_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() % other)
    }
}

impl<T, P> Signed for ConstrainedFloat<T, P>
where
    T: Float + Primitive + Signed,
    P: FloatConstraint<T> + FloatEq<T>,
{
    #[inline(always)]
    fn abs(&self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().abs())
    }

    #[inline(always)]
    fn abs_sub(&self, other: &Self) -> Self {
        ConstrainedFloat::from_raw_float(self.into_raw_float().abs_sub(other.into_raw_float()))
    }

    #[inline(always)]
    fn signum(&self) -> Self {
        ConstrainedFloat::from_raw_float_unchecked(self.into_raw_float().signum())
    }

    #[inline(always)]
    fn is_positive(&self) -> bool {
        self.into_raw_float().is_positive()
    }

    #[inline(always)]
    fn is_negative(&self) -> bool {
        self.into_raw_float().is_negative()
    }
}

impl<T, P> Sub for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() - other.into_raw_float())
    }
}

impl<T, P> Sub<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() - other)
    }
}

impl<T, P> SubAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() - other.into_raw_float())
    }
}

impl<T, P> SubAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn sub_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() - other)
    }
}

impl<T, P> ToPrimitive for ConstrainedFloat<T, P>
where
    T: Float + Primitive + ToPrimitive,
    P: FloatConstraint<T>,
{
    fn to_i8(&self) -> Option<i8> {
        self.into_raw_float().to_i8()
    }

    fn to_u8(&self) -> Option<u8> {
        self.into_raw_float().to_u8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.into_raw_float().to_i16()
    }

    fn to_u16(&self) -> Option<u16> {
        self.into_raw_float().to_u16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.into_raw_float().to_i32()
    }

    fn to_u32(&self) -> Option<u32> {
        self.into_raw_float().to_u32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.into_raw_float().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.into_raw_float().to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self.into_raw_float().to_isize()
    }

    fn to_usize(&self) -> Option<usize> {
        self.into_raw_float().to_usize()
    }

    fn to_f32(&self) -> Option<f32> {
        self.into_raw_float().to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.into_raw_float().to_f64()
    }
}

impl<T, P> Zero for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    fn zero() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        T::is_zero(&self.into_raw_float())
    }
}

#[cfg(feature = "serialize-serde")]
mod feature_serialize_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde::de::{Error, Unexpected};
    use std::f64;

    use super::*;

    impl<'a, T, P> Deserialize<'a> for ConstrainedFloat<T, P>
    where
        T: Deserialize<'a> + Float + Primitive,
        P: FloatConstraint<T>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'a>,
        {
            let value = T::deserialize(deserializer)?;
            ConstrainedFloat::try_from_raw_float(value)
                .map_err(|_| Error::invalid_value(Unexpected::Float(f64::NAN), &""))
        }
    }

    impl<T, P> Serialize for ConstrainedFloat<T, P>
    where
        T: Float + Primitive + Serialize,
        P: FloatConstraint<T>,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.into_raw_float().serialize(serializer)
        }
    }
}

#[cfg(test)]
mod tests {
    use {Finite, NotNan, Ordered};
    use super::*;

    // TODO: This test is incomplete: it only ensures that the expected traits
    //       are implemented, but there is no test that ensures that unwanted
    //       traits are NOT implemented.
    #[test]
    fn constrained_float_impl_traits() {
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

        let ordered = Ordered::<f32>::default();
        as_float(ordered);
        as_infinite(ordered);
        as_nan(ordered);
        as_real(ordered);
    }
}
