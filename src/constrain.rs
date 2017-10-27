//! Constrained, ordered, hashable wrapper for floating point values.

use num_traits::{Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, Signed,
                 ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use {Infinite, Real};
use hash;

pub trait FloatEq<T>
where
    T: Float,
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

pub trait FloatOrd<T>
where
    T: Float,
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

/// Constraint on floating point values.
pub trait FloatConstraint<T>: Copy + PartialEq + PartialOrd + Sized
where
    T: Float,
{
    /// Filters a floating point value based on some constraints.
    ///
    /// Returns `Some` for values that satisfy constraints and `None` for
    /// values that do not.
    fn evaluate(value: T) -> Option<T>;
}

/// Disallows `NaN` floating point values.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct NotNanConstraint<T>
where
    T: Float,
{
    phantom: PhantomData<T>,
}

impl<T> FloatConstraint<T> for NotNanConstraint<T>
where
    T: Float,
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

// TODO: Customize equality and ordering to omit `NaN` checks.
impl<T> FloatEq<T> for NotNanConstraint<T>
where
    T: Float,
{
}

impl<T> FloatOrd<T> for NotNanConstraint<T>
where
    T: Float,
{
}

/// Disallows `NaN`, `INF`, and `-INF` floating point values.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteConstraint<T>
where
    T: Float,
{
    phantom: PhantomData<T>,
}

impl<T> FloatConstraint<T> for FiniteConstraint<T>
where
    T: Float,
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

// TODO: Customize equality and ordering to omit `NaN` and infinity checks.
impl<T> FloatEq<T> for FiniteConstraint<T>
where
    T: Float,
{
}

impl<T> FloatOrd<T> for FiniteConstraint<T>
where
    T: Float,
{
}

/// Constrained, ordered, hashable floating point proxy.
///
/// Wraps floating point values and provides a proxy that implements operation
/// and numerical traits, including `Hash`, `Ord`, and `Eq`.
#[derivative(Clone, Copy, Debug, Default)]
#[derive(Derivative)]
pub struct ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    value: T,
    #[derivative(Debug = "ignore")] phantom: PhantomData<P>,
}

impl<T, P> ConstrainedFloat<T, P>
where
    T: Float,
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
    T: Float,
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
    T: Float,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    pub fn from_raw_float(value: T) -> Self {
        Self::from_raw_float_unchecked(value)
    }
}

impl<T, P> AsRef<T> for ConstrainedFloat<T, P>
where
    T: Float,
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
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() + other.into_raw_float())
    }
}

impl<T, P> Add<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() + other)
    }
}

impl<T, P> AddAssign for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn add_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() + other.into_raw_float())
    }
}

impl<T, P> AddAssign<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn add_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() + other)
    }
}

impl<T, P> Bounded for ConstrainedFloat<T, P>
where
    T: Float,
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
    T: Display + Float,
    P: FloatConstraint<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, P> Div for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() / other.into_raw_float())
    }
}

impl<T, P> Div<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() / other)
    }
}

impl<T, P> DivAssign for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn div_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() / other.into_raw_float())
    }
}

impl<T, P> DivAssign<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn div_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() / other)
    }
}

impl<T, P> Eq for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T> + FloatEq<T>,
{
}

impl<T, P> FloatConst for ConstrainedFloat<T, P>
where
    T: Float + FloatConst,
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
    T: Float + FromPrimitive,
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
    T: Float,
    P: FloatConstraint<T>,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        hash::hash_float(self.into_raw_float(), state);
    }
}

impl<T> Infinite for ConstrainedFloat<T, NotNanConstraint<T>>
where
    T: Float,
{
    #[inline(always)]
    fn infinity() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::infinity())
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::neg_infinity())
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        T::is_infinite(self.into_raw_float())
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        T::is_finite(self.into_raw_float())
    }
}

impl<T, P> Mul for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() * other.into_raw_float())
    }
}

impl<T, P> Mul<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() * other)
    }
}

impl<T, P> MulAssign for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() * other.into_raw_float())
    }
}

impl<T, P> MulAssign<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn mul_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() * other)
    }
}

impl<T, P> Neg for ConstrainedFloat<T, P>
where
    T: Float + Num,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        ConstrainedFloat::from_raw_float_unchecked(-self.into_raw_float())
    }
}

impl<T, P> Num for ConstrainedFloat<T, P>
where
    T: Float + Num,
    P: FloatConstraint<T> + FloatEq<T>,
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
    T: Float + Num,
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
    T: Float,
    P: FloatConstraint<T>,
{
    #[inline(always)]
    fn one() -> Self {
        ConstrainedFloat::from_raw_float_unchecked(T::one())
    }
}

impl<T, P> Ord for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T> + FloatEq<T> + FloatOrd<T>,
{
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        <P as FloatOrd<T>>::cmp(self.into_raw_float(), other.into_raw_float())
    }
}

impl<T, P> PartialEq for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T> + FloatEq<T>,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        <P as FloatEq<T>>::eq(self.into_raw_float(), other.into_raw_float())
    }
}

impl<T, P> PartialOrd for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T> + FloatEq<T> + FloatOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, P> Real for ConstrainedFloat<T, P>
where
    T: Float,
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
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() % other.into_raw_float())
    }
}

impl<T, P> Rem<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn rem(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() % other)
    }
}

impl<T, P> RemAssign for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn rem_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() % other.into_raw_float())
    }
}

impl<T, P> RemAssign<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn rem_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() % other)
    }
}

impl<T, P> Signed for ConstrainedFloat<T, P>
where
    T: Float + Signed,
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
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() - other.into_raw_float())
    }
}

impl<T, P> Sub<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        ConstrainedFloat::from_raw_float(self.into_raw_float() - other)
    }
}

impl<T, P> SubAssign for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() - other.into_raw_float())
    }
}

impl<T, P> SubAssign<T> for ConstrainedFloat<T, P>
where
    T: Float,
    P: FloatConstraint<T>,
{
    fn sub_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_raw_float(self.into_raw_float() - other)
    }
}

impl<T, P> ToPrimitive for ConstrainedFloat<T, P>
where
    T: Float + ToPrimitive,
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
    T: Float,
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
        T: Deserialize<'a> + Float,
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
        T: Float + Serialize,
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
