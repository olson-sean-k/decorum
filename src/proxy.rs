use core::cmp::Ordering;
use core::fmt::{self, Display, Formatter, LowerExp, UpperExp};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::num::FpCategory;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::str::FromStr;
use num_traits::{
    Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};

use crate::canonical;
use crate::constraint::{
    ConstraintEq, ConstraintInfinity, ConstraintNan, ConstraintOrd, ConstraintPartialOrd,
    FloatConstraint, SubsetOf, SupersetOf,
};
use crate::{Encoding, Finite, Infinite, Nan, NotNan, Ordered, Primitive, Real};

/// Constrained, ordered, hashable floating-point proxy.
///
/// Wraps floating-point values and provides a proxy that implements operation
/// and numerical traits, including `Hash`, `Ord`, and `Eq`. May apply
/// constraints that prevent certain values from occurring (by panicing).
#[cfg_attr(feature = "serialize-serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    value: T,
    phantom: PhantomData<P>,
}

impl<T, P> ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    // TODO: Avoid the overhead of `filter` and `expect` for the `()`
    //       constraint (i.e., no constraints). When specialization lands, this
    //       may be easy to implement.
    pub fn from_inner(value: T) -> Self {
        Self::try_from_inner(value).expect("floating-point constraint violated")
    }

    pub fn into_inner(self) -> T {
        let ConstrainedFloat { value, .. } = self;
        value
    }

    pub fn from_subset<Q>(other: ConstrainedFloat<T, Q>) -> Self
    where
        Q: FloatConstraint<T> + SubsetOf<P>,
    {
        Self::from_inner_unchecked(other.into_inner())
    }

    pub fn into_superset<Q>(self) -> ConstrainedFloat<T, Q>
    where
        Q: FloatConstraint<T> + SupersetOf<P>,
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

    fn from_inner_unchecked(value: T) -> Self {
        ConstrainedFloat {
            value,
            phantom: PhantomData,
        }
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

// It is not possible to implement `From` for proxies in a generic way, because
// the `FloatConstraint` types `T` and `U` may be the same and conflict with
// the reflexive implementation in core. A similar problem prevents
// implementing `From` over a type `T: Float`.

impl<T> From<NotNan<T>> for Ordered<T>
where
    T: Float + Primitive,
{
    fn from(other: NotNan<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<T> From<Finite<T>> for Ordered<T>
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
    P: FloatConstraint<T>,
{
    fn from(value: T) -> Self {
        Self::from_inner(value)
    }
}

impl<P> From<ConstrainedFloat<f32, P>> for f32
where
    P: FloatConstraint<f32>,
{
    fn from(value: ConstrainedFloat<f32, P>) -> Self {
        value.into_inner()
    }
}

impl<P> From<ConstrainedFloat<f64, P>> for f64
where
    P: FloatConstraint<f64>,
{
    fn from(value: ConstrainedFloat<f64, P>) -> Self {
        value.into_inner()
    }
}

impl<T, P> Add for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() + other.into_inner())
    }
}

impl<T, P> Add<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() + other)
    }
}

impl<T, P> AddAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn add_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_inner(self.into_inner() + other.into_inner())
    }
}

impl<T, P> AddAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn add_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_inner(self.into_inner() + other)
    }
}

impl<T, P> Bounded for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn min_value() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::min_value())
    }

    fn max_value() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::max_value())
    }
}

impl<T, P> Default for ConstrainedFloat<T, P>
where
    T: Default + Float + Primitive,
    P: FloatConstraint<T>,
{
    fn default() -> Self {
        ConstrainedFloat::from_inner_unchecked(Default::default())
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
        ConstrainedFloat::from_inner(self.into_inner() / other.into_inner())
    }
}

impl<T, P> Div<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() / other)
    }
}

impl<T, P> DivAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn div_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_inner(self.into_inner() / other.into_inner())
    }
}

impl<T, P> DivAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn div_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_inner(self.into_inner() / other)
    }
}

impl<T, P> Encoding for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn max_value() -> Self {
        <Self as Bounded>::max_value()
    }

    fn min_value() -> Self {
        <Self as Bounded>::min_value()
    }

    fn min_positive_value() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::min_positive_value())
    }

    fn epsilon() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::epsilon())
    }

    fn classify(self) -> FpCategory {
        T::classify(self.into_inner())
    }

    fn is_normal(self) -> bool {
        T::is_normal(self.into_inner())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        T::integer_decode(self.into_inner())
    }
}

impl<T, P> Eq for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + ConstraintEq<T>,
{
}

impl<T, P> Float for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>
        + ConstraintEq<T>
        + ConstraintInfinity<T>
        + ConstraintNan<T>
        + ConstraintPartialOrd<T>,
{
    fn infinity() -> Self {
        Infinite::infinity()
    }

    fn neg_infinity() -> Self {
        Infinite::neg_infinity()
    }

    fn is_infinite(self) -> bool {
        Infinite::is_infinite(self)
    }

    fn is_finite(self) -> bool {
        Infinite::is_finite(self)
    }

    fn nan() -> Self {
        Nan::nan()
    }

    fn is_nan(self) -> bool {
        Nan::is_nan(self)
    }

    fn max_value() -> Self {
        Encoding::max_value()
    }

    fn min_value() -> Self {
        Encoding::min_value()
    }

    fn min_positive_value() -> Self {
        Encoding::min_positive_value()
    }

    fn epsilon() -> Self {
        Encoding::epsilon()
    }

    fn min(self, other: Self) -> Self {
        Real::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Real::max(self, other)
    }

    fn neg_zero() -> Self {
        Self::from_inner(T::neg_zero())
    }

    fn is_sign_positive(self) -> bool {
        Real::is_sign_positive(self)
    }

    fn is_sign_negative(self) -> bool {
        Real::is_sign_negative(self)
    }

    fn signum(self) -> Self {
        Real::signum(self)
    }

    fn abs(self) -> Self {
        Real::abs(self)
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
        Real::abs_sub(self, other)
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

impl<T, P> FloatConst for ConstrainedFloat<T, P>
where
    T: Float + FloatConst + Primitive,
    P: FloatConstraint<T>,
{
    fn E() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::E())
    }

    fn PI() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::PI())
    }

    fn SQRT_2() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::SQRT_2())
    }

    fn FRAC_1_PI() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_1_PI())
    }

    fn FRAC_2_PI() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_2_PI())
    }

    fn FRAC_1_SQRT_2() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_1_SQRT_2())
    }

    fn FRAC_2_SQRT_PI() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_2_SQRT_PI())
    }

    fn FRAC_PI_2() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_PI_2())
    }

    fn FRAC_PI_3() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_PI_3())
    }

    fn FRAC_PI_4() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_PI_4())
    }

    fn FRAC_PI_6() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_PI_6())
    }

    fn FRAC_PI_8() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::FRAC_PI_8())
    }

    fn LN_10() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::LN_10())
    }

    fn LN_2() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::LN_2())
    }

    fn LOG10_E() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::LOG10_E())
    }

    fn LOG2_E() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::LOG2_E())
    }
}

impl<T, P> FromPrimitive for ConstrainedFloat<T, P>
where
    T: Float + FromPrimitive + Primitive,
    P: FloatConstraint<T>,
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
    P: FloatConstraint<T>,
{
    type Err = <T as FromStr>::Err;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        T::from_str(string).map(|value| Self::from_inner(value))
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
        canonical::hash_float(self.into_inner(), state);
    }
}

impl<T, P> Infinite for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + ConstraintInfinity<T>,
{
    fn infinity() -> Self {
        ConstrainedFloat::from_inner_unchecked(P::infinity())
    }

    fn neg_infinity() -> Self {
        ConstrainedFloat::from_inner_unchecked(P::neg_infinity())
    }

    fn is_infinite(self) -> bool {
        P::is_infinite(self.into_inner())
    }

    fn is_finite(self) -> bool {
        P::is_finite(self.into_inner())
    }
}

impl<T, P> LowerExp for ConstrainedFloat<T, P>
where
    T: Float + LowerExp + Primitive,
    P: FloatConstraint<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, P> Mul for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() * other.into_inner())
    }
}

impl<T, P> Mul<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() * other)
    }
}

impl<T, P> MulAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_inner(self.into_inner() * other.into_inner())
    }
}

impl<T, P> MulAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn mul_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_inner(self.into_inner() * other)
    }
}

impl<T, P> Nan for ConstrainedFloat<T, P>
where
    T: Float + Num + Primitive,
    P: FloatConstraint<T> + ConstraintNan<T>,
{
    fn nan() -> Self {
        Self::from_inner_unchecked(P::nan())
    }

    fn is_nan(self) -> bool {
        P::is_nan(self.into_inner())
    }
}

impl<T, P> Neg for ConstrainedFloat<T, P>
where
    T: Float + Num + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        ConstrainedFloat::from_inner_unchecked(-self.into_inner())
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
            .and_then(|value| ConstrainedFloat::try_from_inner(value).map_err(|_| ()))
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
        T::from(value).and_then(|value| ConstrainedFloat::try_from_inner(value).ok())
    }
}

impl<T, P> One for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn one() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::one())
    }
}

impl<T, P> Ord for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + ConstraintEq<T> + ConstraintOrd<T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        <P as ConstraintOrd<T>>::cmp(self.into_inner(), other.into_inner())
    }
}

impl<T, P> PartialEq for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + ConstraintEq<T>,
{
    fn eq(&self, other: &Self) -> bool {
        <P as ConstraintEq<T>>::eq(self.into_inner(), other.into_inner())
    }
}

impl<T, P> PartialEq<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + ConstraintEq<T>,
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
    P: FloatConstraint<T> + ConstraintEq<T> + ConstraintPartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        <P as ConstraintPartialOrd<T>>::partial_cmp(self.into_inner(), other.into_inner())
    }
}

impl<T, P> PartialOrd<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + ConstraintEq<T> + ConstraintPartialOrd<T>,
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
    P: FloatConstraint<T>,
{
    fn product<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(One::one(), |a, b| a * b)
    }
}

// This implementation uses unchecked conversions for some operations, but
// applies to general proxy types and so must support the most constrained types
// exposed by Decorum. Operations that use unchecked conversions must be chosen
// carefully to avoid exposing `NaN`, `INF`, and other potentially disallowed
// values from going unchecked.
impl<T, P> Real for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T> + ConstraintEq<T> + ConstraintPartialOrd<T>,
{
    fn min(self, other: Self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::min(self.into_inner(), other.into_inner()))
    }

    fn max(self, other: Self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::max(self.into_inner(), other.into_inner()))
    }

    fn is_sign_positive(self) -> bool {
        T::is_sign_positive(self.into_inner())
    }

    fn is_sign_negative(self) -> bool {
        T::is_sign_negative(self.into_inner())
    }

    fn signum(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::signum(self.into_inner()))
    }

    fn abs(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::abs(self.into_inner()))
    }

    fn floor(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::floor(self.into_inner()))
    }

    fn ceil(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::ceil(self.into_inner()))
    }

    fn round(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::round(self.into_inner()))
    }

    fn trunc(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::trunc(self.into_inner()))
    }

    fn fract(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(T::fract(self.into_inner()))
    }

    fn recip(self) -> Self {
        ConstrainedFloat::from_inner(T::recip(self.into_inner()))
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        ConstrainedFloat::from_inner(T::mul_add(
            self.into_inner(),
            a.into_inner(),
            b.into_inner(),
        ))
    }

    fn abs_sub(self, other: Self) -> Self {
        ConstrainedFloat::from_inner(T::abs_sub(self.into_inner(), other.into_inner()))
    }

    fn powi(self, n: i32) -> Self {
        ConstrainedFloat::from_inner(T::powi(self.into_inner(), n))
    }

    fn powf(self, n: Self) -> Self {
        ConstrainedFloat::from_inner(T::powf(self.into_inner(), n.into_inner()))
    }

    fn sqrt(self) -> Self {
        ConstrainedFloat::from_inner(T::sqrt(self.into_inner()))
    }

    fn cbrt(self) -> Self {
        ConstrainedFloat::from_inner(T::cbrt(self.into_inner()))
    }

    fn exp(self) -> Self {
        ConstrainedFloat::from_inner(T::exp(self.into_inner()))
    }

    fn exp2(self) -> Self {
        ConstrainedFloat::from_inner(T::exp2(self.into_inner()))
    }

    fn exp_m1(self) -> Self {
        ConstrainedFloat::from_inner(T::exp_m1(self.into_inner()))
    }

    fn log(self, base: Self) -> Self {
        ConstrainedFloat::from_inner(T::log(self.into_inner(), base.into_inner()))
    }

    fn ln(self) -> Self {
        ConstrainedFloat::from_inner(T::ln(self.into_inner()))
    }

    fn log2(self) -> Self {
        ConstrainedFloat::from_inner(T::log2(self.into_inner()))
    }

    fn log10(self) -> Self {
        ConstrainedFloat::from_inner(T::log10(self.into_inner()))
    }

    fn ln_1p(self) -> Self {
        ConstrainedFloat::from_inner(T::ln_1p(self.into_inner()))
    }

    fn hypot(self, other: Self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().hypot(other.into_inner()))
    }

    fn sin(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(self.into_inner().sin())
    }

    fn cos(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(self.into_inner().cos())
    }

    fn tan(self) -> Self {
        ConstrainedFloat::from_inner_unchecked(self.into_inner().tan())
    }

    fn asin(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().asin())
    }

    fn acos(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().acos())
    }

    fn atan(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().atan())
    }

    fn atan2(self, other: Self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().atan2(other.into_inner()))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.into_inner().sin_cos();
        (
            ConstrainedFloat::from_inner_unchecked(sin),
            ConstrainedFloat::from_inner_unchecked(cos),
        )
    }

    fn sinh(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().sinh())
    }

    fn cosh(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().cosh())
    }

    fn tanh(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().tanh())
    }

    fn asinh(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().asinh())
    }

    fn acosh(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().acosh())
    }

    fn atanh(self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().atanh())
    }
}

impl<T, P> Rem for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() % other.into_inner())
    }
}

impl<T, P> Rem<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn rem(self, other: T) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() % other)
    }
}

impl<T, P> RemAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn rem_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_inner(self.into_inner() % other.into_inner())
    }
}

impl<T, P> RemAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn rem_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_inner(self.into_inner() % other)
    }
}

impl<T, P> Signed for ConstrainedFloat<T, P>
where
    T: Float + Primitive + Signed,
    P: FloatConstraint<T> + ConstraintEq<T>,
{
    fn abs(&self) -> Self {
        ConstrainedFloat::from_inner_unchecked(self.into_inner().abs())
    }

    fn abs_sub(&self, other: &Self) -> Self {
        ConstrainedFloat::from_inner(self.into_inner().abs_sub(other.into_inner()))
    }

    fn signum(&self) -> Self {
        ConstrainedFloat::from_inner_unchecked(self.into_inner().signum())
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
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() - other.into_inner())
    }
}

impl<T, P> Sub<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        ConstrainedFloat::from_inner(self.into_inner() - other)
    }
}

impl<T, P> SubAssign for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = ConstrainedFloat::from_inner(self.into_inner() - other.into_inner())
    }
}

impl<T, P> SubAssign<T> for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn sub_assign(&mut self, other: T) {
        *self = ConstrainedFloat::from_inner(self.into_inner() - other)
    }
}

impl<T, P> Sum for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn sum<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(Zero::zero(), |a, b| a + b)
    }
}

impl<T, P> ToPrimitive for ConstrainedFloat<T, P>
where
    T: Float + Primitive + ToPrimitive,
    P: FloatConstraint<T>,
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

impl<T, P> UpperExp for ConstrainedFloat<T, P>
where
    T: Float + UpperExp + Primitive,
    P: FloatConstraint<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, P> Zero for ConstrainedFloat<T, P>
where
    T: Float + Primitive,
    P: FloatConstraint<T>,
{
    fn zero() -> Self {
        ConstrainedFloat::from_inner_unchecked(T::zero())
    }

    fn is_zero(&self) -> bool {
        T::is_zero(&self.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Finite, NotNan, Ordered, N32, R32};

    #[test]
    fn ordered_no_panic_on_inf() {
        let x: Ordered<f32> = 1.0.into();
        let y = x / 0.0;
        assert!(Infinite::is_infinite(y));
    }

    #[test]
    fn ordered_no_panic_on_nan() {
        let x: Ordered<f32> = 0.0.into();
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
    fn ordered_nan_eq() {
        let x: Ordered<f32> = (0.0 / 0.0).into();
        let y: Ordered<f32> = (0.0 / 0.0).into();
        assert_eq!(x, y);

        let z: Ordered<f32> =
            (<f32 as Infinite>::infinity() + <f32 as Infinite>::neg_infinity()).into();
        assert_eq!(x, z);

        let w: Ordered<f32> = (Real::sqrt(-1.0)).into();
        assert_eq!(x, w);
    }

    #[test]
    fn cmp_proxy_to_primitive() {
        // Compare a canonicalized `NaN` with a primitive `NaN` with a
        // different representation.
        let x: Ordered<f32> = (0.0 / 0.0).into();
        assert_eq!(x, f32::sqrt(-1.0));

        // Compare a canonicalized `INF` with a primitive `NaN`.
        let y: Ordered<f32> = (1.0 / 0.0).into();
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

    #[test]
    fn fmt() {
        let x: Ordered<f32> = 1.0.into();
        println!("{0} {0:e} {0:E} {0:?}", x);
        let y: NotNan<f32> = 1.0.into();
        println!("{0} {0:e} {0:E} {0:?}", y);
        let z: Finite<f32> = 1.0.into();
        println!("{0} {0:e} {0:E} {0:?}", z);
    }
}
