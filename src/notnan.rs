use num_traits::{Bounded, Float, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use Real;
use hash;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct NotNan<T>(T)
where
    T: Float;

impl<T> NotNan<T>
where
    T: Float,
{
    pub fn from_raw_float(value: T) -> Result<Self, ()> {
        match value {
            ref value if value.is_nan() => Err(()),
            value => Ok(NotNan(value)),
        }
    }

    pub fn from_raw_float_unchecked(value: T) -> Self {
        NotNan(value)
    }

    pub fn into_raw_float(self) -> T {
        let NotNan(value) = self;
        value
    }
}

impl<T> AsRef<T> for NotNan<T>
where
    T: Float,
{
    fn as_ref(&self) -> &T {
        &self.0
    }
}

// Because of the reflexive implementation in core, this `Into` cannot be
// implemented over a type `T`.
impl Into<f32> for NotNan<f32> {
    fn into(self) -> f32 {
        self.into_raw_float()
    }
}

// Because of the reflexive implementation in core, this `Into` cannot be
// implemented over a type `T`.
impl Into<f64> for NotNan<f64> {
    fn into(self) -> f64 {
        self.into_raw_float()
    }
}

impl<T> Add for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() + other.into_raw_float()).unwrap()
    }
}

impl<T> Add<T> for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() + other).unwrap()
    }
}

impl<T> AddAssign for NotNan<T>
where
    T: Float,
{
    fn add_assign(&mut self, other: Self) {
        *self = NotNan::from_raw_float(self.into_raw_float() + other.into_raw_float()).unwrap()
    }
}

impl<T> AddAssign<T> for NotNan<T>
where
    T: Float,
{
    fn add_assign(&mut self, other: T) {
        *self = NotNan::from_raw_float(self.into_raw_float() + other).unwrap()
    }
}

impl<T> Bounded for NotNan<T>
where
    T: Float,
{
    fn min_value() -> Self {
        NotNan::from_raw_float_unchecked(T::min_value())
    }

    fn max_value() -> Self {
        NotNan::from_raw_float_unchecked(T::max_value())
    }
}

impl<T> Display for NotNan<T>
where
    T: Display + Float,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T> Div for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() / other.into_raw_float()).unwrap()
    }
}

impl<T> Div<T> for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() / other).unwrap()
    }
}

impl<T> DivAssign for NotNan<T>
where
    T: Float,
{
    fn div_assign(&mut self, other: Self) {
        *self = NotNan::from_raw_float(self.into_raw_float() / other.into_raw_float()).unwrap()
    }
}

impl<T> DivAssign<T> for NotNan<T>
where
    T: Float,
{
    fn div_assign(&mut self, other: T) {
        *self = NotNan::from_raw_float(self.into_raw_float() / other).unwrap()
    }
}

impl<T> Eq for NotNan<T>
where
    T: Float,
{
}

impl<T> FromPrimitive for NotNan<T>
where
    T: Float + FromPrimitive,
{
    fn from_i8(value: i8) -> Option<Self> {
        T::from_i8(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_u8(value: u8) -> Option<Self> {
        T::from_u8(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_i16(value: i16) -> Option<Self> {
        T::from_i16(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_u16(value: u16) -> Option<Self> {
        T::from_u16(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_i32(value: i32) -> Option<Self> {
        T::from_i32(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_u32(value: u32) -> Option<Self> {
        T::from_u32(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_i64(value: i64) -> Option<Self> {
        T::from_i64(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_u64(value: u64) -> Option<Self> {
        T::from_u64(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_isize(value: isize) -> Option<Self> {
        T::from_isize(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_usize(value: usize) -> Option<Self> {
        T::from_usize(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_f32(value: f32) -> Option<Self> {
        T::from_f32(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }

    fn from_f64(value: f64) -> Option<Self> {
        T::from_f64(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }
}

impl<T> Hash for NotNan<T>
where
    T: Float,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        hash::hash_float(self.into_raw_float(), state);
    }
}

impl<T> Mul for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() * other.into_raw_float()).unwrap()
    }
}

impl<T> Mul<T> for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() * other).unwrap()
    }
}

impl<T> MulAssign for NotNan<T>
where
    T: Float,
{
    fn mul_assign(&mut self, other: Self) {
        *self = NotNan::from_raw_float(self.into_raw_float() * other.into_raw_float()).unwrap()
    }
}

impl<T> MulAssign<T> for NotNan<T>
where
    T: Float,
{
    fn mul_assign(&mut self, other: T) {
        *self = NotNan::from_raw_float(self.into_raw_float() * other).unwrap()
    }
}

impl<T> Neg for NotNan<T>
where
    T: Float + Num,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        NotNan::from_raw_float_unchecked(-self.into_raw_float())
    }
}

impl<T> Num for NotNan<T>
where
    T: Float + Num,
{
    type FromStrRadixErr = ();

    fn from_str_radix(source: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(source, radix)
            .map_err(|_| ())
            .and_then(|value| NotNan::from_raw_float(value).map_err(|_| ()))
    }
}

impl<T> NumCast for NotNan<T>
where
    T: Float + Num,
{
    fn from<U>(value: U) -> Option<Self>
    where
        U: ToPrimitive,
    {
        T::from(value).and_then(|value| NotNan::from_raw_float(value).ok())
    }
}

impl<T> One for NotNan<T>
where
    T: Float,
{
    #[inline(always)]
    fn one() -> Self {
        NotNan::from_raw_float_unchecked(T::one())
    }
}

impl<T> Ord for NotNan<T>
where
    T: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.partial_cmp(other) {
            Some(order) => order,
            _ => panic!(),
        }
    }
}

impl<T> Real for NotNan<T>
where
    T: Float,
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
        NotNan::from_raw_float_unchecked(T::min_positive_value())
    }

    #[inline(always)]
    fn infinity() -> Self {
        NotNan::from_raw_float_unchecked(T::infinity())
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        NotNan::from_raw_float_unchecked(T::neg_infinity())
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        T::is_infinite(self.into_raw_float())
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        T::is_finite(self.into_raw_float())
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        NotNan::from_raw_float_unchecked(T::neg_zero())
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
}

impl<T> Rem for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() % other.into_raw_float()).unwrap()
    }
}

impl<T> Rem<T> for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn rem(self, other: T) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() % other).unwrap()
    }
}

impl<T> RemAssign for NotNan<T>
where
    T: Float,
{
    fn rem_assign(&mut self, other: Self) {
        *self = NotNan::from_raw_float(self.into_raw_float() % other.into_raw_float()).unwrap()
    }
}

impl<T> RemAssign<T> for NotNan<T>
where
    T: Float,
{
    fn rem_assign(&mut self, other: T) {
        *self = NotNan::from_raw_float(self.into_raw_float() % other).unwrap()
    }
}

impl<T> Signed for NotNan<T>
where
    T: Float + Signed,
{
    fn abs(&self) -> Self {
        NotNan::from_raw_float_unchecked(self.into_raw_float().abs())
    }

    fn abs_sub(&self, other: &Self) -> Self {
        NotNan::from_raw_float(self.into_raw_float().abs_sub(other.into_raw_float())).unwrap()
    }

    fn signum(&self) -> Self {
        NotNan::from_raw_float_unchecked(self.into_raw_float().signum())
    }

    fn is_positive(&self) -> bool {
        self.into_raw_float().is_positive()
    }

    fn is_negative(&self) -> bool {
        self.into_raw_float().is_negative()
    }
}

impl<T> Sub for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() - other.into_raw_float()).unwrap()
    }
}

impl<T> Sub<T> for NotNan<T>
where
    T: Float,
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        NotNan::from_raw_float(self.into_raw_float() - other).unwrap()
    }
}

impl<T> SubAssign for NotNan<T>
where
    T: Float,
{
    fn sub_assign(&mut self, other: Self) {
        *self = NotNan::from_raw_float(self.into_raw_float() - other.into_raw_float()).unwrap()
    }
}

impl<T> SubAssign<T> for NotNan<T>
where
    T: Float,
{
    fn sub_assign(&mut self, other: T) {
        *self = NotNan::from_raw_float(self.into_raw_float() - other).unwrap()
    }
}

impl<T> ToPrimitive for NotNan<T>
where
    T: Float + ToPrimitive,
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

impl<T> Zero for NotNan<T>
where
    T: Float,
{
    #[inline(always)]
    fn zero() -> Self {
        NotNan::from_raw_float_unchecked(T::zero())
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

    impl<'a, T> Deserialize<'a> for NotNan<T>
    where
        T: Deserialize<'a> + Float,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'a>,
        {
            let value = T::deserialize(deserializer)?;
            NotNan::from_raw_float(value)
                .map_err(|_| Error::invalid_value(Unexpected::Float(f64::NAN), &""))
        }
    }

    impl<T> Serialize for NotNan<T>
    where
        T: Float + Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.into_raw_float().serialize(serializer)
        }
    }
}
