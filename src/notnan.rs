use num_traits::{Float, Num, One, Zero};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

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
        // TODO: Is it really safe to avoid NaN checks for this operation? What
        //       about other operations?
        NotNan::from_raw_float_unchecked(self.into_raw_float() + other.into_raw_float())
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
        NotNan::from_raw_float_unchecked(Float::max_value())
    }

    #[inline(always)]
    fn min_value() -> Self {
        NotNan::from_raw_float_unchecked(Float::min_value())
    }

    #[inline(always)]
    fn min_positive_value() -> Self {
        NotNan::from_raw_float_unchecked(Float::min_positive_value())
    }

    #[inline(always)]
    fn infinity() -> Self {
        NotNan::from_raw_float_unchecked(Float::infinity())
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        NotNan::from_raw_float_unchecked(Float::neg_infinity())
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        Float::is_infinite(self.into_raw_float())
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        Float::is_finite(self.into_raw_float())
    }

    #[inline(always)]
    fn neg_zero() -> Self {
        NotNan::from_raw_float_unchecked(Float::neg_zero())
    }

    #[inline(always)]
    fn is_sign_positive(self) -> bool {
        Float::is_sign_positive(self.into_raw_float())
    }

    #[inline(always)]
    fn is_sign_negative(self) -> bool {
        Float::is_sign_negative(self.into_raw_float())
    }

    #[inline(always)]
    fn classify(self) -> FpCategory {
        Float::classify(self.into_raw_float())
    }

    #[inline(always)]
    fn is_normal(self) -> bool {
        Float::is_normal(self.into_raw_float())
    }

    #[inline(always)]
    fn integer_decode(self) -> (u64, i16, i8) {
        Float::integer_decode(self.into_raw_float())
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
