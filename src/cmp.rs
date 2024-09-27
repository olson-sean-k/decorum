//! Ordering and comparisons of IEEE 754 floating-point and other partially ordered types.
//!
//! This module provides traits and functions for total ordering of floating-point values and
//! handling partial ordering via intrinsic types. For primitive floating-point types, the
//! following total ordering is provided via the [`CanonicalEq`] and [`CanonicalOrd`] traits:
//!
//! $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
//!
//! Note that both zero and `NaN` have more than one representation in IEEE 754 encoding. Given the
//! set of zero representations $Z$ and set of `NaN` representations $N$, this ordering coalesces
//! `-0`, `+0`, and `NaN`s such that:
//!
//! $$
//! \begin{aligned}
//! a=b&\mid a\in{Z},~b\in{Z}\cr\[1em\]
//! a=b&\mid a\in{N},~b\in{N}\cr\[1em\]
//! n>x&\mid n\in{N},~x\notin{N}
//! \end{aligned}
//! $$
//!
//! These same semantics are used in the [`Eq`] and [`Ord`] implementations for the [`Total`]
//! proxy.
//!
//! # Examples
//!
//! Comparing `f64` values using a total ordering:
//!
//! ```rust
//! use core::cmp::Ordering;
//! use decorum::cmp::CanonicalOrd;
//! use decorum::NanEncoding;
//!
//! let x = f64::NAN;
//! let y = 1.0f64;
//!
//! let (min, max) = match x.cmp_canonical(&y) {
//!     Ordering::Less | Ordering::Equal => (x, y),
//!     _ => (y, x),
//! };
//! ```
//!
//! Computing a pairwise minimum that propagates `NaN`s:
//!
//! ```rust
//! use decorum::cmp;
//! use decorum::NanEncoding;
//!
//! let x = f64::NAN;
//! let y = 1.0f64;
//!
//! // `NaN` is incomparable and represents an undefined computation with respect to ordering, so
//! // `min` is assigned a `NaN` value in this example.
//! let min = cmp::min_or_undefined(x, y);
//! ```
//!
//! [`CanonicalEq`]: crate::cmp::CanonicalEq
//! [`CanonicalOrd`]: crate::cmp::CanonicalOrd
//! [`Eq`]: core::cmp::Eq
//! [`Ord`]: core::cmp::Ord
//! [`Total`]: crate::Total

use core::cmp::Ordering;
use core::convert::Infallible;

use crate::{with_primitives, Primitive, ToCanonical};

/// Total equivalence relation of IEEE 754 floating-point encoded types.
///
/// `CanonicalEq` agrees with the total ordering provided by `CanonicalOrd`. See the module
/// documentation for more. Given the set of `NaN` representations $N$, `CanonicalEq` expresses:
///
/// $$
/// \begin{aligned}
/// a=b&\mid a\in{N},~b\in{N}\cr\[1em\]
/// n\ne x&\mid n\in{N},~x\notin{N}
/// \end{aligned}
/// $$
///
/// # Examples
///
/// Comparing `NaN`s using primitive floating-point types:
///
/// ```rust
/// use decorum::cmp::CanonicalEq;
///
/// let x = 0.0f64 / 0.0; // `NaN`.
/// let y = f64::INFINITY - f64::INFINITY; // `NaN`.
///
/// assert!(x.eq_canonical(&y));
/// ```
pub trait CanonicalEq {
    fn eq_canonical(&self, other: &Self) -> bool;
}

impl<T> CanonicalEq for T
where
    T: ToCanonical,
{
    fn eq_canonical(&self, other: &Self) -> bool {
        self.to_canonical() == other.to_canonical()
    }
}

impl<T, const N: usize> CanonicalEq for [T; N]
where
    T: CanonicalEq,
{
    fn eq_canonical(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.eq_canonical(b))
    }
}

impl<T> CanonicalEq for [T]
where
    T: CanonicalEq,
{
    fn eq_canonical(&self, other: &Self) -> bool {
        if self.len() == other.len() {
            self.iter()
                .zip(other.iter())
                .all(|(a, b)| a.eq_canonical(b))
        }
        else {
            false
        }
    }
}

/// Total ordering of IEEE 754 floating-point encoded types.
///
/// `CanonicalOrd` expresses the total ordering:
///
/// $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
///
/// This trait can be used to compare primitive floating-point types without the need to wrap them
/// within a proxy type. See the module documentation for more about the ordering used by
/// `CanonicalOrd` and proxy types.
pub trait CanonicalOrd {
    fn cmp_canonical(&self, other: &Self) -> Ordering;
}

impl<T> CanonicalOrd for T
where
    // This implementation is bound on `Primitive` rather than something more general to exclude
    // `PartialOrd` implementations that do not comply with IEEE 754 floating-point partial
    // ordering. This must be implemented independently for proxy types.
    T: Primitive,
{
    fn cmp_canonical(&self, other: &Self) -> Ordering {
        match self.partial_cmp(other) {
            Some(ordering) => ordering,
            None => {
                if self.is_nan() {
                    if other.is_nan() {
                        Ordering::Equal
                    }
                    else {
                        Ordering::Greater
                    }
                }
                else {
                    Ordering::Less
                }
            }
        }
    }
}

impl<T> CanonicalOrd for [T]
where
    T: CanonicalOrd,
{
    fn cmp_canonical(&self, other: &Self) -> Ordering {
        match self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a.cmp_canonical(b))
            .find(|ordering| *ordering != Ordering::Equal)
        {
            Some(ordering) => ordering,
            None => self.len().cmp(&other.len()),
        }
    }
}

pub trait Undefined {
    fn undefined() -> Self;
}

impl<T> Undefined for Option<T> {
    #[inline(always)]
    fn undefined() -> Self {
        None
    }
}

impl<T, E> Undefined for Result<T, E>
where
    E: Undefined,
{
    #[inline(always)]
    fn undefined() -> Self {
        Err(E::undefined())
    }
}

pub trait IntrinsicOrd: PartialOrd {
    type Undefined;

    fn from_undefined(undefined: Self::Undefined) -> Self;

    fn is_undefined(&self) -> bool;

    fn intrinsic_cmp(&self, other: &Self) -> Result<Ordering, Self::Undefined>;
}

impl<T> IntrinsicOrd for Option<T>
where
    T: IntrinsicOrd,
{
    type Undefined = Self;

    #[inline(always)]
    fn from_undefined(undefined: <Self as IntrinsicOrd>::Undefined) -> Self {
        undefined
    }

    fn is_undefined(&self) -> bool {
        self.is_none()
    }

    fn intrinsic_cmp(&self, other: &Self) -> Result<Ordering, Self::Undefined> {
        self.as_ref().zip(other.as_ref()).map_or_else(
            || Err(Undefined::undefined()),
            |(a, b)| a.intrinsic_cmp(b).map_err(|_| Undefined::undefined()),
        )
    }
}

impl<T, E> IntrinsicOrd for Result<T, E>
where
    Self: PartialOrd,
    T: IntrinsicOrd,
    E: Undefined,
{
    type Undefined = Self;

    #[inline(always)]
    fn from_undefined(undefined: Self::Undefined) -> Self {
        undefined
    }

    fn is_undefined(&self) -> bool {
        self.is_err()
    }

    fn intrinsic_cmp(&self, other: &Self) -> Result<Ordering, Self::Undefined> {
        match (self.as_ref(), other.as_ref()) {
            (Ok(a), Ok(b)) => a.intrinsic_cmp(b).map_err(|_| Undefined::undefined()),
            _ => Err(Undefined::undefined()),
        }
    }
}

macro_rules! impl_intrinsic_ord_for_float_primitive {
    () => {
        with_primitives!(impl_intrinsic_ord_for_float_primitive);
    };
    (primitive => $t:ty) => {
        impl IntrinsicOrd for $t {
            type Undefined = Self;

            #[inline(always)]
            fn from_undefined(undefined: Self::Undefined) -> Self {
                undefined
            }

            fn is_undefined(&self) -> bool {
                self.is_nan()
            }

            fn intrinsic_cmp(&self, other: &Self) -> Result<Ordering, Self::Undefined> {
                self.partial_cmp(other)
                    .ok_or_else(|| Undefined::undefined())
            }
        }
    };
}
impl_intrinsic_ord_for_float_primitive!();

macro_rules! impl_intrinsic_ord_for_total_primitive {
    () => {
        impl_intrinsic_ord_for_total_primitive!(primitive => isize);
        impl_intrinsic_ord_for_total_primitive!(primitive => i8);
        impl_intrinsic_ord_for_total_primitive!(primitive => i16);
        impl_intrinsic_ord_for_total_primitive!(primitive => i32);
        impl_intrinsic_ord_for_total_primitive!(primitive => i64);
        impl_intrinsic_ord_for_total_primitive!(primitive => i128);
        impl_intrinsic_ord_for_total_primitive!(primitive => usize);
        impl_intrinsic_ord_for_total_primitive!(primitive => u8);
        impl_intrinsic_ord_for_total_primitive!(primitive => u16);
        impl_intrinsic_ord_for_total_primitive!(primitive => u32);
        impl_intrinsic_ord_for_total_primitive!(primitive => u64);
        impl_intrinsic_ord_for_total_primitive!(primitive => u128);
    };
    (primitive => $t:ty) => {
        impl IntrinsicOrd for $t {
            type Undefined = Infallible;

            fn from_undefined(_: Self::Undefined) -> Self {
                unreachable!()
            }
            #[inline(always)]
            fn is_undefined(&self) -> bool {
                false
            }

            #[inline(always)]
            fn intrinsic_cmp(&self, other: &Self) -> Result<Ordering, Self::Undefined> {
                Ok(self.cmp(other))
            }
        }
    };
}
impl_intrinsic_ord_for_total_primitive!();

macro_rules! impl_undefined_for_float_primitive {
    () => {
        with_primitives!(impl_undefined_for_float_primitive);
    };
    (primitive => $t:ty) => {
        impl Undefined for $t {
            #[inline(always)]
            fn undefined() -> Self {
                Self::NAN
            }
        }
    };
}
impl_undefined_for_float_primitive!();

/// Partial maximum of types with intrinsic representations for undefined.
///
/// See the [`IntrinsicOrd`] trait.
///
/// [`IntrinsicOrd`]: crate::cmp::IntrinsicOrd
pub fn max_or_undefined<T>(a: T, b: T) -> T
where
    T: IntrinsicOrd,
{
    match a.intrinsic_cmp(&b) {
        Ok(Ordering::Less | Ordering::Equal) => b,
        Ok(Ordering::Greater) => a,
        Err(undefined) => T::from_undefined(undefined),
    }
}

/// Partial minimum of types with intrinsic representations for undefined.
///
/// See the [`IntrinsicOrd`] trait.
///
/// [`IntrinsicOrd`]: crate::cmp::IntrinsicOrd
pub fn min_or_undefined<T>(a: T, b: T) -> T
where
    T: IntrinsicOrd,
{
    match a.intrinsic_cmp(&b) {
        Ok(Ordering::Less | Ordering::Equal) => a,
        Ok(Ordering::Greater) => b,
        Err(undefined) => T::from_undefined(undefined),
    }
}

pub fn min_max_or_undefined<T>(a: T, b: T) -> (T, T)
where
    T: Copy + IntrinsicOrd,
{
    match a.intrinsic_cmp(&b) {
        Ok(Ordering::Less | Ordering::Equal) => (a, b),
        Ok(Ordering::Greater) => (b, a),
        Err(undefined) => {
            let undefined = T::from_undefined(undefined);
            (undefined, undefined)
        }
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::cmp::{self, CanonicalEq, IntrinsicOrd};
    use crate::{NanEncoding, Total};

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::zero_divided_by_zero)]
    fn primitive_eq() {
        let x = 0.0f64 / 0.0f64; // `NaN`.
        let y = f64::INFINITY + f64::NEG_INFINITY; // `NaN`.
        let xs = [1.0f64, f64::NAN, f64::INFINITY];
        let ys = [1.0f64, f64::NAN, f64::INFINITY];

        assert!(x.eq_canonical(&y));
        assert!(xs.eq_canonical(&ys));
    }

    #[test]
    fn intrinsic_ord_option() {
        let zero = Some(0u64);
        let one = Some(1u64);

        assert_eq!(zero, cmp::min_or_undefined(zero, one));
        assert_eq!(one, cmp::max_or_undefined(zero, one));
        assert!(cmp::min_or_undefined(None, zero).is_undefined());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn intrinsic_ord_primitive() {
        let zero = 0.0f64;
        let one = 1.0f64;

        assert_eq!(zero, cmp::min_or_undefined(zero, one));
        assert_eq!(one, cmp::max_or_undefined(zero, one));
        assert!(cmp::min_or_undefined(f64::NAN, zero).is_undefined());
    }

    #[test]
    fn intrinsic_ord_proxy() {
        let nan = Total::<f64>::NAN;
        let zero = Total::zero();
        let one = Total::one();

        assert_eq!((zero, one), cmp::min_max_or_undefined(zero, one));
        assert_eq!((zero, one), cmp::min_max_or_undefined(one, zero));

        assert_eq!((nan, nan), cmp::min_max_or_undefined(nan, zero));
        assert_eq!((nan, nan), cmp::min_max_or_undefined(zero, nan));
        assert_eq!((nan, nan), cmp::min_max_or_undefined(nan, nan));

        assert_eq!(nan, cmp::min_or_undefined(nan, zero));
        assert_eq!(nan, cmp::max_or_undefined(nan, zero));
        assert_eq!(nan, cmp::min_or_undefined(nan, nan));
        assert_eq!(nan, cmp::max_or_undefined(nan, nan));
    }
}
