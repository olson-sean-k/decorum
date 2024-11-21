//! Ordering and comparisons of IEEE 754 floating-point and other partially ordered types.
//!
//! This module provides traits and functions for partial and total orderings, in particular of
//! floating-point types. For primitive floating-point types, the following total ordering is
//! provided via the [`CanonicalEq`] and [`CanonicalOrd`] traits:
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
//! The [`EmptyOrd`] trait provides a particular ordering for types with a notion of empty
//! inhabitants. These inhabitants are considered incomparable, regardless of whether or not they
//! are ordered with respect to [`PartialOrd`] and [`Ord`]. For example, `None` is the empty
//! inhabitant of [`Option`] and so cannot be compared with [`EmptyOrd`].
//!
//! For floating-point types, `NaN`s are considered empty inhabitants. In this context, _empty_ can
//! be thought of as _undefined_, but note that **empty inhabitants are unrelated to proxy
//! constraints**. Unlike the standard ordering traits, [`EmptyOrd`] forwards empty inhabitants in
//! comparisons. For floating-point types, this importantly means that `NaN`s are forwarded
//! consistently in comparisons.
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
//! Computing a pairwise minimum that propagates `NaN`s with [`EmptyOrd`]:
//!
//! ```rust
//! use decorum::cmp;
//! use decorum::NanEncoding;
//!
//! let x = f64::NAN;
//! let y = 1.0f64;
//!
//! // `NaN` is considered an empty inhabitant in this ordering, so `min` is assigned `NaN` in this
//! // example, regardless of the order of parameters.
//! let min = cmp::min_or_empty(x, y);
//! ```
//!
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

pub trait EmptyInhabitant {
    fn empty() -> Self;
}

impl EmptyInhabitant for () {
    fn empty() -> Self {}
}

impl<T> EmptyInhabitant for Option<T> {
    #[inline(always)]
    fn empty() -> Self {
        None
    }
}

impl<T, E> EmptyInhabitant for Result<T, E>
where
    E: EmptyInhabitant,
{
    #[inline(always)]
    fn empty() -> Self {
        Err(E::empty())
    }
}

/// Defines an ordering for types that (may) have empty inhabitants.
///
/// An empty inhabitant is an intrinsic value of a type that is considered incomparable in this
/// ordering, regardless of [`PartialOrd`] and [`Ord`] implementations. If an empty inhabitant is
/// compared, then the comparison is considered undefined and the output is an empty inhabitant.
///
/// For floating-point types, `NaN`s are considered empty inhabitants.
///
/// `EmptyOrd` can be implemented for types with no empty inhabitants. Notably, this trait is
/// implemented by totally ordered primitive numeric types. This better supports types that are
/// defined by conditional compilation.
///
/// See [`min_or_empty`] and [`max_or_empty`].
pub trait EmptyOrd: PartialOrd {
    type Empty;

    fn from_empty(empty: Self::Empty) -> Self;

    fn is_empty(&self) -> bool;

    fn cmp_empty(&self, other: &Self) -> Result<Ordering, Self::Empty>;
}

impl<T> EmptyOrd for Option<T>
where
    T: Ord,
{
    type Empty = Self;

    #[inline(always)]
    fn from_empty(empty: <Self as EmptyOrd>::Empty) -> Self {
        empty
    }

    fn is_empty(&self) -> bool {
        self.is_none()
    }

    fn cmp_empty(&self, other: &Self) -> Result<Ordering, Self::Empty> {
        self.as_ref()
            .zip(other.as_ref())
            .map_or_else(|| Err(EmptyInhabitant::empty()), |(a, b)| Ok(a.cmp(b)))
    }
}

impl<T, E> EmptyOrd for Result<T, E>
where
    Self: PartialOrd,
    T: Ord,
    E: EmptyInhabitant,
{
    type Empty = Self;

    #[inline(always)]
    fn from_empty(empty: Self::Empty) -> Self {
        empty
    }

    fn is_empty(&self) -> bool {
        self.is_err()
    }

    fn cmp_empty(&self, other: &Self) -> Result<Ordering, Self::Empty> {
        match (self.as_ref(), other.as_ref()) {
            (Ok(a), Ok(b)) => Ok(a.cmp(b)),
            _ => Err(EmptyInhabitant::empty()),
        }
    }
}

macro_rules! impl_empty_ord_for_float_primitive {
    () => {
        with_primitives!(impl_empty_ord_for_float_primitive);
    };
    (primitive => $t:ty) => {
        impl EmptyOrd for $t {
            type Empty = Self;

            #[inline(always)]
            fn from_empty(empty: Self::Empty) -> Self {
                empty
            }

            fn is_empty(&self) -> bool {
                self.is_nan()
            }

            fn cmp_empty(&self, other: &Self) -> Result<Ordering, Self::Empty> {
                self.partial_cmp(other)
                    .ok_or_else(|| EmptyInhabitant::empty())
            }
        }
    };
}
impl_empty_ord_for_float_primitive!();

macro_rules! impl_empty_ord_for_total_primitive {
    () => {
        impl_empty_ord_for_total_primitive!(primitive => isize);
        impl_empty_ord_for_total_primitive!(primitive => i8);
        impl_empty_ord_for_total_primitive!(primitive => i16);
        impl_empty_ord_for_total_primitive!(primitive => i32);
        impl_empty_ord_for_total_primitive!(primitive => i64);
        impl_empty_ord_for_total_primitive!(primitive => i128);
        impl_empty_ord_for_total_primitive!(primitive => usize);
        impl_empty_ord_for_total_primitive!(primitive => u8);
        impl_empty_ord_for_total_primitive!(primitive => u16);
        impl_empty_ord_for_total_primitive!(primitive => u32);
        impl_empty_ord_for_total_primitive!(primitive => u64);
        impl_empty_ord_for_total_primitive!(primitive => u128);
    };
    (primitive => $t:ty) => {
        impl EmptyOrd for $t {
            type Empty = Infallible;

            fn from_empty(_: Self::Empty) -> Self {
                unreachable!()
            }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                false
            }

            #[inline(always)]
            fn cmp_empty(&self, other: &Self) -> Result<Ordering, Self::Empty> {
                Ok(self.cmp(other))
            }
        }
    };
}
impl_empty_ord_for_total_primitive!();

macro_rules! impl_empty_inhabitant_for_float_primitive {
    () => {
        with_primitives!(impl_empty_inhabitant_for_float_primitive);
    };
    (primitive => $t:ty) => {
        impl EmptyInhabitant for $t {
            #[inline(always)]
            fn empty() -> Self {
                Self::NAN
            }
        }
    };
}
impl_empty_inhabitant_for_float_primitive!();

/// Pairwise maximum for types that may have an empty inhabitant that is incomparable.
///
/// See the [`EmptyOrd`] trait.
pub fn max_or_empty<T>(a: T, b: T) -> T
where
    T: EmptyOrd,
{
    match a.cmp_empty(&b) {
        Ok(Ordering::Less | Ordering::Equal) => b,
        Ok(Ordering::Greater) => a,
        Err(empty) => T::from_empty(empty),
    }
}

/// Pairwise minimum for types that may have an empty inhabitant that is incomparable.
///
/// See the [`EmptyOrd`] trait.
pub fn min_or_empty<T>(a: T, b: T) -> T
where
    T: EmptyOrd,
{
    match a.cmp_empty(&b) {
        Ok(Ordering::Less | Ordering::Equal) => a,
        Ok(Ordering::Greater) => b,
        Err(empty) => T::from_empty(empty),
    }
}

/// Pairwise ordering for types that may have an empty inhabitant that is incomparable.
///
/// The output tuple contains either the minimum and maximum (in that order) or empty inhabitants.
///
/// See the [`EmptyOrd`] trait.
pub fn min_max_or_empty<T>(a: T, b: T) -> (T, T)
where
    T: EmptyOrd,
    T::Empty: Copy,
{
    match a.cmp_empty(&b) {
        Ok(Ordering::Less | Ordering::Equal) => (a, b),
        Ok(Ordering::Greater) => (b, a),
        Err(empty) => (T::from_empty(empty), T::from_empty(empty)),
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::cmp::{self, CanonicalEq, EmptyOrd};
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
    fn empty_ord_option() {
        let zero = Some(0u64);
        let one = Some(1u64);

        assert_eq!(zero, cmp::min_or_empty(zero, one));
        assert_eq!(one, cmp::max_or_empty(zero, one));
        assert!(cmp::min_or_empty(None, zero).is_empty());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn empty_ord_primitive() {
        let zero = 0.0f64;
        let one = 1.0f64;

        assert_eq!(zero, cmp::min_or_empty(zero, one));
        assert_eq!(one, cmp::max_or_empty(zero, one));
        assert!(cmp::min_or_empty(f64::NAN, zero).is_empty());
    }

    #[test]
    fn empty_ord_proxy() {
        let nan = Total::<f64>::NAN;
        let zero = Total::zero();
        let one = Total::one();

        assert_eq!((zero, one), cmp::min_max_or_empty(zero, one));
        assert_eq!((zero, one), cmp::min_max_or_empty(one, zero));

        assert_eq!((nan, nan), cmp::min_max_or_empty(nan, zero));
        assert_eq!((nan, nan), cmp::min_max_or_empty(zero, nan));
        assert_eq!((nan, nan), cmp::min_max_or_empty(nan, nan));

        assert_eq!(nan, cmp::min_or_empty(nan, zero));
        assert_eq!(nan, cmp::max_or_empty(nan, zero));
        assert_eq!(nan, cmp::min_or_empty(nan, nan));
        assert_eq!(nan, cmp::max_or_empty(nan, nan));
    }
}
