//! Ordering and comparisons of IEEE 754 floating-point and other partially ordered values.
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
//! let (min, max) = match x.cmp_canonical_bits(&y) {
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

use crate::constraint::Constraint;
use crate::expression::{Defined, Expression, Undefined};
use crate::proxy::Proxy;
use crate::{with_primitives, NanEncoding, Primitive, ToCanonicalBits};

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
/// assert!(x.eq_canonical_bits(&y));
/// ```
pub trait CanonicalEq {
    fn eq_canonical_bits(&self, other: &Self) -> bool;
}

impl<T> CanonicalEq for T
where
    T: ToCanonicalBits,
{
    fn eq_canonical_bits(&self, other: &Self) -> bool {
        self.to_canonical_bits() == other.to_canonical_bits()
    }
}

impl<T, const N: usize> CanonicalEq for [T; N]
where
    T: CanonicalEq,
{
    fn eq_canonical_bits(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(a, b)| a.eq_canonical_bits(b))
    }
}

impl<T> CanonicalEq for [T]
where
    T: CanonicalEq,
{
    fn eq_canonical_bits(&self, other: &Self) -> bool {
        if self.len() == other.len() {
            self.iter()
                .zip(other.iter())
                .all(|(a, b)| a.eq_canonical_bits(b))
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
    // TODO: The naming convention for canonical forms and relations is a bit odd here: ordering
    //       considers a notion of canonical semantics, but does not compare
    //       `ToCanonicalBits::Bits`. Is there a convension that works better here? Alternatively,
    //       is there a good name for this despite bucking the convention?
    fn cmp_canonical_bits(&self, other: &Self) -> Ordering;
}

impl<T> CanonicalOrd for T
where
    // This implementation is bound on `Primitive` rather than something more general to exclude
    // `PartialOrd` implementations that do not comply with IEEE 754 floating-point partial
    // ordering. This must be implemented independently for proxy types.
    T: Primitive,
{
    fn cmp_canonical_bits(&self, other: &Self) -> Ordering {
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
    fn cmp_canonical_bits(&self, other: &Self) -> Ordering {
        match self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a.cmp_canonical_bits(b))
            .find(|ordering| *ordering != Ordering::Equal)
        {
            Some(ordering) => ordering,
            None => self.len().cmp(&other.len()),
        }
    }
}

pub trait UndefinedError {
    fn undefined() -> Self;
}

/// Pairwise ordering of types with intrinsic representations for undefined comparisons (or total
/// ordering).
///
/// `IntrinsicOrd` compares two values of the same type to produce a pairwise minimum and maximum.
/// This contrasts [`PartialOrd`], which expresses comparisons via the extrinsic type
/// [`Option<Ordering>`].
///
/// Some types have intrinsic representations for _undefined_, such as the `None` variant of
/// [`Option`] and `NaN`s for floating-point primitives. For these types, **regardless of having
/// only a partial ordering or a total ordering**, comparisons wherein any of the input values are
/// undefined also yield a value that represents undefined. For types with total ordering and no
/// representation for undefined, such as integer primitives, comparisons have no error conditions
/// and always yield a valid ordering.
///
/// This trait can be used in generic APIs to peform comparisons while ergonomically propogating
/// `NaN`s or other undefined values when a comparison cannot be performed. For floating-point
/// primitives, this mirrors the behavior of mathematical operations like addition, multiplication,
/// etc. with respect to `NaN`s.
///
/// See the [`min_or_undefined`] and [`max_or_undefined`] functions.
///
/// [`max_or_undefined`]: crate::cmp::max_or_undefined
/// [`min_or_undefined`]: crate::cmp::min_or_undefined
/// [`Option`]: core::option::Option
/// [`Option<Ordering>`]: core::cmp::Ordering
/// [`PartialOrd`]: core::cmp::PartialOrd
pub trait IntrinsicOrd: PartialOrd + Sized {
    /// Returns `true` if a value encodes _undefined_, otherwise `false`.
    ///
    /// Prefer this predicate over direct comparisons. For floating-point representations, `NaN` is
    /// considered undefined, but direct comparisons with `NaN` values should be avoided.
    fn is_undefined(&self) -> bool;

    /// Compares two values and returns their pairwise minimum and maximum.
    ///
    /// This function returns a representation of _undefined_ for both the minimum and maximum if
    /// either of the inputs are undefined or the inputs cannot be compared, **even if undefined
    /// values are ordered and the type has a total ordering**. Undefined values are always
    /// propagated.
    ///
    /// Some types have multiple representations of _undefined_. The representation returned by
    /// this function for undefined comparisons is arbitrary.
    ///
    /// # Examples
    ///
    /// Propagating `NaN` values when comparing proxy types with a total ordering:
    ///
    /// ```rust
    /// use decorum::cmp::{self, IntrinsicOrd};
    /// use decorum::{NanEncoding, Total};
    ///
    /// let x: Total<f64> = 0.0.into();
    /// let y: Total<f64> = (0.0 / 0.0).into(); // `NaN`.
    ///
    /// // `Total` provides a total ordering in which zero is less than `NaN`, but `NaN` is considered
    /// // undefined and is the result of the intrinsic comparison.
    /// assert!(y.is_undefined());
    /// assert!(cmp::min_or_undefined(x, y).is_undefined());
    /// ```
    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self);

    fn min_or_undefined(&self, other: &Self) -> Self {
        self.min_max_or_undefined(other).0
    }

    fn max_or_undefined(&self, other: &Self) -> Self {
        self.min_max_or_undefined(other).1
    }
}

impl<T, E> IntrinsicOrd for Expression<T, E>
where
    T: IntrinsicOrd,
    E: UndefinedError,
{
    fn is_undefined(&self) -> bool {
        Expression::is_undefined(self)
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        let undefined = || (Undefined(E::undefined()), Undefined(E::undefined()));
        match (self, other) {
            (Defined(ref a), Defined(ref b)) => {
                let (min, max) = a.min_max_or_undefined(b);
                if min.is_undefined() {
                    undefined()
                }
                else {
                    (Defined(min), Defined(max))
                }
            }
            _ => undefined(),
        }
    }
}

impl<T> IntrinsicOrd for Option<T>
where
    T: IntrinsicOrd,
{
    fn is_undefined(&self) -> bool {
        self.is_none()
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        let undefined = (None, None);
        match (self, other) {
            (Some(ref a), Some(ref b)) => {
                let (min, max) = a.min_max_or_undefined(b);
                if min.is_undefined() {
                    undefined
                }
                else {
                    (Some(min), Some(max))
                }
            }
            _ => undefined,
        }
    }
}

impl<T, C> IntrinsicOrd for Proxy<T, C>
where
    T: IntrinsicOrd + Primitive,
    C: Constraint,
{
    fn is_undefined(&self) -> bool {
        self.into_inner().is_nan()
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        let a = self.into_inner();
        let b = other.into_inner();
        let (min, max) = a.min_max_or_undefined(&b);
        // Both `min` and `max` are `NaN` if `a` and `b` are incomparable.
        if min.is_nan() {
            // This relies on the correctness of the implementation of `IntrinsicOrd` for `T`. For
            // constrained (and nonresidual) types like `ExtendedReal` and `Real`, `a` and `b` must
            // not be undefined (`NaN`) and so `min` and `max` also must not be undefined.
            let nan = Proxy::<_, C>::unchecked(T::NAN.into_inner());
            (nan, nan)
        }
        else {
            (Proxy::<_, C>::unchecked(min), Proxy::<_, C>::unchecked(max))
        }
    }
}

impl<T, E> IntrinsicOrd for Result<T, E>
where
    Self: PartialOrd,
    T: IntrinsicOrd,
    E: UndefinedError,
{
    fn is_undefined(&self) -> bool {
        self.is_err()
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        let undefined = || (Err(E::undefined()), Err(E::undefined()));
        match (self, other) {
            (Ok(ref a), Ok(ref b)) => {
                let (min, max) = a.min_max_or_undefined(b);
                if min.is_undefined() {
                    undefined()
                }
                else {
                    (Ok(min), Ok(max))
                }
            }
            _ => undefined(),
        }
    }
}

macro_rules! impl_total_intrinsic_ord {
    () => {
        impl_total_intrinsic_ord!(primitive => isize);
        impl_total_intrinsic_ord!(primitive => i8);
        impl_total_intrinsic_ord!(primitive => i16);
        impl_total_intrinsic_ord!(primitive => i32);
        impl_total_intrinsic_ord!(primitive => i64);
        impl_total_intrinsic_ord!(primitive => i128);
        impl_total_intrinsic_ord!(primitive => usize);
        impl_total_intrinsic_ord!(primitive => u8);
        impl_total_intrinsic_ord!(primitive => u16);
        impl_total_intrinsic_ord!(primitive => u32);
        impl_total_intrinsic_ord!(primitive => u64);
        impl_total_intrinsic_ord!(primitive => u128);
    };
    (primitive => $t:ty) => {
        impl IntrinsicOrd for $t {
            fn is_undefined(&self) -> bool {
                false
            }

            fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
                let (min, max) = partial_min_max(self, other).unwrap();
                (*min, *max)
            }
        }
    };
}
impl_total_intrinsic_ord!();

macro_rules! impl_float_intrinsic_ord {
    () => {
        with_primitives!(impl_float_intrinsic_ord);
    };
    (primitive => $t:ty) => {
        impl IntrinsicOrd for $t {
            fn is_undefined(&self) -> bool {
                self.is_nan()
            }

            fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
                match partial_min_max(self, other) {
                    // `NaN`s cannot be compared, so `min` and `max` cannot be undefined here.
                    Some((min, max)) => (*min, *max),
                    _ => (
                        <$t as NanEncoding>::NAN.into_inner(),
                        <$t as NanEncoding>::NAN.into_inner(),
                    ),
                }
            }
        }
    };
}
impl_float_intrinsic_ord!();

/// Partial maximum of types with intrinsic representations for undefined.
///
/// See the [`IntrinsicOrd`] trait.
///
/// [`IntrinsicOrd`]: crate::cmp::IntrinsicOrd
pub fn max_or_undefined<T>(a: T, b: T) -> T
where
    T: IntrinsicOrd,
{
    a.max_or_undefined(&b)
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
    a.min_or_undefined(&b)
}

fn partial_min_max<'t, T>(a: &'t T, b: &'t T) -> Option<(&'t T, &'t T)>
where
    T: PartialOrd,
{
    a.partial_cmp(b).map(|ordering| match ordering {
        Ordering::Less | Ordering::Equal => (a, b),
        _ => (b, a),
    })
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

        assert!(x.eq_canonical_bits(&y));
        assert!(xs.eq_canonical_bits(&ys));
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

        assert_eq!((zero, one), zero.min_max_or_undefined(&one));
        assert_eq!((zero, one), one.min_max_or_undefined(&zero));

        assert_eq!((nan, nan), nan.min_max_or_undefined(&zero));
        assert_eq!((nan, nan), zero.min_max_or_undefined(&nan));
        assert_eq!((nan, nan), nan.min_max_or_undefined(&nan));

        assert_eq!(nan, cmp::min_or_undefined(nan, zero));
        assert_eq!(nan, cmp::max_or_undefined(nan, zero));
        assert_eq!(nan, cmp::min_or_undefined(nan, nan));
        assert_eq!(nan, cmp::max_or_undefined(nan, nan));
    }
}
