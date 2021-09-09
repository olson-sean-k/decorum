//! Ordering and comparisons.
//!
//! This module provides traits and functions for comparing floating-point and
//! other partially ordered values. For primitive floating-point types, a total
//! ordering is provided via the [`FloatEq`] and [`FloatOrd`] traits:
//!
//! $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
//!
//! Note that both zero and `NaN` have more than one representation in IEEE-754
//! encoding. Given the set of zero representations $Z$ and set of `NaN`
//! representations $N$, this ordering coalesces `-0`, `+0`, and `NaN`s such
//! that:
//!
//! $$
//! \begin{aligned}
//! a=b&\mid a\in{Z},~b\in{Z}\cr\[1em\]
//! a=b&\mid a\in{N},~b\in{N}\cr\[1em\]
//! n>x&\mid n\in{N},~x\notin{N}
//! \end{aligned}
//! $$
//!
//! These same semantics are used in the [`Eq`] and [`Ord`] implementations for
//! [`Proxy`], which includes the [`Total`], [`NotNan`], and [`Finite`] type
//! definitions.
//!
//! # Examples
//!
//! Comparing `f64` values using a total ordering:
//!
//! ```rust
//! use core::cmp::Ordering;
//! use decorum::cmp::FloatOrd;
//! use decorum::Nan;
//!
//! let x = f64::NAN;
//! let y = 1.0f64;
//!
//! let (min, max) = match x.float_cmp(&y) {
//!     Ordering::Less | Ordering::Equal => (x, y),
//!     _ => (y, x),
//! };
//! ```
//!
//! Computing a pairwise minimum that propagates `NaN`s:
//!
//! ```rust
//! use decorum::cmp;
//! use decorum::Nan;
//!
//! let x = f64::NAN;
//! let y = 1.0f64;
//!
//! // `Nan` is incomparable and represents an undefined computation with respect to
//! // ordering, so `min` is assigned a `NaN` value in this example.
//! let min = cmp::min_or_undefined(x, y);
//! ```
//!
//! [`Eq`]: core::cmp::Eq
//! [`Finite`]: crate::Finite
//! [`NotNan`]: crate::NotNan
//! [`Ord`]: core::cmp::Ord
//! [`Proxy`]: crate::Proxy
//! [`Total`]: crate::Total

use core::cmp::Ordering;

use crate::constraint::Constraint;
use crate::proxy::Proxy;
use crate::{Float, Nan, Primitive, ToCanonicalBits};

/// Equivalence relation for floating-point primitives.
///
/// `FloatEq` agrees with the total ordering provided by `FloatOrd`. See the
/// module documentation for more. Importantly, given the set of `NaN`
/// representations $N$, `FloatEq` expresses:
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
/// use decorum::cmp::FloatEq;
/// use decorum::Infinite;
///
/// let x = 0.0f64 / 0.0; // `NaN`.
/// let y = f64::INFINITY - f64::INFINITY; // `NaN`.
///
/// assert!(x.float_eq(&y));
/// ```
pub trait FloatEq {
    fn float_eq(&self, other: &Self) -> bool;
}

impl<T> FloatEq for T
where
    T: Float + Primitive,
{
    fn float_eq(&self, other: &Self) -> bool {
        self.to_canonical_bits() == other.to_canonical_bits()
    }
}

impl<T> FloatEq for [T]
where
    T: Float + Primitive,
{
    fn float_eq(&self, other: &Self) -> bool {
        if self.len() == other.len() {
            self.iter().zip(other.iter()).all(|(a, b)| a.float_eq(b))
        }
        else {
            false
        }
    }
}

/// Total ordering of primitive floating-point types.
///
/// `FloatOrd` expresses the total ordering:
///
/// $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
///
/// This trait can be used to compare primitive floating-point types without the
/// need to wrap them within a proxy type. See the module documentation for more
/// about the ordering used by `FloatOrd` and proxy types.
pub trait FloatOrd {
    fn float_cmp(&self, other: &Self) -> Ordering;
}

impl<T> FloatOrd for T
where
    T: Float + Primitive,
{
    fn float_cmp(&self, other: &Self) -> Ordering {
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

impl<T> FloatOrd for [T]
where
    T: Float + Primitive,
{
    fn float_cmp(&self, other: &Self) -> Ordering {
        match self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a.float_cmp(b))
            .find(|ordering| *ordering != Ordering::Equal)
        {
            Some(ordering) => ordering,
            None => self.len().cmp(&other.len()),
        }
    }
}

/// Partial ordering of types with intrinsic representations for undefined
/// comparisons.
///
/// `IntrinsicOrd` provides similar functionality to [`PartialOrd`], but exposes
/// a pairwise minimum-maximum API that is closed with respect to type and, for
/// types without a total ordering, is only implemented for types that
/// additionally have intrinsic representations for _undefined_, such as the
/// `None` variant of [`Option`] and `NaN`s for floating-point primitives.
///
/// This trait is also implemented for numeric types with total orderings, and
/// can be used for comparisons that propagate `NaN`s for floating-point
/// primitives (unlike [`PartialOrd`], which expresses comparisons of types `T`
/// and `U` with the extrinsic type [`Option<Ordering>`]).
///
/// See the [`min_or_undefined`] and [`max_or_undefined`] functions.
///
/// [`max_or_undefined`]: crate::cmp::max_or_undefined
/// [`min_or_undefined`]: crate::cmp::min_or_undefined
/// [`Option`]: core::option::Option
/// [`Option<Ordering>`]: core::cmp::Ordering
/// [`PartialOrd`]: core::cmp::PartialOrd
pub trait IntrinsicOrd: Copy + PartialOrd + Sized {
    /// Returns `true` if a value encodes _undefined_, otherwise `false`.
    ///
    /// Prefer this predicate over direct comparisons. For floating-point
    /// representations, `NaN` is considered undefined, but direct comparisons
    /// with `NaN` values should be avoided.
    fn is_undefined(&self) -> bool;

    /// Compares two values and returns their pairwise minimum and maximum.
    ///
    /// This function returns a representation of _undefined_ for both the
    /// minimum and maximum if either of the inputs are _undefined_ or the
    /// inputs cannot be compared, **even if undefined values are ordered or the
    /// type has a total ordering**. Undefined values are always propagated.
    ///
    /// # Examples
    ///
    /// Propagating `NaN` values when comparing proxy types with a total
    /// ordering:
    ///
    /// ```rust
    /// use decorum::cmp::{self, IntrinsicOrd};
    /// use decorum::{Nan, Total};
    ///
    /// let x: Total<f64> = 0.0.into();
    /// let y: Total<f64> = (0.0 / 0.0).into(); // `NaN`.
    ///
    /// // `Total` provides a total ordering in which zero is less than `NaN`, but `NaN`
    /// // is considered undefined and is the result of the intrinsic comparison.
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

impl<T> IntrinsicOrd for Option<T>
where
    T: Copy + PartialOrd,
{
    fn is_undefined(&self) -> bool {
        self.is_none()
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        match (self.as_ref(), other.as_ref()) {
            (Some(a), Some(b)) => match a.partial_cmp(b) {
                Some(ordering) => match ordering {
                    Ordering::Less | Ordering::Equal => (Some(*a), Some(*b)),
                    _ => (Some(*b), Some(*a)),
                },
                _ => (None, None),
            },
            _ => (None, None),
        }
    }
}

// Note that it is not necessary for `NaN` to be a member of the constraint.
// This implementation explicitly detects `NaN`s and emits `NaN` as the
// maximum and minimum (it does not use `FloatOrd`).
impl<T, P> IntrinsicOrd for Proxy<T, P>
where
    T: Float + IntrinsicOrd + Primitive,
    P: Constraint<T>,
{
    fn is_undefined(&self) -> bool {
        self.into_inner().is_nan()
    }

    fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
        // This function operates on primitive floating-point values. This
        // avoids the need for implementations for each combination of proxy and
        // constraint (proxy types do not always implement `Nan`, but primitive
        // types do).
        let a = self.into_inner();
        let b = other.into_inner();
        let (min, max) = a.min_max_or_undefined(&b);
        // Both `min` and `max` are `NaN` if `a` and `b` are incomparable.
        if min.is_nan() {
            let nan = Proxy::assert(T::NAN);
            (nan, nan)
        }
        else {
            (Proxy::assert(min), Proxy::assert(max))
        }
    }
}

macro_rules! impl_intrinsic_ord {
    (no_nan_total => $t:ty) => {
        impl IntrinsicOrd for $t {
            fn is_undefined(&self) -> bool {
                false
            }

            fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
                match self.partial_cmp(other) {
                    Some(ordering) => match ordering {
                        Ordering::Less | Ordering::Equal => (*self, *other),
                        _ => (*other, *self),
                    },
                    _ => unreachable!(),
                }
            }
        }
    };
    (nan_partial => $t:ty) => {
        impl IntrinsicOrd for $t {
            fn is_undefined(&self) -> bool {
                self.is_nan()
            }

            fn min_max_or_undefined(&self, other: &Self) -> (Self, Self) {
                match self.partial_cmp(other) {
                    Some(ordering) => match ordering {
                        Ordering::Less | Ordering::Equal => (*self, *other),
                        _ => (*other, *self),
                    },
                    _ => (Nan::NAN, Nan::NAN),
                }
            }
        }
    };
}
impl_intrinsic_ord!(no_nan_total => isize);
impl_intrinsic_ord!(no_nan_total => i8);
impl_intrinsic_ord!(no_nan_total => i16);
impl_intrinsic_ord!(no_nan_total => i32);
impl_intrinsic_ord!(no_nan_total => i64);
impl_intrinsic_ord!(no_nan_total => i128);
impl_intrinsic_ord!(no_nan_total => usize);
impl_intrinsic_ord!(no_nan_total => u8);
impl_intrinsic_ord!(no_nan_total => u16);
impl_intrinsic_ord!(no_nan_total => u32);
impl_intrinsic_ord!(no_nan_total => u64);
impl_intrinsic_ord!(no_nan_total => u128);
impl_intrinsic_ord!(nan_partial => f32);
impl_intrinsic_ord!(nan_partial => f64);

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

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::cmp::{self, FloatEq, IntrinsicOrd};
    use crate::{Nan, Total};

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::zero_divided_by_zero)]
    fn primitive_eq() {
        let x = 0.0f64 / 0.0f64; // `NaN`.
        let y = f64::INFINITY + f64::NEG_INFINITY; // `NaN`.
        let xs = [1.0f64, f64::NAN, f64::INFINITY];
        let ys = [1.0f64, f64::NAN, f64::INFINITY];

        assert!(x.float_eq(&y));
        assert!(xs.float_eq(&ys));
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
