//! Constraints on floating-point values.
//!
//! The `FloatConstraint` trait describes a constraint by filtering illegal
//! values and optionally supporting variants of `Ord`, `Eq`, etc. that are
//! also provided in this module. These analogous traits determine if and how
//! constrained values support these operations and in turn whether or not a
//! proxy using a constraint does too.
//!
//! The unit type `()` is used to implement an empty constraint that has no
//! illegal values.

use core::cmp::Ordering;
use core::marker::PhantomData;
use num_traits::Float;

use canonical;
use Primitive;

pub trait ConstraintEq<T>
where
    T: Float + Primitive,
{
    fn eq(lhs: T, rhs: T) -> bool {
        canonical::eq_float(lhs, rhs)
    }
}

pub trait ConstraintPartialOrd<T>
where
    T: Float + Primitive,
{
    fn partial_cmp(lhs: T, rhs: T) -> Option<Ordering> {
        lhs.partial_cmp(&rhs)
    }
}

impl<T, U> ConstraintPartialOrd<T> for U
where
    T: Float + Primitive,
    U: ConstraintOrd<T>,
{
    fn partial_cmp(lhs: T, rhs: T) -> Option<Ordering> {
        Some(U::cmp(lhs, rhs))
    }
}

pub trait ConstraintOrd<T>
where
    T: Float + Primitive,
{
    fn cmp(lhs: T, rhs: T) -> Ordering {
        canonical::cmp_float(lhs, rhs)
    }
}

pub trait ConstraintInfinity<T>
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

pub trait ConstraintNan<T>
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

pub trait SupersetOf<P> {}

pub trait SubsetOf<P> {}

impl<P, Q> SubsetOf<Q> for P where Q: SupersetOf<P> {}

/// Constraint on floating-point values.
pub trait FloatConstraint<T>: Copy + PartialEq + PartialOrd + Sized
where
    T: Float + Primitive,
{
    /// Filters a floating-point value based on some constraints.
    ///
    /// Returns `Some` for values that satisfy constraints and `None` for
    /// values that do not.
    fn filter(value: T) -> Option<T>;
}

impl<T> FloatConstraint<T> for ()
where
    T: Float + Primitive,
{
    fn filter(value: T) -> Option<T> {
        Some(value)
    }
}

impl<T> ConstraintEq<T> for () where T: Float + Primitive {}

impl<T> ConstraintOrd<T> for () where T: Float + Primitive {}

impl<T> ConstraintInfinity<T> for () where T: Float + Primitive {}

impl<T> ConstraintNan<T> for () where T: Float + Primitive {}

impl<T> SupersetOf<NotNanConstraint<T>> for () where T: Float + Primitive {}

impl<T> SupersetOf<FiniteConstraint<T>> for () where T: Float + Primitive {}

/// Disallows `NaN` floating-point values.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
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
    fn filter(value: T) -> Option<T> {
        if value.is_nan() {
            None
        }
        else {
            Some(value)
        }
    }
}

impl<T> ConstraintEq<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
    fn eq(lhs: T, rhs: T) -> bool {
        // The input values should never be `NaN`, so just compare the raw
        // floating-point values.
        lhs == rhs
    }
}

impl<T> ConstraintOrd<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
    fn cmp(lhs: T, rhs: T) -> Ordering {
        // The input values should never be `NaN`, so just compare the raw
        // floating-point values.
        lhs.partial_cmp(&rhs).unwrap()
    }
}

impl<T> ConstraintInfinity<T> for NotNanConstraint<T> where T: Float + Primitive {}

impl<T> SupersetOf<FiniteConstraint<T>> for NotNanConstraint<T> where T: Float + Primitive {}

/// Disallows `NaN`, `INF`, and `-INF` floating-point values.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
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
    fn filter(value: T) -> Option<T> {
        if value.is_nan() | value.is_infinite() {
            None
        }
        else {
            Some(value)
        }
    }
}

impl<T> ConstraintEq<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    fn eq(lhs: T, rhs: T) -> bool {
        // The input values should never be `NaN`, so just compare the raw
        // floating-point values.
        lhs == rhs
    }
}

impl<T> ConstraintOrd<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    fn cmp(lhs: T, rhs: T) -> Ordering {
        // The input values should never be `NaN`, so just compare the raw
        // floating-point values.
        lhs.partial_cmp(&rhs).unwrap()
    }
}
