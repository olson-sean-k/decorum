use num_traits::Float;
use std::cmp::Ordering;
use std::marker::PhantomData;

use Primitive;
use canonical;

pub trait FloatEq<T>
where
    T: Float + Primitive,
{
    fn eq(lhs: T, rhs: T) -> bool {
        canonical::eq_float(lhs, rhs)
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
        canonical::cmp_float(lhs, rhs)
    }
}

pub trait FloatInfinity<T>
where
    T: Float + Primitive,
{
    #[inline(always)]
    fn infinity() -> T {
        T::infinity()
    }

    #[inline(always)]
    fn neg_infinity() -> T {
        T::neg_infinity()
    }

    #[inline(always)]
    fn is_infinite(value: T) -> bool {
        value.is_infinite()
    }

    #[inline(always)]
    fn is_finite(value: T) -> bool {
        value.is_finite()
    }
}

pub trait FloatNan<T>
where
    T: Float + Primitive,
{
    #[inline(always)]
    fn nan() -> T {
        T::nan()
    }

    #[inline(always)]
    fn is_nan(value: T) -> bool {
        value.is_nan()
    }
}

pub trait SupersetOf<P> {}

pub trait SubsetOf<P> {}

impl<P, Q> SubsetOf<Q> for P
where
    Q: SupersetOf<P>,
{
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

impl<T> SupersetOf<NotNanConstraint<T>> for ()
where
    T: Float + Primitive,
{
}

impl<T> SupersetOf<FiniteConstraint<T>> for ()
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

impl<T> SupersetOf<FiniteConstraint<T>> for NotNanConstraint<T>
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
