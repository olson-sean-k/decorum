//! Constraints on the members of floating-point values that proxy types may
//! represent.

use core::marker::PhantomData;

use crate::{Float, Primitive};

pub enum RealSet {}
pub enum InfiniteSet {}
pub enum NanSet {}

pub trait Member<T> {}

pub trait SupersetOf<P> {}

pub trait SubsetOf<P> {}

impl<P, Q> SubsetOf<Q> for P where Q: SupersetOf<P> {}

/// Describes constraints on the set of floating-point values that a proxy type
/// may represent.
///
/// This trait expresses a constraint by filter-mapping values. Note that
/// constraints require `Member<RealSet>`, meaning that the set of real numbers
/// must always be supported and is implied.
pub trait Constraint<T>: Copy + Member<RealSet> + Sized
where
    T: Float + Primitive,
{
    /// Filter-maps a primitive floating-point value based on some constraints.
    ///
    /// Returns `None` for values that cannot satify constraints.
    fn filter_map(value: T) -> Option<T>;
}

#[derive(Clone, Copy, Debug)]
pub struct UnitConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Member<RealSet> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> Member<InfiniteSet> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> Member<NanSet> for UnitConstraint<T> where T: Float + Primitive {}

// TODO: Should implementations map values like zero and `NaN` to canonical
//       forms?
impl<T> Constraint<T> for UnitConstraint<T>
where
    T: Float + Primitive,
{
    fn filter_map(value: T) -> Option<T> {
        Some(value)
    }
}

impl<T> SupersetOf<NotNanConstraint<T>> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> SupersetOf<FiniteConstraint<T>> for UnitConstraint<T> where T: Float + Primitive {}

/// Disallows `NaN`s.
#[derive(Clone, Copy, Debug)]
pub struct NotNanConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Member<RealSet> for NotNanConstraint<T> where T: Float + Primitive {}

impl<T> Member<InfiniteSet> for NotNanConstraint<T> where T: Float + Primitive {}

impl<T> Constraint<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
    fn filter_map(value: T) -> Option<T> {
        if value.is_nan() {
            None
        }
        else {
            Some(value)
        }
    }
}

impl<T> SupersetOf<FiniteConstraint<T>> for NotNanConstraint<T> where T: Float + Primitive {}

/// Disallows `NaN`s and infinities.
#[derive(Clone, Copy, Debug)]
pub struct FiniteConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Member<RealSet> for FiniteConstraint<T> where T: Float + Primitive {}

impl<T> Constraint<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    fn filter_map(value: T) -> Option<T> {
        if value.is_nan() || value.is_infinite() {
            None
        }
        else {
            Some(value)
        }
    }
}
