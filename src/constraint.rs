//! Constraints on the members of floating-point values that proxy types may
//! represent.

use core::marker::PhantomData;

use crate::primitive::Primitive;
use crate::Float;

pub enum RealClass {}
pub enum InfiniteClass {}
pub enum NanClass {}

pub trait Member<T> {}

pub trait SupersetOf<P> {}

pub trait SubsetOf<P> {}

impl<P, Q> SubsetOf<Q> for P where Q: SupersetOf<P> {}

/// Describes constraints on the set of floating-point values that a proxy type
/// may take.
///
/// This trait expresses a constraint by filtering values. Note that constraints
/// require `Member<RealClass>`, meaning that the class of real numbers must
/// always be supported and is implied.
pub trait Constraint<T>: Copy + Member<RealClass> + Sized
where
    T: Float + Primitive,
{
    /// Filters a floating-point value based on some constraints.
    ///
    /// Returns `Some` for values that satisfy constraints and `None` for
    /// values that do not.
    fn filter(value: T) -> Option<T>;
}

#[derive(Clone, Copy, Debug)]
pub struct UnitConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Member<RealClass> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> Member<InfiniteClass> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> Member<NanClass> for UnitConstraint<T> where T: Float + Primitive {}

// TODO: Should implementations map values like zero and `NaN` to canonical
//       forms?
impl<T> Constraint<T> for UnitConstraint<T>
where
    T: Float + Primitive,
{
    fn filter(value: T) -> Option<T> {
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

impl<T> Member<RealClass> for NotNanConstraint<T> where T: Float + Primitive {}

impl<T> Member<InfiniteClass> for NotNanConstraint<T> where T: Float + Primitive {}

impl<T> Constraint<T> for NotNanConstraint<T>
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

impl<T> SupersetOf<FiniteConstraint<T>> for NotNanConstraint<T> where T: Float + Primitive {}

/// Disallows `NaN`s and infinities.
#[derive(Clone, Copy, Debug)]
pub struct FiniteConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Member<RealClass> for FiniteConstraint<T> where T: Float + Primitive {}

impl<T> Constraint<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    fn filter(value: T) -> Option<T> {
        if value.is_nan() || value.is_infinite() {
            None
        }
        else {
            Some(value)
        }
    }
}
