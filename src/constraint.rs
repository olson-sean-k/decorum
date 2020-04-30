//! Constraints on floating-point values.
//!
//! The `FloatConstraint` trait describes a constraint by filtering illegal
//! values and optionally supporting variants of `Ord`, `Eq`, etc. that are
//! also provided in this module. These analogous traits determine if and how
//! constrained values support these operations and in turn whether or not a
//! proxy using a constraint does too.

use core::marker::PhantomData;

use crate::primitive::Primitive;
use crate::{Infinite, Nan};

// TODO: `RealClass` should likely apply to virtually all constraint type
//       bounds (see `ConstrainedFloat`). Is it necessary at all?
pub enum RealClass {}
pub enum InfiniteClass {}
pub enum NanClass {}

// TODO: Can this be used to provide blanket implementations for `SupersetOf`?
pub trait Class<T> {}

pub trait SupersetOf<P> {}

pub trait SubsetOf<P> {}

impl<P, Q> SubsetOf<Q> for P where Q: SupersetOf<P> {}

/// Constraint on floating-point values.
pub trait Constraint<T>: Copy + Sized
where
    T: Primitive,
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
    T: Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Class<RealClass> for UnitConstraint<T> where T: Primitive {}

impl<T> Class<InfiniteClass> for UnitConstraint<T> where T: Primitive {}

impl<T> Class<NanClass> for UnitConstraint<T> where T: Primitive {}

// TODO: Should implementations map values like zero and `NaN` to canonical
//       forms?
impl<T> Constraint<T> for UnitConstraint<T>
where
    T: Primitive,
{
    fn filter(value: T) -> Option<T> {
        Some(value)
    }
}

impl<T> SupersetOf<NotNanConstraint<T>> for UnitConstraint<T> where T: Primitive {}

impl<T> SupersetOf<FiniteConstraint<T>> for UnitConstraint<T> where T: Primitive {}

/// Disallows `NaN` floating-point values.
#[derive(Clone, Copy, Debug)]
pub struct NotNanConstraint<T>
where
    T: Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Class<RealClass> for NotNanConstraint<T> where T: Primitive {}

impl<T> Class<InfiniteClass> for NotNanConstraint<T> where T: Primitive {}

impl<T> Constraint<T> for NotNanConstraint<T>
where
    T: Nan + Primitive,
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

impl<T> SupersetOf<FiniteConstraint<T>> for NotNanConstraint<T> where T: Primitive {}

/// Disallows `NaN`, `INF`, and `-INF` floating-point values.
#[derive(Clone, Copy, Debug)]
pub struct FiniteConstraint<T>
where
    T: Primitive,
{
    phantom: PhantomData<T>,
}

impl<T> Class<RealClass> for FiniteConstraint<T> where T: Primitive {}

impl<T> Constraint<T> for FiniteConstraint<T>
where
    T: Infinite + Nan + Primitive,
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
