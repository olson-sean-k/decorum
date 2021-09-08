//! Constraints on the members of floating-point values that proxy types may
//! represent.

use core::convert::Infallible;
use core::fmt::Debug;
#[cfg(not(feature = "std"))]
use core::fmt::{self, Display, Formatter};
use core::marker::PhantomData;
#[cfg(feature = "std")]
use thiserror::Error;

use crate::{Float, Primitive};

const VIOLATION_MESSAGE: &str = "floating-point constraint violated";

#[cfg_attr(feature = "std", derive(Error))]
#[cfg_attr(feature = "std", error("{}", VIOLATION_MESSAGE))]
#[derive(Clone, Copy, Debug)]
pub struct ConstraintViolation;

// When the `std` feature is enabled, the `thiserror` crate is used to implement
// `Display`.
#[cfg(not(feature = "std"))]
impl Display for ConstraintViolation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", VIOLATION_MESSAGE)
    }
}

pub trait ExpectConstrained<T>: Sized {
    fn expect_constrained(self) -> T;
}

impl<T, E> ExpectConstrained<T> for Result<T, E>
where
    E: Debug,
{
    #[cfg(not(feature = "std"))]
    fn expect_constrained(self) -> T {
        self.expect(VIOLATION_MESSAGE)
    }

    #[cfg(feature = "std")]
    fn expect_constrained(self) -> T {
        // When the `std` feature is enabled, `ConstraintViolation` implements
        // `Error` and an appropriate error message is displayed when
        // unwrapping.
        self.unwrap()
    }
}

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
/// This trait expresses a constraint by defining an error and emitting that
/// error from its `check` function if a primitive floating-point value violates
/// the constraint. Note that constraints require `Member<RealSet>`, meaning
/// that the set of real numbers must always be supported and is implied.
pub trait Constraint<T>: Member<RealSet>
where
    T: Float + Primitive,
{
    type Error: Debug;

    /// Determines if a primitive floating-point value satisfies the constraint.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` if the primitive floating-point value violates the
    /// constraint.
    fn check(inner: &T) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct UnitConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<fn() -> T>,
}

impl<T> Constraint<T> for UnitConstraint<T>
where
    T: Float + Primitive,
{
    type Error = Infallible;

    fn check(_: &T) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<T> Member<InfiniteSet> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> Member<NanSet> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> Member<RealSet> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> SupersetOf<FiniteConstraint<T>> for UnitConstraint<T> where T: Float + Primitive {}

impl<T> SupersetOf<NotNanConstraint<T>> for UnitConstraint<T> where T: Float + Primitive {}

/// Disallows `NaN`s.
#[derive(Debug)]
pub struct NotNanConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<fn() -> T>,
}

impl<T> Constraint<T> for NotNanConstraint<T>
where
    T: Float + Primitive,
{
    type Error = ConstraintViolation;

    fn check(inner: &T) -> Result<(), Self::Error> {
        if inner.is_nan() {
            Err(ConstraintViolation)
        }
        else {
            Ok(())
        }
    }
}

impl<T> Member<InfiniteSet> for NotNanConstraint<T> where T: Float + Primitive {}

impl<T> Member<RealSet> for NotNanConstraint<T> where T: Float + Primitive {}

impl<T> SupersetOf<FiniteConstraint<T>> for NotNanConstraint<T> where T: Float + Primitive {}

/// Disallows `NaN`s and infinities.
#[derive(Debug)]
pub struct FiniteConstraint<T>
where
    T: Float + Primitive,
{
    phantom: PhantomData<fn() -> T>,
}

impl<T> Constraint<T> for FiniteConstraint<T>
where
    T: Float + Primitive,
{
    type Error = ConstraintViolation;

    fn check(inner: &T) -> Result<(), Self::Error> {
        if inner.is_nan() || inner.is_infinite() {
            Err(ConstraintViolation)
        }
        else {
            Ok(())
        }
    }
}

impl<T> Member<RealSet> for FiniteConstraint<T> where T: Float + Primitive {}
