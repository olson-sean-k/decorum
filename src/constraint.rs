//! Constraints on the set of IEEE 754 floating-point values that [`Proxy`] types may represent.
//!
//! This module provides traits and marker types that define the error conditions of [`Proxy`]s.
//! Constraints determine when, if ever, a particular floating-point value is considered an error
//! and must [diverge][`divergence`]. Constraints are defined in terms of subsets of IEEE 754
//! floating-point values and each constraint has associated [`Proxy`] type definitions:
//!
//! | Constraint           | Divergent | Type Definition | Disallowed Values     |
//! |----------------------|-----------|-----------------|-----------------------|
//! | [`UnitConstraint`]   | no        | [`Total`]       |                       |
//! | [`NotNanConstraint`] | yes       | [`NotNan`]      | `NaN`                 |
//! | [`FiniteConstraint`] | yes       | [`Finite`]      | `NaN`, `+INF`, `-INF` |
//!
//! [`UnitConstraint`] and [`Total`] apply no constraints on floating-point values. Unlike
//! primitive floating-point types however, [`Total`] defines equivalence and total ordering to
//! `NaN`, which allows it to implement numerous standard traits like `Eq`, `Hash`, and `Ord`.
//!
//! [`NotNan`], [`Finite`], and their corresponding constraints disallow certain IEEE 754 values.
//! Because the output of some floating-point operations may yield these values (even when the
//! inputs are real numbers), these constraints must specify a [`Divergence`], which determines the
//! behavior of [`Proxy`]s when such a value is encountered. These proxy type definitions specify
//! the [`Assert`] divergence by default, **which panics when a disallowed value is encountered.**
//!
//! [`Assert`]: crate::divergence::Assert
//! [`cmp`]: crate::cmp
//! [`divergence`]: crate::divergence
//! [`Divergence`]: crate::divergence::Divergence
//! [`Finite`]: crate::Finite
//! [`FiniteConstraint`]: crate::constraint::FiniteConstraint
//! [`NotNan`]: crate::NotNan
//! [`NotNanConstraint`]: crate::constraint::NotNanConstraint
//! [`Proxy`]: crate::proxy::Proxy
//! [`Total`]: crate::Total
//! [`UnitConstraint`]: crate::constraint::UnitConstraint

use core::convert::Infallible;
use core::fmt::Debug;
#[cfg(not(feature = "std"))]
use core::fmt::{self, Display, Formatter};
use core::marker::PhantomData;
#[cfg(feature = "std")]
use thiserror::Error;

use crate::cmp::UndefinedError;
use crate::divergence::Divergence;
use crate::sealed::Sealed;
use crate::{Float, Primitive};

const VIOLATION_MESSAGE: &str = "floating-point constraint violated";

#[cfg_attr(feature = "std", derive(Error))]
#[cfg_attr(feature = "std", error("{}", VIOLATION_MESSAGE))]
#[derive(Clone, Copy, Debug)]
pub struct ConstraintViolation;

// When the `std` feature is enabled, the `thiserror` crate is used to implement `Display`.
#[cfg(not(feature = "std"))]
impl Display for ConstraintViolation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", VIOLATION_MESSAGE)
    }
}

impl UndefinedError for ConstraintViolation {
    fn undefined() -> Self {
        ConstraintViolation
    }
}

pub(crate) trait ExpectConstrained<T>: Sized {
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
        // When the `std` feature is enabled, `ConstraintViolation` implements `Error` and an
        // appropriate error message is displayed when unwrapping.
        self.unwrap()
    }
}

pub enum RealSet {}

pub enum InfinitySet {}

pub enum NanSet {}

pub trait Member<T>: Sealed {}

pub trait SupersetOf<C>: Sealed {}

pub trait SubsetOf<C>: Sealed {}

impl<C1, C2> SubsetOf<C2> for C1
where
    C1: Sealed,
    C2: SupersetOf<C1>,
{
}

/// Describes constraints on the set of floating-point values that a [`Proxy`] may represent.
///
/// Note that constraints require [`Member<RealSet>`][`Member`], meaning that the set of real
/// numbers must always be supported and is implied where ever a `Constraint` bound is used.
///
/// [`Member`]: crate::constraint::Member
/// [`Proxy`]: crate::proxy::Proxy
pub trait Constraint: Member<RealSet> {
    type Divergence: Divergence;
    type Error: Debug;

    /// Determines if a primitive IEEE 754 floating-point value satisfies the constraint.
    ///
    /// # Errors
    ///
    /// Returns a `Self::Error` if the floating-point value violates the constraint, otherwise
    /// `None`.
    fn noncompliance<T>(inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive;

    fn compliance<T>(inner: T) -> Result<T, Self::Error>
    where
        T: Float + Primitive,
    {
        Self::noncompliance(inner).map_or(Ok(inner), |error| Err(error))
    }

    fn branch<T, U, F>(inner: T, f: F) -> <Self::Divergence as Divergence>::Branch<U, Self::Error>
    where
        T: Float + Primitive,
        F: FnOnce(T) -> U,
    {
        match Self::noncompliance(inner) {
            Some(error) => Self::Divergence::diverge(error),
            _ => Self::Divergence::from_output(f(inner)),
        }
    }
}

/// Constraint that disallows **no** IEEE 754 floating-point values.
#[derive(Debug)]
pub enum UnitConstraint {}

impl Constraint for UnitConstraint {
    type Divergence = Infallible;
    type Error = Infallible;

    fn noncompliance<T>(_inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive,
    {
        None
    }
}

impl Member<InfinitySet> for UnitConstraint {}

impl Member<NanSet> for UnitConstraint {}

impl Member<RealSet> for UnitConstraint {}

impl Sealed for UnitConstraint {}

impl<D> SupersetOf<FiniteConstraint<D>> for UnitConstraint {}

impl<D> SupersetOf<NotNanConstraint<D>> for UnitConstraint {}

/// Constraint that disallows IEEE 754 `NaN` values.
#[derive(Debug)]
pub struct NotNanConstraint<D> {
    phantom: PhantomData<fn() -> D>,
}

impl<D> Constraint for NotNanConstraint<D>
where
    D: Divergence,
{
    type Divergence = D;
    type Error = ConstraintViolation;

    fn noncompliance<T>(inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive,
    {
        inner.is_nan().then_some(ConstraintViolation)
    }
}

impl<D> Member<InfinitySet> for NotNanConstraint<D> {}

impl<D> Member<RealSet> for NotNanConstraint<D> {}

impl<D> Sealed for NotNanConstraint<D> {}

impl<D> SupersetOf<FiniteConstraint<D>> for NotNanConstraint<D> {}

/// Constraint that disallows IEEE 754 `NaN`, `+INF`, and `-INF` values.
#[derive(Debug)]
pub struct FiniteConstraint<D> {
    phantom: PhantomData<fn() -> D>,
}

impl<D> Constraint for FiniteConstraint<D>
where
    D: Divergence,
{
    type Divergence = D;
    type Error = ConstraintViolation;

    fn noncompliance<T>(inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive,
    {
        (inner.is_nan() || inner.is_infinite()).then_some(ConstraintViolation)
    }
}

impl<D> Member<RealSet> for FiniteConstraint<D> {}

impl<D> Sealed for FiniteConstraint<D> {}
