//! Constraints on the set of IEEE 754 floating-point values that [`Constrained`] types may
//! represent.
//!
//! This module provides traits and types that define the error conditions of [`Constrained`]s.
//! Constraints determine when, if ever, a particular floating-point value is considered an error
//! and so construction must [diverge][`divergence`]. Constraints are defined in terms of subsets
//! of IEEE 754 floating-point values and each constraint has associated [`Constrained`] type
//! definitions for convenience:
//!
//! | Constraint         | Divergent | Type Definition  | Disallowed Values     |
//! |--------------------|-----------|------------------|-----------------------|
//! | [`IsFloat`]        | no        | [`Total`]        |                       |
//! | [`IsExtendedReal`] | yes       | [`ExtendedReal`] | `NaN`                 |
//! | [`IsReal`]         | yes       | [`Real`]         | `NaN`, `+INF`, `-INF` |
//!
//! [`IsFloat`] and [`Total`] apply no constraints on floating-point values. Unlike primitive
//! floating-point types however, [`Total`] defines equivalence and total ordering to `NaN`, which
//! allows it to implement related standard traits like `Eq`, `Hash`, and `Ord`.
//!
//! [`ExtendedReal`], [`Real`], and their corresponding constraints disallow certain IEEE 754
//! values. Because the output of some floating-point operations may yield these values (even when
//! the inputs are real numbers), these constraints must specify a [divergence][`divergence`],
//! which determines the behavior of [`Constrained`]s when such a value is encountered.
//!
//! [`cmp`]: crate::cmp
//! [`Constrained`]: crate::proxy::Constrained
//! [`divergence`]: crate::divergence
//! [`ExtendedReal`]: crate::ExtendedReal
//! [`IsExtendedReal`]: crate::constraint::IsExtendedReal
//! [`IsFloat`]: crate::constraint::IsFloat
//! [`IsReal`]: crate::constraint::IsReal
//! [`OrPanic`]: crate::divergence::OrPanic
//! [`Real`]: crate::Real
//! [`Total`]: crate::Total

use core::convert::Infallible;
use core::fmt::{self, Debug, Display, Formatter};
use core::marker::PhantomData;
#[cfg(feature = "std")]
use thiserror::Error;

use crate::cmp::EmptyInhabitant;
use crate::divergence::{Divergence, OrPanic, OutputOf};
use crate::proxy::{Constrained, ConstrainedProxy};
use crate::sealed::{Sealed, StaticDebug};
use crate::{NanEncoding, Primitive};

pub(crate) mod sealed {
    use crate::proxy::Constrained;
    use crate::Primitive;

    /// Defines a notion of empty inhabitants for [`Constraint`] types.
    ///
    /// This trait is a corollary to [`EmptyOrd`] and is used to implement that trait for
    /// [`Constrained`] types more generally than is otherwise possible.
    ///
    /// **The notion of an empty inhabitant is independent of constraints.** Regardless of
    /// constraint, `NaN`s are considered empty inhabitants.
    ///
    /// [`Constrained`]: crate::proxy::Constrained
    /// [`Constraint`]: crate::constraint::Constraint
    /// [`EmptyOrd`]: crate::cmp::EmptyOrd
    pub trait FromEmpty: Sized {
        type Empty<T>;

        fn empty<T>() -> Self::Empty<T>
        where
            T: Primitive;

        fn from_empty<T>(empty: Self::Empty<T>) -> Constrained<T, Self>
        where
            T: Primitive;

        fn is_empty<T>(primitive: &T) -> bool
        where
            T: Primitive;
    }
}
use sealed::FromEmpty;

pub(crate) trait Description {
    const DESCRIPTION: &'static str;
}

#[cfg_attr(feature = "std", derive(Error))]
#[derive(Clone, Copy, Debug)]
pub enum ConstraintError {
    #[cfg_attr(feature = "std", error(transparent))]
    NotExtendedReal(NotExtendedRealError),
    #[cfg_attr(feature = "std", error(transparent))]
    NotReal(NotRealError),
}

#[cfg(not(feature = "std"))]
impl Display for ConstraintError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ConstraintError::NotExtendedReal(_) => NotExtendedRealError::DESCRIPTION,
                ConstraintError::NotReal(_) => NotRealError::DESCRIPTION,
            },
        )
    }
}

impl From<NotExtendedRealError> for ConstraintError {
    fn from(error: NotExtendedRealError) -> Self {
        ConstraintError::NotExtendedReal(error)
    }
}

impl From<NotRealError> for ConstraintError {
    fn from(error: NotRealError) -> Self {
        ConstraintError::NotReal(error)
    }
}

#[cfg_attr(feature = "std", derive(Error))]
#[cfg_attr(feature = "std", error("{}", NotExtendedRealError::DESCRIPTION))]
#[derive(Clone, Copy, Debug)]
pub struct NotExtendedRealError;

impl Description for NotExtendedRealError {
    const DESCRIPTION: &'static str = "floating-point value must be an extended real";
}

#[cfg(not(feature = "std"))]
impl Display for NotExtendedRealError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", NotExtendedRealError::DESCRIPTION)
    }
}

impl EmptyInhabitant for NotExtendedRealError {
    fn empty() -> Self {
        NotExtendedRealError
    }
}

#[cfg_attr(feature = "std", derive(Error))]
#[cfg_attr(feature = "std", error("{}", NotRealError::DESCRIPTION))]
#[derive(Clone, Copy, Debug)]
pub struct NotRealError;

impl Description for NotRealError {
    const DESCRIPTION: &'static str = "floating-point value must be a real";
}

#[cfg(not(feature = "std"))]
impl Display for NotRealError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", NotRealError::DESCRIPTION)
    }
}

impl EmptyInhabitant for NotRealError {
    fn empty() -> Self {
        NotRealError
    }
}

pub(crate) trait ExpectConstrained<T>: Sized {
    fn expect_constrained(self) -> T;
}

impl<T, E> ExpectConstrained<T> for Result<T, E>
where
    E: Debug,
{
    fn expect_constrained(self) -> T {
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

/// Describes constraints on the set of floating-point values that a [`Constrained`] may represent.
///
/// Note that constraints require [`Member<RealSet>`][`Member`], meaning that the set of real
/// numbers must always be supported and is implied wherever a `Constraint` bound is used.
///
/// [`Constrained`]: crate::proxy::Constrained
/// [`Member`]: crate::constraint::Member
pub trait Constraint: FromEmpty + Member<RealSet> + StaticDebug {
    type Divergence: Divergence;
    // TODO: Bound this on `core::Error` once it is stabilized.
    type Error: Debug + Display;

    // It is not possible for constraints to map accepted values because of reference conversions,
    // so the successful output is the unit type and primitive values must be used as-is. That is,
    // this function only expresses the membership of the given value and no other.
    fn check<T>(inner: T) -> Result<(), Self::Error>
    where
        T: Primitive;

    fn map<T, U, F>(inner: T, f: F) -> OutputOf<Self::Divergence, U, Self::Error>
    where
        T: Primitive,
        U: ConstrainedProxy<Constraint = Self, Primitive = T>,
        F: FnOnce(T) -> U,
    {
        Self::Divergence::diverge(Self::check(inner).map(|_| f(inner)))
    }
}

#[derive(Debug)]
pub enum IsFloat {}

impl Constraint for IsFloat {
    // Branching in the `Divergence` is completely bypassed in this implementation.
    type Divergence = OrPanic;
    type Error = Infallible;

    #[inline(always)]
    fn check<T>(_inner: T) -> Result<(), Self::Error>
    where
        T: Primitive,
    {
        Ok(())
    }

    #[inline(always)]
    fn map<T, U, F>(inner: T, f: F) -> U
    where
        T: Primitive,
        U: ConstrainedProxy<Constraint = Self, Primitive = T>,
        F: FnOnce(T) -> U,
    {
        f(inner)
    }
}

impl FromEmpty for IsFloat {
    type Empty<T> = Constrained<T, Self>;

    #[inline(always)]
    fn empty<T>() -> Self::Empty<T>
    where
        T: Primitive,
    {
        Constrained::NAN
    }

    #[inline(always)]
    fn from_empty<T>(empty: Self::Empty<T>) -> Constrained<T, Self>
    where
        T: Primitive,
    {
        empty
    }

    #[inline(always)]
    fn is_empty<T>(primitive: &T) -> bool
    where
        T: Primitive,
    {
        primitive.is_nan()
    }
}

impl Member<InfinitySet> for IsFloat {}

impl Member<NanSet> for IsFloat {}

impl Member<RealSet> for IsFloat {}

impl Sealed for IsFloat {}

impl StaticDebug for IsFloat {
    fn fmt(formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "IsFloat")
    }
}

impl<D> SupersetOf<IsReal<D>> for IsFloat {}

impl<D> SupersetOf<IsExtendedReal<D>> for IsFloat {}

#[derive(Debug)]
pub struct IsExtendedReal<D>(PhantomData<fn() -> D>, Infallible);

pub type IsNotNan<D> = IsExtendedReal<D>;

impl<D> Constraint for IsExtendedReal<D>
where
    D: Divergence,
{
    type Divergence = D;
    type Error = NotExtendedRealError;

    fn check<T>(inner: T) -> Result<(), Self::Error>
    where
        T: Primitive,
    {
        if inner.is_nan() {
            Err(NotExtendedRealError)
        }
        else {
            Ok(())
        }
    }
}

impl<D> FromEmpty for IsExtendedReal<D>
where
    D: Divergence,
{
    type Empty<T> = Infallible;

    fn empty<T>() -> Self::Empty<T>
    where
        T: Primitive,
    {
        unreachable!()
    }

    fn from_empty<T>(_: Self::Empty<T>) -> Constrained<T, Self>
    where
        T: Primitive,
    {
        unreachable!()
    }

    #[inline(always)]
    fn is_empty<T>(_: &T) -> bool
    where
        T: Primitive,
    {
        false
    }
}

impl<D> Member<InfinitySet> for IsExtendedReal<D> {}

impl<D> Member<RealSet> for IsExtendedReal<D> {}

impl<D> Sealed for IsExtendedReal<D> {}

impl<D> StaticDebug for IsExtendedReal<D>
where
    D: StaticDebug,
{
    fn fmt(formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "IsExtendedReal<")?;
        D::fmt(formatter)?;
        write!(formatter, ">")
    }
}

impl<D> SupersetOf<IsReal<D>> for IsExtendedReal<D> {}

#[derive(Debug)]
pub struct IsReal<D>(PhantomData<fn() -> D>, Infallible);

impl<D> Constraint for IsReal<D>
where
    D: Divergence,
{
    type Divergence = D;
    type Error = NotRealError;

    fn check<T>(inner: T) -> Result<(), Self::Error>
    where
        T: Primitive,
    {
        if inner.is_nan() || inner.is_infinite() {
            Err(NotRealError)
        }
        else {
            Ok(())
        }
    }
}

impl<D> FromEmpty for IsReal<D>
where
    D: Divergence,
{
    type Empty<T> = Infallible;

    fn empty<T>() -> Self::Empty<T>
    where
        T: Primitive,
    {
        unreachable!()
    }

    fn from_empty<T>(_: Self::Empty<T>) -> Constrained<T, Self>
    where
        T: Primitive,
    {
        unreachable!()
    }

    #[inline(always)]
    fn is_empty<T>(_: &T) -> bool
    where
        T: Primitive,
    {
        false
    }
}

impl<D> Member<RealSet> for IsReal<D> {}

impl<D> Sealed for IsReal<D> {}

impl<D> StaticDebug for IsReal<D>
where
    D: StaticDebug,
{
    fn fmt(formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "IsReal<")?;
        D::fmt(formatter)?;
        write!(formatter, ">")
    }
}
