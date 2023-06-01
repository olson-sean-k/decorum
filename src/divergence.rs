//! Output types for fallible operations and error behavior.
//!
//! This module provides types that determine the output types of fallible [`Proxy`] operations as
//! well as the behavior when an error occurs. These types are used as parameters of some
//! [constraints][`constraint`].
//!
//! # Output Types
//!
//! The output type of a fallible operations is the _branch_ type. The branch type is determined by
//! a divergence type parameter called the _branch kind_. These marker types are described below:
//!
//! | Branch Kind          | Branch Type        | Continue     | Break          |
//! |----------------------|--------------------|--------------|----------------|
//! | [`OutputBranch`]     | `T`                | `T`          |                |
//! | [`OptionBranch`]     | `Option<T>`        | `Some(T)`    | `None`         |
//! | [`ResultBranch`]     | `Result<T, E>`     | `Ok(T)`      | `Err(E)`       |
//! | [`ExpressionBranch`] | `Expression<T, E>` | `Defined(T)` | `Undefined(E)` |
//!
//! In the above table, `T` refers to a [`Proxy`] type and `E` refers to the associated error type
//! of its [constraint][`constraint`]. [`OutputBranch`] is unique in that it does not support
//! breaking and cannot produce a representation of errors. The remaining branch kinds produce
//! branch types that can represent both success and failure.
//!
//! # Error Behaviors
//!
//! Error behavior is determined by a _divergence_ marker type:
//!
//! | Divergence | OK       | Error     | Default Branch Kind  |
//! |------------|----------|-----------|----------------------|
//! | [`Assert`] | continue | **panic** | [`OutputBranch`]     |
//! | [`Try`]    | continue | break     | [`ExpressionBranch`] |
//!
//! The output of a divergence is determined by its branch kind. Note that **[`Assert`] panics if
//! an error occurs** and is the only divergence that supports [`OutputBranch`].
//!
//! # Examples
//!
//! The following example demonstrates a conditionally compiled `Real` type definition with a
//! [`Result`] branch type that, when an error occurs, returns `Err` in non-debug builds and panics
//! in debug builds.
//!
//! ```rust
//! use decorum::constraint::FiniteConstraint;
//! use decorum::divergence::ResultBranch;
//! use decorum::proxy::{BranchOf, Proxy};
//! use decorum::real::UnaryReal as _;
//!
//! #[cfg(debug_assertions)]
//! type Divergence = decorum::divergence::Assert<ResultBranch>;
//! #[cfg(not(debug_assertions))]
//! type Divergence = decorum::divergence::Try<ResultBranch>;
//!
//! pub type Real = Proxy<f64, FiniteConstraint<Divergence>>;
//! pub type RealResult = BranchOf<Real>;
//!
//! pub fn f(x: Real) -> RealResult {
//!     // This panics in debug builds and returns `Err` in non-debug builds.
//!     x / Real::ZERO
//! }
//! ```
//!
//! [`Assert`]: crate::divergence::Assert
//! [`constraint`]: crate::contraint
//! [`ExpressionBranch`]: crate::divergence::ExpressionBranch
//! [`OptionBranch`]: crate::divergence::OptionBranch
//! [`OutputBranch`]: crate::divergence::OutputBranch
//! [`Proxy`]: crate::proxy::Proxy
//! [`Result`]: core::result::Result
//! [`ResultBranch`]: crate::divergence::ResultBranch
//! [`Try`]: crate::divergence::Try

use core::convert::Infallible;
use core::fmt::Debug;
use core::marker::PhantomData;

use crate::constraint::ExpectConstrained as _;
use crate::expression::{Defined, Expression, Undefined};
use crate::proxy::{ClosedProxy, ErrorOf};
use crate::sealed::Sealed;

pub trait Continue: Sealed {
    type Branch<T, E>;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E>;
}

pub trait Break: Continue {
    fn break_with_error<T, E>(error: E) -> Self::Branch<T, E>;
}

pub enum ExpressionBranch {}

impl Break for ExpressionBranch {
    fn break_with_error<T, E>(error: E) -> Self::Branch<T, E> {
        Undefined(error)
    }
}

impl Continue for ExpressionBranch {
    type Branch<T, E> = Expression<T, E>;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        Defined(output)
    }
}

impl Sealed for ExpressionBranch {}

pub enum OptionBranch {}

impl Break for OptionBranch {
    fn break_with_error<T, E>(_: E) -> Self::Branch<T, E> {
        None
    }
}

impl Continue for OptionBranch {
    type Branch<T, E> = Option<T>;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        Some(output)
    }
}

impl Sealed for OptionBranch {}

pub enum OutputBranch {}

impl Continue for OutputBranch {
    type Branch<T, E> = T;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        output
    }
}

impl Sealed for OutputBranch {}

pub enum ResultBranch {}

impl Break for ResultBranch {
    fn break_with_error<T, E>(error: E) -> Self::Branch<T, E> {
        Err(error)
    }
}

impl Continue for ResultBranch {
    type Branch<T, E> = Result<T, E>;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        Ok(output)
    }
}

impl Sealed for ResultBranch {}

/// Determines the output type and behavior of a [`Proxy`] when it is fallibly constructed.
///
/// This trait defines a branch type and behavior when that type is constructed from an error. In
/// the error case, this may or may not yield a value and may instead diverge by panicking. When
/// constructed from an output, the branch type is always constructed and returned.
///
/// [`Proxy`]: crate::proxy::Proxy
pub trait Diverge: Sealed {
    type Branch<T, E>;

    fn diverge<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug;
}

impl Diverge for Infallible {
    type Branch<T, E> = T;

    fn diverge<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        match result {
            Ok(output) => output,
            _ => unreachable!(),
        }
    }
}

/// A [`Divergence`] with a branch type that is the same as its output type.
///
/// [`Divergence`]: crate::divergence::Divergence
pub trait NonResidual<P>: Diverge<Branch<P, ErrorOf<P>> = P>
where
    P: ClosedProxy,
{
}

impl<P, D> NonResidual<P> for D
where
    P: ClosedProxy,
    D: Diverge<Branch<P, ErrorOf<P>> = P>,
{
}

/// Divergence that breaks on errors by **panicking**.
///
/// **`Assert` panics if an error occurs.** This behavior is independent of the branch type, so
/// even an `Assert` divergence configured to construct [`Result`]s panics in the face of errors.
///
/// The branch type is determined by a branch kind type parameter. By default, `Assert` uses the
/// [`OutputBranch`] kind and therefore constructs [`Proxy`]s (no representation for errors) as the
/// output of fallible operations.
///
/// [`OutputBranch`]: crate::divergence::OutputBranch
/// [`Proxy`]: crate::proxy::Proxy
/// [`Result`]: core::result::Result
pub struct Assert<C = OutputBranch>(PhantomData<fn() -> C>, Infallible);

impl<C> Diverge for Assert<C>
where
    C: Continue,
{
    type Branch<T, E> = C::Branch<T, E>;

    fn diverge<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        C::continue_with_output(result.expect_constrained())
    }
}

impl<C> Sealed for Assert<C> {}

/// Divergence that breaks on errors by constructing a branch type that represents the error.
///
/// The branch type is determined by a branch kind type parameter. By default, `Try` uses the
/// [`ExpressionBranch`] kind and therefore constructs [`Expression`]s as the output of fallible
/// operations.
///
/// [`Expression`]: crate::expression::Expression
/// [`ExpressionBranch`]: crate::divergence::ExpressionBranch
pub struct Try<C = ExpressionBranch>(PhantomData<fn() -> C>, Infallible);

impl<C> Diverge for Try<C>
where
    C: Break,
{
    type Branch<T, E> = C::Branch<T, E>;

    fn diverge<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        match result {
            Ok(output) => C::continue_with_output(output),
            Err(error) => C::break_with_error(error),
        }
    }
}

impl<C> Sealed for Try<C> {}
