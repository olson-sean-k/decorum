//! Output types for fallible operations and error behavior.
//!
//! This module provides type constructors that determine the output types of fallible [`Proxy`]
//! operations as well as the behavior when an error occurs. These types are used as parameters of
//! some [constraints][`constraint`].
//!
//! # Error Behaviors
//!
//! Error behavior is determined by a [_divergence_ marker type][`Diverge`]:
//!
//! | Divergence | OK       | Error     | Default Branch     |
//! |------------|----------|-----------|--------------------|
//! | [`Assert`] | continue | **panic** | [`ThenSelf`]       |
//! | [`Try`]    | continue | break     | [`ThenExpression`] |
//!
//! The output of a divergence is determined by its [_branch_ marker type][`Continue`]. Note that
//! **[`Assert`] panics if an error occurs** and is the only divergence that supports [`ThenSelf`].
//! See more about branch marker types below.
//!
//! # Output Types
//!
//! The output type of fallible [`Proxy`] operations is the [_branch_ type][`Continue::Branch`] and
//! is determined by a marker type. These marker types are described below:
//!
//! | Branch             | Type               | Continue     | Break          |
//! |--------------------|--------------------|--------------|----------------|
//! | [`ThenSelf`]       | `T`                | `T`          |                |
//! | [`ThenOption`]     | `Option<T>`        | `Some(T)`    | `None`         |
//! | [`ThenResult`]     | `Result<T, E>`     | `Ok(T)`      | `Err(E)`       |
//! | [`ThenExpression`] | `Expression<T, E>` | `Defined(T)` | `Undefined(E)` |
//!
//! In the above table, `T` refers to a [`Proxy`] type and `E` refers to the associated error type
//! of its [constraint][`constraint`]. [`ThenSelf`] is unique in that it cannot represent errors
//! and so does not support breaks: its output type is the identity. The remaining marker types
//! produce branch types that can encode both success and failure.
//!
//! # Examples
//!
//! The following example illustrates how to define a [`Proxy`] type.
//!
//! ```rust
//! use decorum::constraint::NotNanConstraint;
//! use decorum::divergence::{Assert, ThenSelf};
//! use decorum::proxy::Proxy;
//!
//! // A 32-bit floating-point representation that must be a real number or an infinity. Panics if
//! // constructed from a `NaN`.
//! pub type ExtendedReal = Proxy<f32, NotNanConstraint<Assert<ThenSelf>>>;
//! ```
//!
//! The following example demonstrates a conditionally compiled `Real` type definition with a
//! [`Result`] branch type that, when an error occurs, returns `Err` in **non**-debug builds and
//! panics in debug builds.
//!
//! ```rust
//! use decorum::constraint::FiniteConstraint;
//! use decorum::divergence::ThenResult;
//! use decorum::proxy::{BranchOf, Proxy};
//! use decorum::real::UnaryReal as _;
//!
//! #[cfg(debug_assertions)]
//! type Divergence = decorum::divergence::Assert<ThenResult>;
//! #[cfg(not(debug_assertions))]
//! type Divergence = decorum::divergence::Try<ThenResult>;
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
//! [`constraint`]: crate::constraint
//! [`Proxy`]: crate::proxy::Proxy
//! [`Result`]: core::result::Result
//! [`ThenExpression`]: crate::divergence::ThenExpression
//! [`ThenOption`]: crate::divergence::ThenOption
//! [`ThenResult`]: crate::divergence::ThenResult
//! [`ThenSelf`]: crate::divergence::ThenSelf
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

#[derive(Debug)]
pub enum ThenExpression {}

impl Break for ThenExpression {
    fn break_with_error<T, E>(error: E) -> Self::Branch<T, E> {
        Undefined(error)
    }
}

impl Continue for ThenExpression {
    type Branch<T, E> = Expression<T, E>;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        Defined(output)
    }
}

impl Sealed for ThenExpression {}

#[derive(Debug)]
pub enum ThenOption {}

impl Break for ThenOption {
    fn break_with_error<T, E>(_: E) -> Self::Branch<T, E> {
        None
    }
}

impl Continue for ThenOption {
    type Branch<T, E> = Option<T>;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        Some(output)
    }
}

impl Sealed for ThenOption {}

#[derive(Debug)]
pub enum ThenResult {}

impl Break for ThenResult {
    fn break_with_error<T, E>(error: E) -> Self::Branch<T, E> {
        Err(error)
    }
}

impl Continue for ThenResult {
    type Branch<T, E> = Result<T, E>;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        Ok(output)
    }
}

impl Sealed for ThenResult {}

#[derive(Debug)]
pub enum ThenSelf {}

impl Continue for ThenSelf {
    type Branch<T, E> = T;

    fn continue_with_output<T, E>(output: T) -> Self::Branch<T, E> {
        output
    }
}

impl Sealed for ThenSelf {}

/// Determines the output type and behavior of a [`Proxy`] when it is fallibly constructed.
///
/// This trait defines a branch type and behavior when that type is constructed from an error. In
/// the error case, this may or may not yield a value and may instead diverge by panicking. When
/// constructed from an output, the branch type is always constructed and returned.
///
/// [`Proxy`]: crate::proxy::Proxy
pub trait Diverge<E>: Sealed {
    type Branch<T, B>;

    fn diverge<T>(result: Result<T, E>) -> Self::Branch<T, E>;
}

impl Diverge<Infallible> for Infallible {
    type Branch<T, B> = T;

    fn diverge<T>(result: Result<T, Infallible>) -> Self::Branch<T, Infallible> {
        match result {
            Ok(output) => output,
            _ => unreachable!(),
        }
    }
}

/// A [`Diverge`] type with an identity branch type.
///
/// [`Diverge`]: crate::divergence::Diverge
pub trait NonResidual<P, E>: Diverge<E, Branch<P, ErrorOf<P>> = P>
where
    P: ClosedProxy,
{
}

impl<P, E, D> NonResidual<P, E> for D
where
    P: ClosedProxy,
    D: Diverge<E, Branch<P, ErrorOf<P>> = P>,
{
}

/// Divergence that breaks on errors by **panicking**.
///
/// **`Assert` panics if an error occurs.** This behavior is independent of the branch type, so
/// even an `Assert` divergence configured to construct [`Result`]s panics in the face of errors.
///
/// The branch type is determined by a branch kind type parameter. By default, `Assert` uses the
/// [`ThenSelf`] kind and therefore constructs [`Proxy`]s (no representation for errors) as the
/// output of fallible operations.
///
/// [`ThenSelf`]: crate::divergence::ThenSelf
/// [`Proxy`]: crate::proxy::Proxy
/// [`Result`]: core::result::Result
pub struct Assert<C = ThenSelf>(PhantomData<fn() -> C>, Infallible);

impl<C, E> Diverge<E> for Assert<C>
where
    C: Continue,
    E: Debug,
{
    type Branch<T, B> = C::Branch<T, B>;

    fn diverge<T>(result: Result<T, E>) -> Self::Branch<T, E> {
        C::continue_with_output(result.expect_constrained())
    }
}

impl<C> Sealed for Assert<C> {}

/// Divergence that breaks on errors by constructing a branch type that represents the error.
///
/// The branch type is determined by a branch kind type parameter. By default, `Try` uses the
/// [`ThenExpression`] kind and therefore constructs [`Expression`]s as the output of fallible
/// operations.
///
/// [`Expression`]: crate::expression::Expression
/// [`ThenExpression`]: crate::divergence::ThenExpression
pub struct Try<C = ThenExpression>(PhantomData<fn() -> C>, Infallible);

impl<C, E> Diverge<E> for Try<C>
where
    C: Break,
{
    type Branch<T, B> = C::Branch<T, B>;

    fn diverge<T>(result: Result<T, E>) -> Self::Branch<T, E> {
        match result {
            Ok(output) => C::continue_with_output(output),
            Err(error) => C::break_with_error(error),
        }
    }
}

impl<C> Sealed for Try<C> {}
