//! Output types for fallible operations and error behavior.
//!
//! This module provides type constructors that determine the output types of fallible [`Proxy`]
//! operations as well as the behavior when an error occurs. These types are used as parameters of
//! some [constraints][`constraint`].
//!
//! # Error Behaviors
//!
//! Error behavior is determined by a [_divergence_ type][`Divergence`]:
//!
//! | Divergence  | OK       | Error     | Default Branch   |
//! |-------------|----------|-----------|------------------|
//! | [`OrPanic`] | continue | **panic** | [`AsSelf`]       |
//! | [`OrError`] | continue | break     | [`AsExpression`] |
//!
//! The output type of fallible operations is determined by the [_branch_ type][`Continue`] of the
//! divergence. Note that **[`OrPanic`] panics if an error occurs** and is the only divergence that
//! supports the [`AsSelf`] branch, which returns `Self` in fallible oeprations.
//!
//! # Output Types
//!
//! The output type of fallible [`Proxy`] operations is an [associated type][`Continue::Branch`] of
//! a [_branch_ type][`Continue`]. These types are described below:
//!
//! | Branch           | Type                  | Continue        | Break          |
//! |------------------|-----------------------|-----------------|----------------|
//! | [`AsSelf`]       | `Self`                | `Self`          |                |
//! | [`AsOption`]     | `Option<Self>`        | `Some(Self)`    | `None`         |
//! | [`AsResult`]     | `Result<Self, E>`     | `Ok(Self)`      | `Err(E)`       |
//! | [`AsExpression`] | `Expression<Self, E>` | `Defined(Self)` | `Undefined(E)` |
//!
//! In the above table, `Self` refers to a [`Proxy`] type and `E` refers to the associated error
//! type of its [constraint][`constraint`]. [`AsSelf`] is unique in that it cannot represent errors
//! and so does not support breaks: its output type is the identity. The remaining branch types
//! produce output types that can encode both success and failure.
//!
//! # Examples
//!
//! The following example illustrates how to define a [`Proxy`] type.
//!
//! ```rust
//! use decorum::constraint::NotNanConstraint;
//! use decorum::divergence::{AsSelf, OrPanic};
//! use decorum::proxy::Proxy;
//!
//! // A 32-bit floating-point representation that must be a real number or an infinity. Panics if
//! // constructed from a `NaN`.
//! pub type ExtendedReal = Proxy<f32, NotNanConstraint<OrPanic<AsSelf>>>;
//! ```
//!
//! The following example demonstrates a conditionally compiled `Real` type definition with a
//! [`Result`] branch type that, when an error occurs, returns `Err` in **non**-debug builds and
//! panics in debug builds.
//!
//! ```rust
//! use decorum::constraint::FiniteConstraint;
//! use decorum::divergence::AsResult;
//! use decorum::proxy::{BranchOf, Proxy};
//! use decorum::real::UnaryReal as _;
//!
//! #[cfg(debug_assertions)]
//! type Divergence = decorum::divergence::OrPanic<AsResult>;
//! #[cfg(not(debug_assertions))]
//! type Divergence = decorum::divergence::OrError<AsResult>;
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
//! [`AsExpression`]: crate::divergence::AsExpression
//! [`AsOption`]: crate::divergence::AsOption
//! [`AsResult`]: crate::divergence::AsResult
//! [`AsSelf`]: crate::divergence::AsSelf
//! [`constraint`]: crate::constraint
//! [`Divergence`]: crate::divergence::Divergence
//! [`OrError`]: crate::divergence::OrError
//! [`OrPanic`]: crate::divergence::OrPanic
//! [`Proxy`]: crate::proxy::Proxy
//! [`Result`]: core::result::Result

// TODO: Rename and refactor types such that type definitions resemble the following:
//
//         pub type Real = Constrained<f64, IsReal<OrPanic>>;
//         pub type Real = Constrained<f64, IsReal<OrPanic<AsSelf>>>;
//         pub type Extended = Constrained<f64, IsExtended<OrError<AsExpression>>>;
//         pub type Total = Constrained<f64, IsFloat>;

use core::convert::Infallible;
use core::fmt::Debug;
use core::marker::PhantomData;

use crate::constraint::ExpectConstrained as _;
use crate::expression::{Defined, Expression, Undefined};
use crate::sealed::Sealed;

pub trait Continue: Sealed {
    type Branch<P, E>;

    fn continue_with_output<P, E>(output: P) -> Self::Branch<P, E>;
}

pub trait Break: Continue {
    fn break_with_error<P, E>(error: E) -> Self::Branch<P, E>;
}

/// A [branch][`Continue`] type that outputs the identity.
///
/// [`Continue`]: crate::divergence::Continue
pub trait NonResidual<P, E>: Continue<Branch<P, E> = P> {}

impl<P, E, B> NonResidual<P, E> for B where B: Continue<Branch<P, E> = P> {}

#[derive(Debug)]
pub enum AsExpression {}

impl Break for AsExpression {
    fn break_with_error<P, E>(error: E) -> Self::Branch<P, E> {
        Undefined(error)
    }
}

impl Continue for AsExpression {
    type Branch<P, E> = Expression<P, E>;

    fn continue_with_output<P, E>(output: P) -> Self::Branch<P, E> {
        Defined(output)
    }
}

impl Sealed for AsExpression {}

#[derive(Debug)]
pub enum AsOption {}

impl Break for AsOption {
    fn break_with_error<P, E>(_: E) -> Self::Branch<P, E> {
        None
    }
}

impl Continue for AsOption {
    type Branch<P, E> = Option<P>;

    fn continue_with_output<P, E>(output: P) -> Self::Branch<P, E> {
        Some(output)
    }
}

impl Sealed for AsOption {}

#[derive(Debug)]
pub enum AsResult {}

impl Break for AsResult {
    fn break_with_error<P, E>(error: E) -> Self::Branch<P, E> {
        Err(error)
    }
}

impl Continue for AsResult {
    type Branch<P, E> = Result<P, E>;

    fn continue_with_output<P, E>(output: P) -> Self::Branch<P, E> {
        Ok(output)
    }
}

impl Sealed for AsResult {}

#[derive(Debug)]
pub enum AsSelf {}

impl Continue for AsSelf {
    type Branch<P, E> = P;

    fn continue_with_output<P, E>(output: P) -> Self::Branch<P, E> {
        output
    }
}

impl Sealed for AsSelf {}

/// Determines the output type and behavior of a [`Proxy`] when it is fallibly constructed.
///
/// This trait defines a branch type and behavior when that type is constructed from an error. In
/// the error case, this may or may not yield a value and may instead diverge by panicking. When
/// constructed from an output, the branch type is always constructed and returned.
///
/// [`Proxy`]: crate::proxy::Proxy
pub trait Divergence<E>: Sealed {
    type Continue: Continue;

    fn diverge<T>(result: Result<T, E>) -> <Self::Continue as Continue>::Branch<T, E>;
}

pub type ContinueOf<D, E> = <D as Divergence<E>>::Continue;
pub type BranchOf<D, P, E> = <ContinueOf<D, E> as Continue>::Branch<P, E>;

impl Divergence<Infallible> for Infallible {
    type Continue = AsSelf;

    #[inline(always)]
    fn diverge<T>(result: Result<T, Infallible>) -> T {
        // SAFETY: The `Err` variant of `Result<_, Infallible>` is impossible to construct, so this
        //         cannot be `Err` here.
        unsafe { result.unwrap_unchecked() }
    }
}

/// Divergence that breaks on errors by **panicking**.
///
/// **`OrPanic` panics if an error occurs.** This behavior is independent of the branch type, so
/// even an `OrPanic` divergence configured to construct [`Result`]s panics in the face of errors.
///
/// The branch type is determined by a branch kind type parameter. By default, `OrPanic` uses the
/// [`AsSelf`] kind and therefore constructs [`Proxy`]s (no representation for errors) as the
/// output of fallible operations.
///
/// [`AsSelf`]: crate::divergence::AsSelf
/// [`Proxy`]: crate::proxy::Proxy
/// [`Result`]: core::result::Result
pub struct OrPanic<B = AsSelf>(PhantomData<fn() -> B>, Infallible);

impl<B, E> Divergence<E> for OrPanic<B>
where
    B: Continue,
    E: Debug,
{
    type Continue = B;

    fn diverge<T>(result: Result<T, E>) -> B::Branch<T, E> {
        B::continue_with_output(result.expect_constrained())
    }
}

impl<B> Sealed for OrPanic<B> {}

/// Divergence that breaks on errors by constructing a branch type that represents the error.
///
/// The branch type is determined by a branch kind type parameter. By default, `OrError` uses the
/// [`AsExpression`] kind and therefore constructs [`Expression`]s as the output of fallible
/// operations.
///
/// [`AsExpression`]: crate::divergence::AsExpression
/// [`Expression`]: crate::expression::Expression
pub struct OrError<B = AsExpression>(PhantomData<fn() -> B>, Infallible);

impl<B, E> Divergence<E> for OrError<B>
where
    B: Break,
{
    type Continue = B;

    fn diverge<T>(result: Result<T, E>) -> B::Branch<T, E> {
        match result {
            Ok(output) => B::continue_with_output(output),
            Err(error) => B::break_with_error(error),
        }
    }
}

impl<B> Sealed for OrError<B> {}
