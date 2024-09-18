//! Error behavior and output types for fallible operations.
//!
//! This module provides type constructors that determine the behavior and output types of fallible
//! [`Proxy`] operations. These types are used as parameters of some [constraints][`constraint`].
//!
//! # Error Behaviors
//!
//! Error behavior is determined by a [divergence type][`Divergence`]:
//!
//! | Divergence  | OK       | Error     | Default Output Kind |
//! |-------------|----------|-----------|---------------------|
//! | [`OrPanic`] | continue | **panic** | [`AsSelf`]          |
//! | [`OrError`] | continue | break     | [`AsExpression`]    |
//!
//! Divergence is independent of output types: the [`OrPanic`] divergence panics when breaking even
//! when the output type can represent errors (e.g., [`Result`]). Because [`OrPanic`] never returns
//! an error value, it can be used with output types that cannot respresent errors. This differs
//! from [`OrError`], which requires an error representation.
//!
//! # Output Types
//!
//! Output types are determined by an [output kind][`Continue`]. An output kind is type constructor
//! with which a [`Divergence`] can construct an output type:
//!
//! | Output Kind      | Output Type           | Continue        | Break          |
//! |------------------|-----------------------|-----------------|----------------|
//! | [`AsSelf`]       | `Self`                | `Self`          |                |
//! | [`AsOption`]     | `Option<Self>`        | `Some(Self)`    | `None`         |
//! | [`AsResult`]     | `Result<Self, E>`     | `Ok(Self)`      | `Err(E)`       |
//! | [`AsExpression`] | `Expression<Self, E>` | `Defined(Self)` | `Undefined(E)` |
//!
//! In the above table, `Self` refers to a [`Proxy`] type and `E` refers to the [associated
//! error][`Constraint::Error`] type of its [constraint][`constraint`]. [`AsSelf`] is unique in
//! that it cannot represent errors and so does not support breaking: its output type is the
//! identity.
//!
//! # Examples
//!
//! The following example illustrates how to define a [`Proxy`] type.
//!
//! ```rust
//! use decorum::constraint::IsNotNan;
//! use decorum::divergence::{AsSelf, OrPanic};
//! use decorum::proxy::Proxy;
//!
//! // A 32-bit floating-point representation that must be a real number or an infinity. Panics if
//! // constructed from a `NaN`.
//! pub type NotNan = Proxy<f32, IsNotNan<OrPanic<AsSelf>>>;
//! ```
//!
//! The following example demonstrates a conditionally compiled `Real` type definition with a
//! [`Result`] branch type that, when an error occurs, returns `Err` in **non**-debug builds but
//! panics in debug builds.
//!
//! ```rust
//! pub mod real {
//!     use decorum::constraint::IsReal;
//!     use decorum::divergence::{self, AsResult};
//!     use decorum::proxy::{OutputOf, Proxy};
//!
//!     #[cfg(debug_assertions)]
//!     type OrDiverge = divergence::OrPanic<AsResult>;
//!     #[cfg(not(debug_assertions))]
//!     type OrDiverge = divergence::OrError<AsResult>;
//!
//!     pub type Real = Proxy<f64, IsReal<OrDiverge>>;
//!     pub type Result = OutputOf<Real>;
//! }
//!
//! use decorum::real::UnaryReal as _;
//!
//! use real::Real;
//!
//! pub fn f(x: Real) -> real::Result {
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
//! [`Constraint::Error`]: crate::constraint::Constraint::Error
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

/// An output kind that can continue with an output.
pub trait Continue: Sealed {
    type As<P, E>;

    fn continue_with_output<P, E>(output: P) -> Self::As<P, E>;
}

/// An output kind that can break with an error.
pub trait Break: Continue {
    fn break_with_error<P, E>(error: E) -> Self::As<P, E>;
}

/// An [output kind][`Continue`] that outputs the identity.
///
/// [`Continue`]: crate::divergence::Continue
pub trait NonResidual<P, E>: Continue<As<P, E> = P> {}

impl<P, E, K> NonResidual<P, E> for K where K: Continue<As<P, E> = P> {}

#[derive(Debug)]
pub enum AsExpression {}

impl Break for AsExpression {
    fn break_with_error<P, E>(error: E) -> Self::As<P, E> {
        Undefined(error)
    }
}

impl Continue for AsExpression {
    type As<P, E> = Expression<P, E>;

    fn continue_with_output<P, E>(output: P) -> Self::As<P, E> {
        Defined(output)
    }
}

impl Sealed for AsExpression {}

#[derive(Debug)]
pub enum AsOption {}

impl Break for AsOption {
    fn break_with_error<P, E>(_: E) -> Self::As<P, E> {
        None
    }
}

impl Continue for AsOption {
    type As<P, E> = Option<P>;

    fn continue_with_output<P, E>(output: P) -> Self::As<P, E> {
        Some(output)
    }
}

impl Sealed for AsOption {}

#[derive(Debug)]
pub enum AsResult {}

impl Break for AsResult {
    fn break_with_error<P, E>(error: E) -> Self::As<P, E> {
        Err(error)
    }
}

impl Continue for AsResult {
    type As<P, E> = Result<P, E>;

    fn continue_with_output<P, E>(output: P) -> Self::As<P, E> {
        Ok(output)
    }
}

impl Sealed for AsResult {}

#[derive(Debug)]
pub enum AsSelf {}

impl Continue for AsSelf {
    type As<P, E> = P;

    fn continue_with_output<P, E>(output: P) -> Self::As<P, E> {
        output
    }
}

impl Sealed for AsSelf {}

/// Determines the output type and behavior of a [`Proxy`] when it is fallibly constructed.
///
/// The output type is defined by an associated [output **kind**][`Continue`]. Regardless of this
/// type, this trait implements continuing and breaking on the [`Result`] of constructing a
/// [`Proxy`]. See the [module documentation][`divergence`].
///
/// [`Continue`]: crate::divergence::Continue
/// [`divergence`]: crate::divergence
/// [`Proxy`]: crate::proxy::Proxy
/// [`Result`]: core::result::Result
pub trait Divergence: Sealed {
    type Continue: Continue;

    fn diverge<T, E>(result: Result<T, E>) -> <Self::Continue as Continue>::As<T, E>
    where
        E: Debug;
}

pub type ContinueOf<D> = <D as Divergence>::Continue;
pub type OutputOf<D, P, E> = <ContinueOf<D> as Continue>::As<P, E>;

/// Divergence that breaks on errors by **panicking**.
///
/// **`OrPanic` panics if a [`Proxy`] cannot be constructed.** This behavior is independent of the
/// output kind, so even an `OrPanic` divergence with a [`Result`] output type panics if an error
/// occurs.
///
/// By default, `OrPanic` uses the [`AsSelf`] output kind.
///
/// [`AsSelf`]: crate::divergence::AsSelf
/// [`Proxy`]: crate::proxy::Proxy
/// [`Result`]: core::result::Result
pub struct OrPanic<K = AsSelf>(PhantomData<fn() -> K>, Infallible);

impl<K> Divergence for OrPanic<K>
where
    K: Continue,
{
    type Continue = K;

    fn diverge<T, E>(result: Result<T, E>) -> K::As<T, E>
    where
        E: Debug,
    {
        K::continue_with_output(result.expect_constrained())
    }
}

impl<K> Sealed for OrPanic<K> {}

/// Divergence that breaks on errors by constructing an error representation of its output type.
///
/// The output kind `K` must support an error representation and implement [`Break`].
///
/// By default, `OrError` uses the [`AsExpression`] kind and therefore has an [`Expression`] output
/// type.
///
/// [`AsExpression`]: crate::divergence::AsExpression
/// [`Expression`]: crate::expression::Expression
pub struct OrError<K = AsExpression>(PhantomData<fn() -> K>, Infallible);

impl<K> Divergence for OrError<K>
where
    K: Break,
{
    type Continue = K;

    fn diverge<T, E>(result: Result<T, E>) -> K::As<T, E>
    where
        E: Debug,
    {
        match result {
            Ok(output) => K::continue_with_output(output),
            Err(error) => K::break_with_error(error),
        }
    }
}

impl<K> Sealed for OrError<K> {}
