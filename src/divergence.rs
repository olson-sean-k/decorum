//! Constraint divergence types and behaviors (error handling).
//!
//! # Error Behaviors
//!
//! This module provides marker types that express the behavior of [`Proxy`]s when a disallowed
//! IEEE 754 floating-point value is encountered. These types and behaviors are summarized in the
//! following table:
//!
//! | Divergence        | Branch Type        | Error Output   |
//! |-------------------|--------------------|----------------|
//! | [`Assert`]        | `T`                | **panic**      |
//! | [`TryOption`]     | `Option<T>`        | `None`         |
//! | [`TryResult`]     | `Result<T, E>`     | `Err(E)`       |
//! | [`TryExpression`] | `Expression<T, E>` | `Undefined(E)` |
//!
//! Note that [`Assert`] is the only intrinsic divergence where the output type of fallible
//! operations is the same as the input type. **However, this divergence panics when encountering a
//! value that violates a [constraint][`constraint`].** The remaining divergence types instead use
//! an extrinsic type to encode errors as values.

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

/// Divergence that panics when an error is encountered.
///
/// This divergence is always intrinsic, so its branch type is the same as the type involved in the
/// fallible construction.
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

/// Divergence that constructs an [`Undefined`] when an error is encountered.
///
/// This divergence is intrinsic with respect to [`Expression`]. For all other types, it is
/// extrinsic.
///
/// [`Undefined`]: crate::divergence::Expression::Undefined
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
