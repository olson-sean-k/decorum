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
//!
//! # Expressions
//!
//! The [`Expression`] type represents the result of an arithmetic expression that may or may not
//! be defined (as determined by [constraints][`constraint`]). It resembles [`Result`], but is
//! additionally a numeric type that can be used fluently in fallible arithmetic expressions
//! involving constrained proxy types.
//!
//! [`constraint`]: crate::constraint
//! [`Assert`]: crate::divergence::Assert
//! [`Proxy`]: crate::proxy::Proxy
//! [`Result`]: core::result::Result
//! [`TryExpression`]: crate::divergence::TryExpression
//! [`TryOption`]: crate::divergence::TryOption
//! [`TryResult`]: crate::divergence::TryResult

use core::cmp::Ordering;
use core::convert::Infallible;
use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Rem, Sub};
#[cfg(all(nightly, feature = "unstable"))]
use core::ops::{ControlFlow, FromResidual, Try};

use crate::cmp::UndefinedError;
use crate::constraint::{Constraint, ExpectConstrained as _};
use crate::proxy::{ClosedProxy, ErrorOf, ExpressionOf, Proxy};
use crate::sealed::Sealed;
use crate::{
    with_binary_operations, with_primitives, BinaryReal, Float, Function, Infinite, Primitive,
    UnaryReal,
};

pub use Expression::Defined;
pub use Expression::Undefined;

/// Unwraps an [`Expression`] or propagates its error.
///
/// This macro mirrors the standard [`try`] macro but operates on [`Expression`]s rather than
/// [`Result`]s. If the given [`Expression`] is the `Defined` variant, then the expression (of the
/// macro) is the accompanying value. Otherwise, the error in the `Undefined` variant is converted
/// via [`From`] and returned in the constructed [`Expression`].
///
/// [`Expression`]: crate::divergence::Expression
/// [`From`]: core::convert::From
/// [`Result`]: core::result::Result
/// [`try`]: core::try
#[macro_export]
macro_rules! try_expression {
    ($x:expr) => {
        match $x {
            $crate::divergence::Expression::Defined(inner) => inner,
            $crate::divergence::Expression::Undefined(error) => {
                return $crate::divergence::Expression::Undefined(core::convert::From::from(error));
            }
        }
    };
}

/// Determines the output type and behavior of a [`Proxy`] when it is fallibly constructed.
///
/// This trait defines a branch type and behavior when that type is constructed from an error. In
/// the error case, this may or may not yield a value and may instead diverge by panicking. When
/// constructed from an output, the branch type is always constructed and returned.
///
/// [`Proxy`]: crate::proxy::Proxy
pub trait Divergence: Sealed {
    type Branch<T, E>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E>;

    fn diverge<T, E>(error: E) -> Self::Branch<T, E>
    where
        E: Debug;
}

impl Divergence for Infallible {
    type Branch<T, E> = T;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        output
    }

    fn diverge<T, E>(_error: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        unreachable!()
    }
}

/// A [`Divergence`] with a branch type that is the same as its output type.
///
/// [`Divergence`]: crate::divergence::Divergence
pub trait NonResidual<P>: Divergence<Branch<P, ErrorOf<P>> = P>
where
    P: ClosedProxy,
{
}

impl<P, D> NonResidual<P> for D
where
    P: ClosedProxy,
    D: Divergence<Branch<P, ErrorOf<P>> = P>,
{
}

/// Divergence that panics when an error is encountered.
///
/// This divergence is always intrinsic, so its branch type is the same as the type involved in the
/// fallible construction.
pub enum Assert {}

impl Divergence for Assert {
    type Branch<T, E> = T;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        output
    }

    fn diverge<T, E>(error: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        Err(error).expect_constrained()
    }
}

impl Sealed for Assert {}

/// Divergence that constructs an [`Undefined`] when an error is encountered.
///
/// This divergence is intrinsic with respect to [`Expression`]. For all other types, it is
/// extrinsic.
///
/// [`Undefined`]: crate::divergence::Expression::Undefined
pub enum TryExpression {}

impl Divergence for TryExpression {
    type Branch<T, E> = Expression<T, E>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        Defined(output)
    }

    fn diverge<T, E>(error: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        Undefined(error)
    }
}

impl Sealed for TryExpression {}

/// Divergence that constructs a [`None`] when an error is encountered.
///
/// This divergence is always extrinsic, so its branch type differs from the type involved in the
/// fallible construction.
///
/// [`None`]: core::option::Option::None
pub enum TryOption {}

impl Divergence for TryOption {
    type Branch<T, E> = Option<T>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        Some(output)
    }

    fn diverge<T, E>(_residual: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        None
    }
}

impl Sealed for TryOption {}

/// Divergence that constructs an [`Err`] when an error is encountered.
///
/// This divergence is always extrinsic, so its branch type differs from the type involved in the
/// fallible construction.
///
/// [`Err`]: core::result::Result::Err
pub enum TryResult {}

impl Divergence for TryResult {
    type Branch<T, E> = Result<T, E>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        Ok(output)
    }

    fn diverge<T, E>(error: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        Err(error)
    }
}

impl Sealed for TryResult {}

/// The result of an arithmetic expression that may or may not be defined.
///
/// `Expression` is a fallible output of arithmetic expressions over [`Proxy`] types. It resembles
/// [`Result`], but `Expression` crucially implements numeric traits and can be used in arithmetic
/// expressions. This allows complex expressions to defer matching or trying for more fluent
/// syntax.
///
/// When the `unstable` Cargo feature is enabled with a nightly Rust toolchain, [`Expression`] also
/// implements the unstable (at time of writing) [`Try`] trait and supports the try operator `?`.
///
/// # Examples
///
/// The following two examples contrast deferred matching and trying of `Expression`s versus
/// immediate matching and trying of `Result`s.
///
/// ```rust
/// use decorum::constraint::FiniteConstraint;
/// use decorum::divergence::TryExpression;
/// use decorum::proxy::{BranchOf, Proxy};
/// use decorum::real::UnaryReal;
/// use decorum::try_expression;
///
/// pub type Real = Proxy<f64, FiniteConstraint<TryExpression>>;
/// pub type Expr = BranchOf<Real>;
///
/// # fn fallible() -> Expr {
/// fn f(x: Real, y: Real, z: Real) -> Expr {
///     let w = (x + y + z);
///     w / Real::ONE
/// }
///
/// let x = Real::ONE;
/// let y: Real = try_expression!(f(x, x, x));
/// // ...
/// # f(x, x, x)
/// # }
/// ```
///
/// ```rust
/// use decorum::constraint::FiniteConstraint;
/// use decorum::divergence::TryResult;
/// use decorum::proxy::{BranchOf, Proxy};
/// use decorum::real::UnaryReal;
///
/// pub type Real = Proxy<f64, FiniteConstraint<TryResult>>;
/// pub type RealResult = BranchOf<Real>;
///
/// # fn fallible() -> RealResult {
/// fn f(x: Real, y: Real, z: Real) -> RealResult {
///     // The expression `x + y` outputs a `Result`, which cannot be used in a mathematical
///     // expression, so it must be tried first.
///     let w = ((x + y)? + z)?;
///     w / Real::ONE
/// }
///
/// let x = Real::ONE;
/// let y: Real = f(x, x, x)?;
/// // ...
/// # f(x, x, x)
/// # }
/// ```
///
/// When the `unstable` Cargo feature is enabled with a nightly Rust toolchain, `Expression`
/// supports the try operator `?`.
///
/// ```rust,ignore
/// use decorum::constraint::FiniteConstraint;
/// use decorum::divergence::TryExpression;
/// use decorum::proxy::{BranchOf, Proxy};
/// use decorum::real::UnaryReal;
///
/// pub type Real = Proxy<f64, FiniteConstraint<TryExpression>>;
/// pub type Expr = BranchOf<Real>;
///
/// # fn fallible() -> Expr {
/// fn f(x: Real, y: Real, z: Real) -> Expr {
///     let w = (x + y + z)?; // Try.
///     eprintln!("x + y + z => defined!");
///     w / Real::ONE
/// }
///
/// let x = Real::ONE;
/// let y = Real::ZERO;
/// let z = f(x, y, x)?; // Try.
/// eprintln!("f(x, y, x) => defined!");
/// // ...
/// # f(x, y, x)
/// # }
/// ```
///
/// [`Proxy`]: crate::proxy::Proxy
/// [`Result`]: core::result::Result
/// [`Try`]: core::ops::Try
#[derive(Clone, Copy, Debug)]
pub enum Expression<T, E = ()> {
    Defined(T),
    Undefined(E),
}

impl<T, E> Expression<T, E> {
    pub fn unwrap(self) -> T {
        match self {
            Defined(defined) => defined,
            _ => panic!(),
        }
    }

    pub fn map<U, F>(self, f: F) -> Expression<U, E>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Defined(defined) => Defined(f(defined)),
            Undefined(undefined) => Undefined(undefined),
        }
    }

    pub fn and_then<U, F>(self, f: F) -> Expression<U, E>
    where
        F: FnOnce(T) -> Expression<U, E>,
    {
        match self {
            Defined(defined) => f(defined),
            Undefined(undefined) => Undefined(undefined),
        }
    }

    pub fn defined(&self) -> Option<&T> {
        match self {
            Defined(ref defined) => Some(defined),
            _ => None,
        }
    }

    pub fn undefined(&self) -> Option<&E> {
        match self {
            Undefined(ref undefined) => Some(undefined),
            _ => None,
        }
    }

    pub fn is_defined(&self) -> bool {
        matches!(self, Defined(_))
    }

    pub fn is_undefined(&self) -> bool {
        matches!(self, Undefined(_))
    }
}

impl<T, C> BinaryReal for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Self) -> Self::Codomain {
        BinaryReal::div_euclid(try_expression!(self), try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Self) -> Self::Codomain {
        BinaryReal::rem_euclid(try_expression!(self), try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Self) -> Self::Codomain {
        BinaryReal::pow(try_expression!(self), try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self::Codomain {
        BinaryReal::log(try_expression!(self), try_expression!(base))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self::Codomain {
        BinaryReal::hypot(try_expression!(self), try_expression!(other))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self::Codomain {
        BinaryReal::atan2(try_expression!(self), try_expression!(other))
    }
}

impl<T, C> BinaryReal<T> for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: T) -> Self::Codomain {
        BinaryReal::div_euclid(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(n)),
        )
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: T) -> Self::Codomain {
        BinaryReal::rem_euclid(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(n)),
        )
    }

    #[cfg(feature = "std")]
    fn pow(self, n: T) -> Self::Codomain {
        BinaryReal::pow(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(n)),
        )
    }

    #[cfg(feature = "std")]
    fn log(self, base: T) -> Self::Codomain {
        BinaryReal::log(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(base)),
        )
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: T) -> Self::Codomain {
        BinaryReal::hypot(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(other)),
        )
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: T) -> Self::Codomain {
        BinaryReal::atan2(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(other)),
        )
    }
}

impl<T, C> BinaryReal<Proxy<T, C>> for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Proxy<T, C>) -> Self::Codomain {
        BinaryReal::div_euclid(try_expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Proxy<T, C>) -> Self::Codomain {
        BinaryReal::rem_euclid(try_expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Proxy<T, C>) -> Self::Codomain {
        BinaryReal::pow(try_expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Proxy<T, C>) -> Self::Codomain {
        BinaryReal::log(try_expression!(self), base)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Proxy<T, C>) -> Self::Codomain {
        BinaryReal::hypot(try_expression!(self), other)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Proxy<T, C>) -> Self::Codomain {
        BinaryReal::atan2(try_expression!(self), other)
    }
}

impl<T, C> BinaryReal<ExpressionOf<Proxy<T, C>>> for Proxy<T, C>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryReal::div_euclid(self, try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryReal::rem_euclid(self, try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryReal::pow(self, try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn log(self, base: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryReal::log(self, try_expression!(base))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryReal::hypot(self, try_expression!(other))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryReal::atan2(self, try_expression!(other))
    }
}

impl<T, C> From<T> for Expression<Proxy<T, C>, ErrorOf<Proxy<T, C>>>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn from(inner: T) -> Self {
        Proxy::try_new(inner).into()
    }
}

impl<'a, T, C> From<&'a T> for ExpressionOf<Proxy<T, C>>
where
    Proxy<T, C>: TryFrom<&'a T, Error = C::Error>,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    fn from(inner: &'a T) -> Self {
        Proxy::<T, C>::try_from(inner).into()
    }
}

impl<'a, T, C> From<&'a mut T> for ExpressionOf<Proxy<T, C>>
where
    Proxy<T, C>: TryFrom<&'a mut T, Error = C::Error>,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    fn from(inner: &'a mut T) -> Self {
        Proxy::<T, C>::try_from(inner).into()
    }
}

impl<T, C> From<Proxy<T, C>> for Expression<Proxy<T, C>, ErrorOf<Proxy<T, C>>>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn from(proxy: Proxy<T, C>) -> Self {
        Defined(proxy)
    }
}

impl<T, E> From<Result<T, E>> for Expression<T, E> {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(output) => Defined(output),
            Err(error) => Undefined(error),
        }
    }
}

impl<T, E> From<Expression<T, E>> for Result<T, E> {
    fn from(result: Expression<T, E>) -> Self {
        match result {
            Defined(defined) => Ok(defined),
            Undefined(undefined) => Err(undefined),
        }
    }
}

#[cfg(all(nightly, feature = "unstable"))]
impl<T, E> FromResidual for Expression<T, E> {
    fn from_residual(expression: Expression<Infallible, E>) -> Self {
        match expression {
            Undefined(undefined) => Undefined(undefined),
            _ => unreachable!(),
        }
    }
}

impl<T, C> Function for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    type Codomain = Self;
}

impl<T, C> Infinite for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Copy,
    Proxy<T, C>: Infinite,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    const INFINITY: Self = Defined(Infinite::INFINITY);
    const NEG_INFINITY: Self = Defined(Infinite::NEG_INFINITY);

    fn is_infinite(self) -> bool {
        self.defined()
            .map_or(false, |defined| defined.is_infinite())
    }

    fn is_finite(self) -> bool {
        self.defined().map_or(false, |defined| defined.is_finite())
    }
}

impl<T, E> PartialEq for Expression<T, E>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.defined()
            .zip(other.defined())
            .map_or(false, |(left, right)| left.eq(right))
    }
}

impl<T, E> PartialOrd for Expression<T, E>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.defined()
            .zip(other.defined())
            .and_then(|(left, right)| left.partial_cmp(right))
    }
}

#[cfg(all(nightly, feature = "unstable"))]
impl<T, E> Try for Expression<T, E> {
    type Output = T;
    type Residual = Expression<Infallible, E>;

    fn from_output(output: T) -> Self {
        Defined(output)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Defined(defined) => ControlFlow::Continue(defined),
            Undefined(undefined) => ControlFlow::Break(Undefined(undefined)),
        }
    }
}

impl<T, C> UnaryReal for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    const ZERO: Self = Defined(UnaryReal::ZERO);
    const ONE: Self = Defined(UnaryReal::ONE);
    const E: Self = Defined(UnaryReal::E);
    const PI: Self = Defined(UnaryReal::PI);
    const FRAC_1_PI: Self = Defined(UnaryReal::FRAC_1_PI);
    const FRAC_2_PI: Self = Defined(UnaryReal::FRAC_2_PI);
    const FRAC_2_SQRT_PI: Self = Defined(UnaryReal::FRAC_2_SQRT_PI);
    const FRAC_PI_2: Self = Defined(UnaryReal::FRAC_PI_2);
    const FRAC_PI_3: Self = Defined(UnaryReal::FRAC_PI_3);
    const FRAC_PI_4: Self = Defined(UnaryReal::FRAC_PI_4);
    const FRAC_PI_6: Self = Defined(UnaryReal::FRAC_PI_6);
    const FRAC_PI_8: Self = Defined(UnaryReal::FRAC_PI_8);
    const SQRT_2: Self = Defined(UnaryReal::SQRT_2);
    const FRAC_1_SQRT_2: Self = Defined(UnaryReal::FRAC_1_SQRT_2);
    const LN_2: Self = Defined(UnaryReal::LN_2);
    const LN_10: Self = Defined(UnaryReal::LN_10);
    const LOG2_E: Self = Defined(UnaryReal::LOG2_E);
    const LOG10_E: Self = Defined(UnaryReal::LOG10_E);

    fn is_zero(self) -> bool {
        self.defined().map_or(false, |defined| defined.is_zero())
    }

    fn is_one(self) -> bool {
        self.defined().map_or(false, |defined| defined.is_one())
    }

    fn is_positive(self) -> bool {
        self.defined()
            .map_or(false, |defined| defined.is_positive())
    }

    fn is_negative(self) -> bool {
        self.defined()
            .map_or(false, |defined| defined.is_negative())
    }

    #[cfg(feature = "std")]
    fn abs(self) -> Self {
        self.map(UnaryReal::abs)
    }

    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        self.map(UnaryReal::signum)
    }

    fn floor(self) -> Self {
        self.map(UnaryReal::floor)
    }

    fn ceil(self) -> Self {
        self.map(UnaryReal::ceil)
    }

    fn round(self) -> Self {
        self.map(UnaryReal::round)
    }

    fn trunc(self) -> Self {
        self.map(UnaryReal::trunc)
    }

    fn fract(self) -> Self {
        self.map(UnaryReal::fract)
    }

    fn recip(self) -> Self::Codomain {
        self.and_then(UnaryReal::recip)
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Codomain {
        self.and_then(|defined| UnaryReal::powi(defined, n))
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Codomain {
        self.and_then(UnaryReal::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map(UnaryReal::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self::Codomain {
        self.and_then(UnaryReal::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Codomain {
        self.and_then(UnaryReal::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Codomain {
        self.and_then(UnaryReal::exp_m1)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self::Codomain {
        self.and_then(UnaryReal::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self::Codomain {
        self.and_then(UnaryReal::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self::Codomain {
        self.and_then(UnaryReal::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self::Codomain {
        self.and_then(UnaryReal::ln_1p)
    }

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Codomain {
        self.and_then(UnaryReal::to_degrees)
    }

    #[cfg(feature = "std")]
    fn to_radians(self) -> Self {
        self.map(UnaryReal::to_radians)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map(UnaryReal::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map(UnaryReal::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self::Codomain {
        self.and_then(UnaryReal::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self::Codomain {
        self.and_then(UnaryReal::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self::Codomain {
        self.and_then(UnaryReal::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map(UnaryReal::atan)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        match self {
            Defined(defined) => {
                let (sin, cos) = defined.sin_cos();
                (Defined(sin), Defined(cos))
            }
            Undefined(undefined) => (Undefined(undefined.clone()), Undefined(undefined)),
        }
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        self.map(UnaryReal::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map(UnaryReal::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map(UnaryReal::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Codomain {
        self.and_then(UnaryReal::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Codomain {
        self.and_then(UnaryReal::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Codomain {
        self.and_then(UnaryReal::atanh)
    }
}

macro_rules! impl_binary_operation {
    () => {
        with_binary_operations!(impl_binary_operation);
    };
    (operation => $trait:ident :: $method:ident) => {
        impl_binary_operation!(operation => $trait :: $method, |left, right| {
            left.zip_map(right, $trait::$method)
        });
    };
    (operation => $trait:ident :: $method:ident, |$left:ident, $right:ident| $f:block) => {
        macro_rules! impl_primitive_binary_operation {
            () => {
                with_primitives!(impl_primitive_binary_operation);
            };
            (primitive => $t:ty) => {
                impl<C> $trait<ExpressionOf<Proxy<$t, C>>> for $t
                where
                    C: Constraint<Divergence = TryExpression>,
                {
                    type Output = ExpressionOf<Proxy<$t, C>>;

                    fn $method(self, other: ExpressionOf<Proxy<$t, C>>) -> Self::Output {
                        let $left = try_expression!(Proxy::<_, C>::new(self));
                        let $right = try_expression!(other);
                        $f
                    }
                }
            };
        }
        impl_primitive_binary_operation!();

        impl<T, C> $trait<ExpressionOf<Self>> for Proxy<T, C>
        where
            T: Float + Primitive,
            C: Constraint<Divergence = TryExpression>,
        {
            type Output = ExpressionOf<Self>;

            fn $method(self, other: ExpressionOf<Self>) -> Self::Output {
                let $left = self;
                let $right = try_expression!(other);
                $f
            }
        }

        impl<T, C> $trait<Proxy<T, C>> for ExpressionOf<Proxy<T, C>>
        where
            T: Float + Primitive,
            C: Constraint<Divergence = TryExpression>,
        {
            type Output = Self;

            fn $method(self, other: Proxy<T, C>) -> Self::Output {
                let $left = try_expression!(self);
                let $right = other;
                $f
            }
        }

        impl<T, C> $trait<ExpressionOf<Proxy<T, C>>> for ExpressionOf<Proxy<T, C>>
        where
            T: Float + Primitive,
            C: Constraint<Divergence = TryExpression>,
        {
            type Output = Self;

            fn $method(self, other: Self) -> Self::Output {
                let $left = try_expression!(self);
                let $right = try_expression!(other);
                $f
            }
        }

        impl<T, C> $trait<T> for ExpressionOf<Proxy<T, C>>
        where
            T: Float + Primitive,
            C: Constraint<Divergence = TryExpression>,
        {
            type Output = Self;

            fn $method(self, other: T) -> Self::Output {
                let $left = try_expression!(self);
                let $right = try_expression!(Proxy::<_, C>::new(other));
                $f
            }
        }
    };
}
impl_binary_operation!();

macro_rules! impl_try_from {
    () => {
        with_primitives!(impl_try_from);
    };
    (primitive => $t:ty) => {
        impl<C> TryFrom<Expression<Proxy<$t, C>, C::Error>> for Proxy<$t, C>
        where
            C: Constraint,
        {
            type Error = C::Error;

            fn try_from(
                expression: Expression<Proxy<$t, C>, C::Error>,
            ) -> Result<Self, Self::Error> {
                match expression {
                    Defined(defined) => Ok(defined),
                    Undefined(undefined) => Err(undefined),
                }
            }
        }

        impl<C> TryFrom<Expression<Proxy<$t, C>, C::Error>> for $t
        where
            C: Constraint,
        {
            type Error = C::Error;

            fn try_from(
                expression: Expression<Proxy<$t, C>, C::Error>,
            ) -> Result<Self, Self::Error> {
                match expression {
                    Defined(defined) => Ok(defined.into()),
                    Undefined(undefined) => Err(undefined),
                }
            }
        }
    };
}
impl_try_from!();
