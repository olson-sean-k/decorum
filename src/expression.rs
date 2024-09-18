use core::cmp::Ordering;
#[cfg(all(nightly, feature = "unstable"))]
use core::convert::Infallible;
use core::fmt::Debug;
#[cfg(all(nightly, feature = "unstable"))]
use core::ops::{self, ControlFlow, FromResidual};
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::cmp::UndefinedError;
use crate::constraint::Constraint;
use crate::divergence::{AsExpression, OrError};
use crate::proxy::{ErrorOf, ExpressionOf, Proxy};
use crate::real::{BinaryRealFunction, Function, Sign, UnaryRealFunction};
use crate::{with_binary_operations, with_primitives, Float, Infinite, Primitive};

pub use Expression::Defined;
pub use Expression::Undefined;

/// Unwraps an [`Expression`] or propagates its error.
///
/// This macro mirrors the standard [`try`] macro but operates on [`Expression`]s rather than
/// [`Result`]s. If the given [`Expression`] is the `Defined` variant, then the expression (of the
/// macro) is the accompanying value. Otherwise, the error in the `Undefined` variant is converted
/// via [`From`] and returned in the constructed [`Expression`].
///
/// [`Expression`]: crate::expression::Expression
/// [`From`]: core::convert::From
/// [`Result`]: core::result::Result
/// [`try`]: core::try
#[macro_export]
macro_rules! try_expression {
    ($x:expr $(,)?) => {{
        let expression: $crate::expression::Expression<_, _> = $x;
        match expression {
            $crate::expression::Expression::Defined(inner) => inner,
            $crate::expression::Expression::Undefined(error) => {
                return $crate::expression::Expression::Undefined(core::convert::From::from(error));
            }
        }
    }};
    ($x:block $(,)?) => {
        let expression: $crate::expression::Expression<_, _> = $x;
        try_expression!(expression);
    };
}

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
/// use decorum::constraint::IsReal;
/// use decorum::divergence::OrError;
/// use decorum::proxy::{OutputOf, Proxy};
/// use decorum::real::UnaryRealFunction;
/// use decorum::try_expression;
///
/// pub type Real = Proxy<f64, IsReal<OrError>>;
/// pub type Expr = OutputOf<Real>;
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
/// use decorum::constraint::IsReal;
/// use decorum::divergence::{AsResult, OrError};
/// use decorum::proxy::{OutputOf, Proxy};
/// use decorum::real::UnaryRealFunction;
///
/// pub type Real = Proxy<f64, IsReal<OrError<AsResult>>>;
/// pub type RealResult = OutputOf<Real>;
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
/// use decorum::constraint::IsReal;
/// use decorum::divergence::{AsExpression, OrError};
/// use decorum::proxy::{OutputOf, Proxy};
/// use decorum::real::UnaryRealFunction;
///
/// pub type Real = Proxy<f64, IsReal<OrError<AsExpression>>>;
/// pub type Expr = OutputOf<Real>;
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

    pub fn as_ref(&self) -> Expression<&T, &E> {
        match self {
            Defined(ref defined) => Defined(defined),
            Undefined(ref undefined) => Undefined(undefined),
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

    pub fn defined(self) -> Option<T> {
        match self {
            Defined(defined) => Some(defined),
            _ => None,
        }
    }

    pub fn undefined(self) -> Option<E> {
        match self {
            Undefined(undefined) => Some(undefined),
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

impl<'a, T, E> Expression<&'a T, E> {
    pub fn copied(self) -> Expression<T, E>
    where
        T: Copy,
    {
        match self {
            Defined(defined) => Defined(*defined),
            Undefined(undefined) => Undefined(undefined),
        }
    }

    pub fn cloned(self) -> Expression<T, E>
    where
        T: Clone,
    {
        match self {
            Defined(defined) => Defined(defined.clone()),
            Undefined(undefined) => Undefined(undefined),
        }
    }
}

impl<'a, T, E> Expression<&'a mut T, E> {
    pub fn copied(self) -> Expression<T, E>
    where
        T: Copy,
    {
        match self {
            Defined(defined) => Defined(*defined),
            Undefined(undefined) => Undefined(undefined),
        }
    }

    pub fn cloned(self) -> Expression<T, E>
    where
        T: Clone,
    {
        match self {
            Defined(defined) => Defined(defined.clone()),
            Undefined(undefined) => Undefined(undefined),
        }
    }
}

impl<T, C> BinaryRealFunction for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Self) -> Self::Codomain {
        BinaryRealFunction::div_euclid(try_expression!(self), try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Self) -> Self::Codomain {
        BinaryRealFunction::rem_euclid(try_expression!(self), try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Self) -> Self::Codomain {
        BinaryRealFunction::pow(try_expression!(self), try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self::Codomain {
        BinaryRealFunction::log(try_expression!(self), try_expression!(base))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self::Codomain {
        BinaryRealFunction::hypot(try_expression!(self), try_expression!(other))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self::Codomain {
        BinaryRealFunction::atan2(try_expression!(self), try_expression!(other))
    }
}

impl<T, C> BinaryRealFunction<T> for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: T) -> Self::Codomain {
        BinaryRealFunction::div_euclid(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(n)),
        )
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: T) -> Self::Codomain {
        BinaryRealFunction::rem_euclid(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(n)),
        )
    }

    #[cfg(feature = "std")]
    fn pow(self, n: T) -> Self::Codomain {
        BinaryRealFunction::pow(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(n)),
        )
    }

    #[cfg(feature = "std")]
    fn log(self, base: T) -> Self::Codomain {
        BinaryRealFunction::log(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(base)),
        )
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: T) -> Self::Codomain {
        BinaryRealFunction::hypot(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(other)),
        )
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: T) -> Self::Codomain {
        BinaryRealFunction::atan2(
            try_expression!(self),
            try_expression!(Proxy::<T, C>::new(other)),
        )
    }
}

impl<T, C> BinaryRealFunction<Proxy<T, C>> for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Proxy<T, C>) -> Self::Codomain {
        BinaryRealFunction::div_euclid(try_expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Proxy<T, C>) -> Self::Codomain {
        BinaryRealFunction::rem_euclid(try_expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Proxy<T, C>) -> Self::Codomain {
        BinaryRealFunction::pow(try_expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Proxy<T, C>) -> Self::Codomain {
        BinaryRealFunction::log(try_expression!(self), base)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Proxy<T, C>) -> Self::Codomain {
        BinaryRealFunction::hypot(try_expression!(self), other)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Proxy<T, C>) -> Self::Codomain {
        BinaryRealFunction::atan2(try_expression!(self), other)
    }
}

impl<T, C> BinaryRealFunction<ExpressionOf<Proxy<T, C>>> for Proxy<T, C>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryRealFunction::div_euclid(self, try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryRealFunction::rem_euclid(self, try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryRealFunction::pow(self, try_expression!(n))
    }

    #[cfg(feature = "std")]
    fn log(self, base: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryRealFunction::log(self, try_expression!(base))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryRealFunction::hypot(self, try_expression!(other))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: ExpressionOf<Proxy<T, C>>) -> Self::Codomain {
        BinaryRealFunction::atan2(self, try_expression!(other))
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
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    fn from(inner: &'a T) -> Self {
        Proxy::<T, C>::try_from(inner).into()
    }
}

impl<'a, T, C> From<&'a mut T> for ExpressionOf<Proxy<T, C>>
where
    Proxy<T, C>: TryFrom<&'a mut T, Error = C::Error>,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
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
    fn from_residual(residual: Expression<Infallible, E>) -> Self {
        use core::hint;

        #[allow(unreachable_patterns)]
        match residual {
            Undefined(undefined) => Undefined(undefined),
            // SAFETY: `Infallible` is uninhabited, so it is not possible to construct the
            //         `Defined` variant here.
            Defined(_) => unsafe { hint::unreachable_unchecked() },
        }
    }
}

impl<T, C> Function for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    type Codomain = Self;
}

impl<T, C> Infinite for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Copy,
    Proxy<T, C>: Infinite,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
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

impl<T, C> Neg for ExpressionOf<Proxy<T, C>>
where
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|defined| -defined)
    }
}

impl<T, E> PartialEq for Expression<T, E>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_ref()
            .defined()
            .zip(other.as_ref().defined())
            .map_or(false, |(left, right)| left.eq(right))
    }
}

impl<T, E> PartialOrd for Expression<T, E>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref()
            .defined()
            .zip(other.as_ref().defined())
            .and_then(|(left, right)| left.partial_cmp(right))
    }
}

#[cfg(all(nightly, feature = "unstable"))]
impl<T, E> ops::Try for Expression<T, E> {
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

impl<T, C> UnaryRealFunction for ExpressionOf<Proxy<T, C>>
where
    ErrorOf<Proxy<T, C>>: Clone + UndefinedError,
    T: Float + Primitive,
    C: Constraint<Divergence = OrError<AsExpression>>,
{
    const ZERO: Self = Defined(UnaryRealFunction::ZERO);
    const ONE: Self = Defined(UnaryRealFunction::ONE);
    const E: Self = Defined(UnaryRealFunction::E);
    const PI: Self = Defined(UnaryRealFunction::PI);
    const FRAC_1_PI: Self = Defined(UnaryRealFunction::FRAC_1_PI);
    const FRAC_2_PI: Self = Defined(UnaryRealFunction::FRAC_2_PI);
    const FRAC_2_SQRT_PI: Self = Defined(UnaryRealFunction::FRAC_2_SQRT_PI);
    const FRAC_PI_2: Self = Defined(UnaryRealFunction::FRAC_PI_2);
    const FRAC_PI_3: Self = Defined(UnaryRealFunction::FRAC_PI_3);
    const FRAC_PI_4: Self = Defined(UnaryRealFunction::FRAC_PI_4);
    const FRAC_PI_6: Self = Defined(UnaryRealFunction::FRAC_PI_6);
    const FRAC_PI_8: Self = Defined(UnaryRealFunction::FRAC_PI_8);
    const SQRT_2: Self = Defined(UnaryRealFunction::SQRT_2);
    const FRAC_1_SQRT_2: Self = Defined(UnaryRealFunction::FRAC_1_SQRT_2);
    const LN_2: Self = Defined(UnaryRealFunction::LN_2);
    const LN_10: Self = Defined(UnaryRealFunction::LN_10);
    const LOG2_E: Self = Defined(UnaryRealFunction::LOG2_E);
    const LOG10_E: Self = Defined(UnaryRealFunction::LOG10_E);

    fn is_zero(self) -> bool {
        self.defined().map_or(false, |defined| defined.is_zero())
    }

    fn is_one(self) -> bool {
        self.defined().map_or(false, |defined| defined.is_one())
    }

    fn sign(self) -> Sign {
        self.defined().map_or(Sign::Zero, |defined| defined.sign())
    }

    #[cfg(feature = "std")]
    fn abs(self) -> Self {
        self.map(UnaryRealFunction::abs)
    }

    #[cfg(feature = "std")]
    fn floor(self) -> Self {
        self.map(UnaryRealFunction::floor)
    }

    #[cfg(feature = "std")]
    fn ceil(self) -> Self {
        self.map(UnaryRealFunction::ceil)
    }

    #[cfg(feature = "std")]
    fn round(self) -> Self {
        self.map(UnaryRealFunction::round)
    }

    #[cfg(feature = "std")]
    fn trunc(self) -> Self {
        self.map(UnaryRealFunction::trunc)
    }

    #[cfg(feature = "std")]
    fn fract(self) -> Self {
        self.map(UnaryRealFunction::fract)
    }

    fn recip(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::recip)
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Codomain {
        self.and_then(|defined| UnaryRealFunction::powi(defined, n))
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map(UnaryRealFunction::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::exp_m1)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::ln_1p)
    }

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::to_degrees)
    }

    #[cfg(feature = "std")]
    fn to_radians(self) -> Self {
        self.map(UnaryRealFunction::to_radians)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map(UnaryRealFunction::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map(UnaryRealFunction::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map(UnaryRealFunction::atan)
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
        self.map(UnaryRealFunction::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map(UnaryRealFunction::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map(UnaryRealFunction::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Codomain {
        self.and_then(UnaryRealFunction::atanh)
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
                    C: Constraint<Divergence = OrError<AsExpression>>,
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
            C: Constraint<Divergence = OrError<AsExpression>>,
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
            C: Constraint<Divergence = OrError<AsExpression>>,
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
            C: Constraint<Divergence = OrError<AsExpression>>,
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
            C: Constraint<Divergence = OrError<AsExpression>>,
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
