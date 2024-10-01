#[cfg(feature = "approx")]
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display, Formatter, LowerExp, UpperExp};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::mem;
use core::num::FpCategory;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::str::FromStr;
#[cfg(not(feature = "std"))]
use num_traits::float::FloatCore as Float;
#[cfg(feature = "std")]
use num_traits::Float;
use num_traits::{
    Bounded, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};
#[cfg(feature = "serde")]
use serde_derive::{Deserialize, Serialize};

use crate::cmp::{CanonicalEq, CanonicalOrd, IntrinsicOrd, IntrinsicUndefined};
use crate::constraint::{
    Constraint, ExpectConstrained, InfinitySet, IsExtendedReal, IsFloat, IsReal, Member, NanSet,
    SubsetOf, SupersetOf,
};
use crate::divergence::{self, Divergence, NonResidual};
use crate::expression::Expression;
use crate::hash::CanonicalHash;
use crate::proxy::Proxy;
#[cfg(feature = "serde")]
use crate::proxy::Serde;
use crate::real::{BinaryRealFunction, Function, Sign, UnaryRealFunction};
use crate::sealed::StaticDebug;
use crate::{
    with_binary_operations, with_primitives, BaseEncoding, ExtendedReal, InfinityEncoding,
    NanEncoding, Primitive, Real, ToCanonical, Total,
};

pub type OutputOf<P> = divergence::OutputOf<DivergenceOf<P>, P, ErrorOf<P>>;
pub type ConstraintOf<P> = <P as ConstrainedProxy>::Constraint;
pub type DivergenceOf<P> = <ConstraintOf<P> as Constraint>::Divergence;
pub type ErrorOf<P> = <ConstraintOf<P> as Constraint>::Error;
pub type ExpressionOf<P> = Expression<P, ErrorOf<P>>;

/// A constrained IEEE 754 floating-point proxy type.
pub trait ConstrainedProxy: Proxy {
    type Constraint: Constraint;
}

/// IEEE 754 floating-point proxy that provides total ordering, equivalence, hashing, constraints,
/// and error handling.
///
/// `Constrained` types wrap primitive floating-point type and extend their behavior. For example,
/// all proxy types implement the standard [`Eq`], [`Hash`], and [`Ord`] traits, sometimes via the
/// non-standard relations described in the [`cmp`] module when `NaN`s must be considered.
/// Constraints and divergence can be composed to determine the subset of floating-point values
/// that a proxy supports and how the proxy behaves when those constraints are violated.
///
/// Various type definitions are provided for various useful proxy constructions, such as the
/// [`Total`] type, which extends floating-point types with a non-standard total ordering.
///
/// [`cmp`]: crate::cmp
/// [`Eq`]: core::cmp::Eq
/// [`Hash`]: core::hash::Hash
/// [`Ord`]: core::cmp::Ord
/// [`Total`]: crate::Total
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            deserialize = "T: serde::Deserialize<'de> + Primitive, \
                           C: Constraint, \
                           C::Error: Display",
            serialize = "T: Primitive + serde::Serialize, \
                         C: Constraint"
        ),
        try_from = "Serde<T>",
        into = "Serde<T>"
    )
)]
#[repr(transparent)]
pub struct Constrained<T, C> {
    inner: T,
    #[cfg_attr(feature = "serde", serde(skip))]
    phantom: PhantomData<fn() -> C>,
}

impl<T, C> Constrained<T, C> {
    pub(crate) const fn unchecked(inner: T) -> Self {
        Constrained {
            inner,
            phantom: PhantomData,
        }
    }

    pub(crate) fn with_inner<U, F>(self, f: F) -> U
    where
        F: FnOnce(T) -> U,
    {
        f(self.inner)
    }
}

impl<T, C> Constrained<T, C>
where
    T: Copy,
{
    /// Converts a proxy into its underlying primitive floating-point type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use decorum::R64;
    ///
    /// fn f() -> R64 {
    /// #    use decorum::real::UnaryRealFunction;
    /// #    R64::ZERO
    ///     // ...
    /// }
    ///
    /// let x: f64 = f().into_inner();
    /// // The standard `From` and `Into` traits can also be used.
    /// let y: f64 = f().into();
    /// ```
    pub const fn into_inner(self) -> T {
        self.inner
    }

    pub(crate) fn map_unchecked<F>(self, f: F) -> Self
    where
        F: FnOnce(T) -> T,
    {
        Constrained::unchecked(f(self.into_inner()))
    }
}

impl<T, C> Constrained<T, C>
where
    T: Debug,
    C: StaticDebug,
{
    /// Writes a thorough [debugging][`Debug`] description of the proxy to the given [`Formatter`].
    ///
    /// This function is similar to [`debug`], but writes a verbose description of the proxy into a
    /// [`Formatter`] rather than returning a [`Debug`] implementation.
    ///
    /// [`debug`]: crate::proxy::Constrained::debug
    /// [`Debug`]: core::fmt::Debug
    /// [`Formatter`]: core::fmt::Formatter
    pub fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "Constrained<")?;
        C::fmt(formatter)?;
        write!(formatter, ">({:?})", self.inner)
    }

    /// Gets a [`Debug`] implementation that thoroughly describes the proxy.
    ///
    /// `Constrained` types implement [`Display`] and [`Debug`], but these implementations omit
    /// more specific information about [constraints][`constraint`] and [divergence]. This function
    /// provides an instance of a verbose [`Debug`] type that more thoroughly describes the
    /// behavior of the proxy.
    ///
    /// [`constraint`]: crate::constraint
    /// [`Debug`]: core::fmt::Debug
    /// [`Display`]: core::fmt::Display
    /// [`divergence`]: crate::divergence
    pub const fn debug(&self) -> impl '_ + Copy + Debug {
        struct Formatted<'a, T, C>(&'a Constrained<T, C>);

        impl<'a, T, C> Clone for Formatted<'a, T, C> {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<'a, T, C> Copy for Formatted<'a, T, C> {}

        impl<'a, T, C> Debug for Formatted<'a, T, C>
        where
            T: Debug,
            C: StaticDebug,
        {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
                Constrained::fmt(self.0, formatter)
            }
        }

        Formatted(self)
    }
}

impl<T, C> Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    /// Constructs a proxy from a primitive IEEE 754 floating-point value.
    ///
    /// This function returns the output type of the [divergence] of the proxy and invokes its
    /// error behavior if the floating-point value does not satisfy constraints. Note that this
    /// function never fails for [`Total`], which has no constraints.
    ///
    /// The distinctions in output and behavior are static and are determined by the type
    /// parameters of the `Constrained` type constructor.
    ///
    /// # Panics
    ///
    /// This function panics if the primitive floating-point value does not satisfy the constraints
    /// of the proxy **and** the [divergence] of the proxy panics. For example, the [`OrPanic`]
    /// divergence asserts constraints and panics.
    ///
    /// # Errors
    ///
    /// Returns an error if the primitive floating-point value does not satisfy the constraints of
    /// the proxy **and** the [divergence] of the proxy encodes errors in its output type. For
    /// example, the output type of the `OrError<AsExpression>` divergence is [`Expression`] and
    /// this function returns the [`Undefined`] variant if the constraint is violated.
    ///
    /// # Examples
    ///
    /// Fallibly constructing proxies from primitive floating-point values:
    ///
    /// ```rust
    /// use decorum::constraint::IsReal;
    /// use decorum::divergence::{AsResult, OrError};
    /// use decorum::proxy::Constrained;
    ///
    /// // The output type of `Real` is `Result`.
    /// type Real = Constrained<f64, IsReal<OrError<AsResult>>>;
    ///
    /// let x = Real::new(2.0).unwrap(); // The output type of `new` is `Result` per `TryResult`.
    /// ```
    ///
    /// Asserting proxy construction from primitive floating-point values:
    ///
    /// ```rust,should_panic
    /// use decorum::constraint::IsReal;
    /// use decorum::divergence::OrPanic;
    /// use decorum::proxy::Constrained;
    ///
    /// // The output type of `OrPanic` is `Real`.
    /// type Real = Constrained<f64, IsReal<OrPanic>>;
    ///
    /// let x = Real::new(2.0); // The output type of `new` is `Real` per `OrPanic`.
    /// let y = Real::new(0.0 / 0.0); // Panics.
    /// ```
    ///
    /// [`divergence`]: crate::divergence
    /// [`Expression`]: crate::expression::Expression
    /// [`OrPanic`]: crate::divergence::OrPanic
    /// [`Total`]: crate::Total
    /// [`Undefined`]: crate::expression::Expression::Undefined
    pub fn new(inner: T) -> OutputOf<Self> {
        C::map(inner, |inner| Constrained {
            inner,
            phantom: PhantomData,
        })
    }

    /// Fallibly constructs a proxy from a primitive IEEE 754 floating-point value.
    ///
    /// This construction mirrors the [`TryFrom`] implementation and is independent of the
    /// [divergence] of the proxy; it always outputs a [`Result`] and never panics.
    ///
    /// # Errors
    ///
    /// Returns an error if the primitive floating-point value does not satisfy the constraints of
    /// the proxy. Note that the error type of the [`IsFloat`] constraint is [`Infallible`] and the
    /// construction of [`Total`]s cannot fail here.
    ///
    /// # Examples
    ///
    /// Constructing proxies from primitive floating-point values:
    ///
    /// ```rust
    /// use decorum::constraint::IsReal;
    /// use decorum::divergence::OrPanic;
    /// use decorum::proxy::Constrained;
    ///
    /// type Real = Constrained<f64, IsReal<OrPanic>>;
    ///
    /// fn f(x: Real) -> Real {
    ///     x * 2.0
    /// }
    ///
    /// let y = f(Real::try_new(2.0).unwrap());
    /// // The `TryFrom` and `TryInto` traits can also be used.
    /// let z = f(2.0.try_into().unwrap());
    /// ```
    ///
    /// A proxy construction that fails:
    ///
    /// ```rust,should_panic
    /// use decorum::constraint::IsReal;
    /// use decorum::divergence::OrPanic;
    /// use decorum::proxy::Constrained;
    ///
    /// type Real = Constrained<f64, IsReal<OrPanic>>;
    ///
    /// // `IsReal` does not allow `NaN`s, but `0.0 / 0.0` produces a `NaN`.
    /// let x = Real::try_new(0.0 / 0.0).unwrap(); // Panics when unwrapping.
    /// ```
    ///
    /// [`divergence`]: crate::divergence
    /// [`Infallible`]: core::convert::Infallible
    /// [`IsFloat`]: crate::constraint::IsFloat
    /// [`Result`]: core::result::Result
    /// [`Total`]: crate::Total
    pub fn try_new(inner: T) -> Result<Self, C::Error> {
        C::check(inner).map(|_| Constrained {
            inner,
            phantom: PhantomData,
        })
    }

    /// Constructs a proxy from a primitive IEEE 754 floating-point value and asserts that its
    /// constraints are satisfied.
    ///
    /// This construction is independent of the [divergence] of the proxy and always asserts
    /// constraints (even when the divergence is fallible). Note that this function never fails
    /// (panics) for [`Total`], which has no constraints.
    ///
    /// # Panics
    ///
    /// **This construction panics if the primitive floating-point value does not satisfy the
    /// constraints of the proxy.**
    ///
    /// # Examples
    ///
    /// Constructing proxies from primitive floating-point values:
    ///
    /// ```rust
    /// use decorum::constraint::IsReal;
    /// use decorum::divergence::OrPanic;
    /// use decorum::proxy::Constrained;
    ///
    /// type Real = Constrained<f64, IsReal<OrPanic>>;
    ///
    /// fn f(x: Real) -> Real {
    ///     x * 2.0
    /// }
    ///
    /// let y = f(Real::assert(2.0));
    /// ```
    ///
    /// A proxy construction that fails:
    ///
    /// ```rust,should_panic
    /// use decorum::constraint::IsReal;
    /// use decorum::divergence::OrPanic;
    /// use decorum::proxy::Constrained;
    ///
    /// type Real = Constrained<f64, IsReal<OrPanic>>;
    ///
    /// // `IsReal` does not allow `NaN`s, but `0.0 / 0.0` produces a `NaN`.
    /// let x = Real::assert(0.0 / 0.0); // Panics.
    /// ```
    ///
    /// [`divergence`]: crate::divergence
    /// [`Total`]: crate::Total
    pub fn assert(inner: T) -> Self {
        Self::try_new(inner).expect_constrained()
    }

    /// Converts a slice of primitive IEEE 754 floating-point values into a slice of proxies.
    ///
    /// This conversion must check the constraints of the proxy against each floating-point value
    /// and so has `O(N)` time complexity. **When using the [`IsFloat`] constraint, prefer the
    /// infallible and `O(1)` [`from_slice`] function.**
    ///
    /// # Errors
    ///
    /// Returns an error if any of the primitive floating-point values in the slice do not satisfy
    /// the constraints of the proxy.
    ///
    /// [`from_slice`]: crate::Total::from_slice
    /// [`IsFloat`]: crate::constraint::IsFloat
    pub fn try_from_slice<'a>(slice: &'a [T]) -> Result<&'a [Self], C::Error> {
        slice.iter().try_for_each(|inner| C::check(*inner))?;
        // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary representation
        //         as its input type `T`. This means that it is safe to transmute `T` to
        //         `Constrained<T>`.
        Ok(unsafe { mem::transmute::<&'a [T], &'a [Self]>(slice) })
    }

    /// Converts a mutable slice of primitive IEEE 754 floating-point values into a mutable slice
    /// of proxies.
    ///
    /// This conversion must check the constraints of the proxy against each floating-point value
    /// and so has `O(N)` time complexity. **When using the [`IsFloat`] constraint, prefer the
    /// infallible and `O(1)` [`from_mut_slice`] function.**
    ///
    /// # Errors
    ///
    /// Returns an error if any of the primitive floating-point values in the
    /// slice do not satisfy the constraints of the proxy.
    ///
    /// [`from_mut_slice`]: crate::Total::from_mut_slice
    /// [`IsFloat`]: crate::constraint::IsFloat
    pub fn try_from_mut_slice<'a>(slice: &'a mut [T]) -> Result<&'a mut [Self], C::Error> {
        slice.iter().try_for_each(|inner| C::check(*inner))?;
        // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary representation
        //         as its input type `T`. This means that it is safe to transmute `T` to
        //         `Constrained<T>`.
        Ok(unsafe { mem::transmute::<&'a mut [T], &'a mut [Self]>(slice) })
    }

    /// Converts a proxy into another proxy that is capable of representing a superset of its
    /// values per its constraint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use decorum::divergence::OrPanic;
    /// use decorum::real::UnaryRealFunction;
    /// use decorum::{E64, R64};
    ///
    /// let x = R64::<OrPanic>::ZERO;
    /// let y = E64::from_subset(x); // `E64` allows a superset of the values of `R64`.
    /// ```
    pub fn from_subset<C2>(other: Constrained<T, C2>) -> Self
    where
        C2: Constraint + SubsetOf<C>,
    {
        Self::unchecked(other.into_inner())
    }

    /// Converts a proxy into another proxy that is capable of representing a superset of its
    /// values per its constraint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use decorum::real::UnaryRealFunction;
    /// use decorum::{E64, R64};
    ///
    /// let x = R64::ZERO;
    /// let y: E64 = x.into_superset(); // `E64` allows a superset of the values of `R64`.
    /// ```
    pub fn into_superset<C2>(self) -> Constrained<T, C2>
    where
        C2: Constraint + SupersetOf<C>,
    {
        Constrained::unchecked(self.into_inner())
    }

    /// Converts a proxy into its corresponding [`Expression`].
    ///
    /// The output of this function is always the [`Defined`] variant.
    ///
    /// [`Defined`]: crate::expression::Expression::Defined
    /// [`Expression`]: crate::expression::Expression
    pub fn into_expression(self) -> ExpressionOf<Self> {
        Expression::from(self)
    }

    pub(crate) fn map<F>(self, f: F) -> OutputOf<Self>
    where
        F: FnOnce(T) -> T,
    {
        Self::new(f(self.into_inner()))
    }

    pub(crate) fn zip_map<C2, F>(self, other: Constrained<T, C2>, f: F) -> OutputOf<Self>
    where
        C2: Constraint,
        F: FnOnce(T, T) -> T,
    {
        Self::new(f(self.into_inner(), other.into_inner()))
    }
}

impl<T> Total<T>
where
    T: Primitive,
{
    /// Converts a slice of primitive IEEE 754 floating-point values into a slice of `Total`s.
    ///
    /// Unlike [`try_from_slice`], this conversion is infallible and trivial and so has `O(1)` time
    /// complexity.
    ///
    /// [`try_from_slice`]: crate::proxy::Constrained::try_from_slice
    pub fn from_slice<'a>(slice: &'a [T]) -> &'a [Self] {
        // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary representation
        //         as its input type `T`. This means that it is safe to transmute `T` to
        //         `Constrained<T>`.
        unsafe { mem::transmute::<&'a [T], &'a [Self]>(slice) }
    }

    /// Converts a mutable slice of primitive floating-point values into a mutable slice of
    /// `Total`s.
    ///
    /// Unlike [`try_from_mut_slice`], this conversion is infallible and trivial and so has `O(1)`
    /// time complexity.
    ///
    /// [`try_from_mut_slice`]: crate::proxy::Constrained::try_from_mut_slice
    pub fn from_mut_slice<'a>(slice: &'a mut [T]) -> &'a mut [Self] {
        // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary representation
        //         as its input type `T`. This means that it is safe to transmute `T` to
        //         `Constrained<T>`.
        unsafe { mem::transmute::<&'a mut [T], &'a mut [Self]>(slice) }
    }
}

#[cfg(feature = "approx")]
impl<T, C> AbsDiffEq for Constrained<T, C>
where
    T: AbsDiffEq<Epsilon = T> + Primitive,
    C: Constraint,
{
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::assert(T::default_epsilon())
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.into_inner()
            .abs_diff_eq(&other.into_inner(), epsilon.into_inner())
    }
}

impl<T, C> Add for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn add(self, other: Self) -> Self::Output {
        self.zip_map(other, Add::add)
    }
}

impl<T, C> Add<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn add(self, other: T) -> Self::Output {
        self.map(|inner| inner + other)
    }
}

impl<T, C, E> AddAssign for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<T, C, E> AddAssign<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn add_assign(&mut self, other: T) {
        *self = self.map(|inner| inner + other);
    }
}

impl<T, C> AsRef<T> for Constrained<T, C> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T, C> BinaryRealFunction for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Self) -> Self::Codomain {
        self.zip_map(n, BinaryRealFunction::div_euclid)
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Self) -> Self::Codomain {
        self.zip_map(n, BinaryRealFunction::rem_euclid)
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Self) -> Self::Codomain {
        self.zip_map(n, BinaryRealFunction::pow)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self::Codomain {
        self.zip_map(base, BinaryRealFunction::log)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self::Codomain {
        self.zip_map(other, BinaryRealFunction::hypot)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self::Codomain {
        self.zip_map(other, BinaryRealFunction::atan2)
    }
}

impl<T, C> BinaryRealFunction<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: T) -> Self::Codomain {
        self.map(|inner| BinaryRealFunction::div_euclid(inner, n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: T) -> Self::Codomain {
        self.map(|inner| BinaryRealFunction::rem_euclid(inner, n))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: T) -> Self::Codomain {
        self.map(|inner| BinaryRealFunction::pow(inner, n))
    }

    #[cfg(feature = "std")]
    fn log(self, base: T) -> Self::Codomain {
        self.map(|inner| BinaryRealFunction::log(inner, base))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: T) -> Self::Codomain {
        self.map(|inner| BinaryRealFunction::hypot(inner, other))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: T) -> Self::Codomain {
        self.map(|inner| BinaryRealFunction::atan2(inner, other))
    }
}

impl<T, C> Bounded for Constrained<T, C>
where
    T: Primitive,
{
    fn min_value() -> Self {
        BaseEncoding::MIN_FINITE
    }

    fn max_value() -> Self {
        BaseEncoding::MAX_FINITE
    }
}

impl<T, C> Clone for Constrained<T, C>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Constrained {
            inner: self.inner.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T, C> ConstrainedProxy for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Constraint = C;
}

impl<T, C> Function for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Codomain = OutputOf<Self>;
}

impl<T, C> Copy for Constrained<T, C> where T: Copy {}

impl<T, D> Debug for Constrained<T, IsExtendedReal<D>>
where
    T: Debug,
    D: Divergence,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ExtendedReal")
            .field(self.as_ref())
            .finish()
    }
}

impl<T> Debug for Constrained<T, IsFloat>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Total").field(self.as_ref()).finish()
    }
}

impl<T, D> Debug for Constrained<T, IsReal<D>>
where
    T: Debug,
    D: Divergence,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Real").field(self.as_ref()).finish()
    }
}

impl<T, C> Default for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    fn default() -> Self {
        // There is no constraint that disallows real numbers such as zero.
        Self::unchecked(T::ZERO)
    }
}

impl<T, C> Display for Constrained<T, C>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, C> Div for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn div(self, other: Self) -> Self::Output {
        self.zip_map(other, Div::div)
    }
}

impl<T, C> Div<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn div(self, other: T) -> Self::Output {
        self.map(|inner| inner / other)
    }
}

impl<T, C, E> DivAssign for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn div_assign(&mut self, other: Self) {
        *self = *self / other
    }
}

impl<T, C, E> DivAssign<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn div_assign(&mut self, other: T) {
        *self = self.map(|inner| inner / other);
    }
}

impl<T, C> BaseEncoding for Constrained<T, C>
where
    T: Primitive,
{
    const MAX_FINITE: Self = Constrained::unchecked(T::MAX_FINITE);
    const MIN_FINITE: Self = Constrained::unchecked(T::MIN_FINITE);
    const MIN_POSITIVE_NORMAL: Self = Constrained::unchecked(T::MIN_POSITIVE_NORMAL);
    const EPSILON: Self = Constrained::unchecked(T::EPSILON);

    fn classify(self) -> FpCategory {
        T::classify(self.into_inner())
    }

    fn is_normal(self) -> bool {
        T::is_normal(self.into_inner())
    }

    fn is_sign_positive(self) -> bool {
        self.into_inner().is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.into_inner().is_sign_negative()
    }

    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        self.map_unchecked(|inner| inner.signum())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        T::integer_decode(self.into_inner())
    }
}

impl<T, C> Eq for Constrained<T, C> where T: Primitive {}

impl<T, C, E> Float for Constrained<T, C>
where
    T: Float + Primitive,
    C: Constraint<Error = E> + Member<InfinitySet> + Member<NanSet>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn infinity() -> Self {
        InfinityEncoding::INFINITY
    }

    fn neg_infinity() -> Self {
        InfinityEncoding::NEG_INFINITY
    }

    fn is_infinite(self) -> bool {
        self.with_inner(Float::is_infinite)
    }

    fn is_finite(self) -> bool {
        self.with_inner(Float::is_finite)
    }

    fn nan() -> Self {
        <Self as NanEncoding>::NAN
    }

    fn is_nan(self) -> bool {
        self.with_inner(Float::is_nan)
    }

    fn max_value() -> Self {
        BaseEncoding::MAX_FINITE
    }

    fn min_value() -> Self {
        BaseEncoding::MIN_FINITE
    }

    fn min_positive_value() -> Self {
        BaseEncoding::MIN_POSITIVE_NORMAL
    }

    fn epsilon() -> Self {
        BaseEncoding::EPSILON
    }

    fn min(self, other: Self) -> Self {
        self.zip_map(other, Float::min)
    }

    fn max(self, other: Self) -> Self {
        self.zip_map(other, Float::max)
    }

    fn neg_zero() -> Self {
        -Self::ZERO
    }

    fn is_sign_positive(self) -> bool {
        self.with_inner(Float::is_sign_positive)
    }

    fn is_sign_negative(self) -> bool {
        self.with_inner(Float::is_sign_negative)
    }

    fn signum(self) -> Self {
        self.map(Float::signum)
    }

    fn abs(self) -> Self {
        self.map(Float::abs)
    }

    fn classify(self) -> FpCategory {
        self.with_inner(Float::classify)
    }

    fn is_normal(self) -> bool {
        self.with_inner(Float::is_normal)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.with_inner(Float::integer_decode)
    }

    fn floor(self) -> Self {
        self.map(Float::floor)
    }

    fn ceil(self) -> Self {
        self.map(Float::ceil)
    }

    fn round(self) -> Self {
        self.map(Float::round)
    }

    fn trunc(self) -> Self {
        self.map(Float::trunc)
    }

    fn fract(self) -> Self {
        self.map(Float::fract)
    }

    fn recip(self) -> Self {
        self.map(Float::recip)
    }

    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let a = a.into_inner();
        let b = b.into_inner();
        // TODO: This implementation requires a `Float` bound and forwards to its `mul_add`.
        //       Consider supporting `mul_add` via a trait that is more specific to floating-point
        //       encoding than `BinaryRealFunction` and friends.
        self.map(|inner| Float::mul_add(inner, a, b))
    }

    #[cfg(feature = "std")]
    fn abs_sub(self, other: Self) -> Self {
        self.zip_map(other, Float::abs_sub)
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self {
        self.map(|inner| Float::powi(inner, n))
    }

    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self {
        self.zip_map(n, Float::powf)
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self {
        self.map(Float::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map(Float::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self {
        self.map(Float::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self {
        self.map(Float::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self {
        self.map(Float::exp_m1)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self {
        self.zip_map(base, Float::log)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self {
        self.map(Float::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self {
        self.map(Float::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self {
        self.map(Float::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self {
        self.map(Float::ln_1p)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self {
        BinaryRealFunction::hypot(self, other)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map(Float::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map(Float::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self {
        self.map(Float::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self {
        self.map(Float::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self {
        self.map(Float::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map(Float::atan)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self {
        BinaryRealFunction::atan2(self, other)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = Float::sin_cos(self.into_inner());
        (Constrained::<_, C>::new(sin), Constrained::<_, C>::new(cos))
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        self.map(Float::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map(Float::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map(Float::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self {
        self.map(Float::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self {
        self.map(Float::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self {
        self.map(Float::atanh)
    }

    #[cfg(not(feature = "std"))]
    fn to_degrees(self) -> Self {
        self.map(Float::to_degrees)
    }

    #[cfg(not(feature = "std"))]
    fn to_radians(self) -> Self {
        self.map(Float::to_radians)
    }
}

impl<T, C> FloatConst for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    fn E() -> Self {
        <Self as UnaryRealFunction>::E
    }

    fn PI() -> Self {
        <Self as UnaryRealFunction>::PI
    }

    fn SQRT_2() -> Self {
        <Self as UnaryRealFunction>::SQRT_2
    }

    fn FRAC_1_PI() -> Self {
        <Self as UnaryRealFunction>::FRAC_1_PI
    }

    fn FRAC_2_PI() -> Self {
        <Self as UnaryRealFunction>::FRAC_2_PI
    }

    fn FRAC_1_SQRT_2() -> Self {
        <Self as UnaryRealFunction>::FRAC_1_SQRT_2
    }

    fn FRAC_2_SQRT_PI() -> Self {
        <Self as UnaryRealFunction>::FRAC_2_SQRT_PI
    }

    fn FRAC_PI_2() -> Self {
        <Self as UnaryRealFunction>::FRAC_PI_2
    }

    fn FRAC_PI_3() -> Self {
        <Self as UnaryRealFunction>::FRAC_PI_3
    }

    fn FRAC_PI_4() -> Self {
        <Self as UnaryRealFunction>::FRAC_PI_4
    }

    fn FRAC_PI_6() -> Self {
        <Self as UnaryRealFunction>::FRAC_PI_6
    }

    fn FRAC_PI_8() -> Self {
        <Self as UnaryRealFunction>::FRAC_PI_8
    }

    fn LN_10() -> Self {
        <Self as UnaryRealFunction>::LN_10
    }

    fn LN_2() -> Self {
        <Self as UnaryRealFunction>::LN_2
    }

    fn LOG10_E() -> Self {
        <Self as UnaryRealFunction>::LOG10_E
    }

    fn LOG2_E() -> Self {
        <Self as UnaryRealFunction>::LOG2_E
    }
}

impl<T> From<Real<T>> for ExtendedReal<T>
where
    T: Primitive,
{
    fn from(other: Real<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<'a, T> From<&'a T> for &'a Total<T>
where
    T: Primitive,
{
    fn from(inner: &'a T) -> Self {
        // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary representation
        //         as its input type `T`. This means that it is safe to transmute `T` to
        //         `Constrained<T>`.
        unsafe { &*(inner as *const T as *const Total<T>) }
    }
}

impl<'a, T> From<&'a mut T> for &'a mut Total<T>
where
    T: Primitive,
{
    fn from(inner: &'a mut T) -> Self {
        // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary representation
        //         as its input type `T`. This means that it is safe to transmute `T` to
        //         `Constrained<T>`.
        unsafe { &mut *(inner as *mut T as *mut Total<T>) }
    }
}

impl<T> From<Real<T>> for Total<T>
where
    T: Primitive,
{
    fn from(other: Real<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<T> From<ExtendedReal<T>> for Total<T>
where
    T: Primitive,
{
    fn from(other: ExtendedReal<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<C> From<Constrained<f32, C>> for f32 {
    fn from(proxy: Constrained<f32, C>) -> Self {
        proxy.into_inner()
    }
}

impl<C> From<Constrained<f64, C>> for f64 {
    fn from(proxy: Constrained<f64, C>) -> Self {
        proxy.into_inner()
    }
}

#[cfg(feature = "serde")]
impl<T, C> From<Constrained<T, C>> for Serde<T>
where
    T: Copy,
{
    fn from(proxy: Constrained<T, C>) -> Self {
        Serde {
            inner: proxy.into_inner(),
        }
    }
}

impl<T> From<T> for Total<T>
where
    T: Primitive,
{
    fn from(inner: T) -> Self {
        Self::unchecked(inner)
    }
}

impl<T, C> FromPrimitive for Constrained<T, C>
where
    T: FromPrimitive + Primitive,
    C: Constraint,
{
    fn from_i8(value: i8) -> Option<Self> {
        T::from_i8(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_u8(value: u8) -> Option<Self> {
        T::from_u8(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_i16(value: i16) -> Option<Self> {
        T::from_i16(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_u16(value: u16) -> Option<Self> {
        T::from_u16(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_i32(value: i32) -> Option<Self> {
        T::from_i32(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_u32(value: u32) -> Option<Self> {
        T::from_u32(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_i64(value: i64) -> Option<Self> {
        T::from_i64(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_u64(value: u64) -> Option<Self> {
        T::from_u64(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_isize(value: isize) -> Option<Self> {
        T::from_isize(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_usize(value: usize) -> Option<Self> {
        T::from_usize(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_f32(value: f32) -> Option<Self> {
        T::from_f32(value).and_then(|inner| Constrained::try_new(inner).ok())
    }

    fn from_f64(value: f64) -> Option<Self> {
        T::from_f64(value).and_then(|inner| Constrained::try_new(inner).ok())
    }
}

impl<T, C, E> FromStr for Constrained<T, C>
where
    T: FromStr + Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    type Err = <T as FromStr>::Err;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        T::from_str(string).map(Self::new)
    }
}

impl<T, C> Hash for Constrained<T, C>
where
    T: Primitive + ToCanonical,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.hash_canonical(state)
    }
}

impl<T, C> InfinityEncoding for Constrained<T, C>
where
    T: Primitive,
    C: Constraint + Member<InfinitySet>,
{
    const INFINITY: Self = Constrained::unchecked(T::INFINITY);
    const NEG_INFINITY: Self = Constrained::unchecked(T::NEG_INFINITY);

    fn is_infinite(self) -> bool {
        self.into_inner().is_infinite()
    }

    fn is_finite(self) -> bool {
        self.into_inner().is_finite()
    }
}

impl<T, C> IntrinsicOrd for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Undefined = C::Undefined<T>;

    #[inline(always)]
    fn from_undefined(undefined: Self::Undefined) -> Self {
        C::from_undefined(undefined)
    }

    #[inline(always)]
    fn is_undefined(&self) -> bool {
        C::is_undefined(self.as_ref())
    }

    fn intrinsic_cmp(&self, other: &Self) -> Result<Ordering, Self::Undefined> {
        match self.as_ref().intrinsic_cmp(other.as_ref()) {
            Ok(ordering) => Ok(ordering),
            Err(_) => Err(C::undefined()),
        }
    }
}

impl<T> IntrinsicUndefined for Total<T>
where
    T: Primitive,
{
    fn undefined() -> Self {
        Total::NAN
    }
}

impl<T, C> LowerExp for Constrained<T, C>
where
    T: LowerExp + Primitive,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, C> Mul for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn mul(self, other: Self) -> Self::Output {
        self.zip_map(other, Mul::mul)
    }
}

impl<T, C> Mul<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn mul(self, other: T) -> Self::Output {
        self.map(|a| a * other)
    }
}

impl<T, C, E> MulAssign for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<T, C, E> MulAssign<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T, C> NanEncoding for Constrained<T, C>
where
    T: Primitive,
    C: Constraint + Member<NanSet>,
{
    type Nan = Self;

    const NAN: Self::Nan = Constrained::unchecked(T::NAN.into_inner());

    fn is_nan(self) -> bool {
        self.into_inner().is_nan()
    }
}

impl<T, C> Neg for Constrained<T, C>
where
    T: Primitive,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        // There is no constraint for which negating a value produces an invalid value.
        Constrained::unchecked(-self.into_inner())
    }
}

impl<T, C, E> Num for Constrained<T, C>
where
    T: Num + Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    // TODO: Differentiate between parse and contraint errors.
    type FromStrRadixErr = ();

    fn from_str_radix(source: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(source, radix)
            .map_err(|_| ())
            .and_then(|inner| Constrained::try_new(inner).map_err(|_| ()))
    }
}

impl<T, C> NumCast for Constrained<T, C>
where
    T: NumCast + Primitive + ToPrimitive,
    C: Constraint,
{
    fn from<U>(value: U) -> Option<Self>
    where
        U: ToPrimitive,
    {
        T::from(value).and_then(|inner| Constrained::try_new(inner).ok())
    }
}

impl<T, C, E> One for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn one() -> Self {
        UnaryRealFunction::ONE
    }
}

impl<T, C> Ord for Constrained<T, C>
where
    T: Primitive,
{
    fn cmp(&self, other: &Self) -> Ordering {
        CanonicalOrd::cmp_canonical(self.as_ref(), other.as_ref())
    }
}

impl<T, C> PartialEq for Constrained<T, C>
where
    T: Primitive,
{
    fn eq(&self, other: &Self) -> bool {
        self.eq_canonical(other)
    }
}

impl<T, C> PartialEq<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    fn eq(&self, other: &T) -> bool {
        if let Ok(other) = Self::try_new(*other) {
            Self::eq(self, &other)
        }
        else {
            false
        }
    }
}

impl<T, C> PartialOrd for Constrained<T, C>
where
    T: Primitive,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, C> PartialOrd<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Self::try_new(*other)
            .ok()
            .and_then(|other| Self::partial_cmp(self, &other))
    }
}

impl<T, C, E> Product for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn product<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(UnaryRealFunction::ONE, |a, b| a * b)
    }
}

impl<T, C> Proxy for Constrained<T, C>
where
    T: Primitive,
{
    type Primitive = T;
}

#[cfg(feature = "approx")]
impl<T, C> RelativeEq for Constrained<T, C>
where
    T: Primitive + RelativeEq<Epsilon = T>,
    C: Constraint,
{
    fn default_max_relative() -> Self::Epsilon {
        Self::assert(T::default_max_relative())
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.into_inner().relative_eq(
            &other.into_inner(),
            epsilon.into_inner(),
            max_relative.into_inner(),
        )
    }
}

impl<T, C> Rem for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn rem(self, other: Self) -> Self::Output {
        self.zip_map(other, Rem::rem)
    }
}

impl<T, C> Rem<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn rem(self, other: T) -> Self::Output {
        self.map(|inner| inner % other)
    }
}

impl<T, C, E> RemAssign for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    }
}

impl<T, C, E> RemAssign<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn rem_assign(&mut self, other: T) {
        *self = self.map(|inner| inner % other);
    }
}

impl<T, C, E> Signed for Constrained<T, C>
where
    T: Primitive + Signed,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn abs(&self) -> Self {
        self.map_unchecked(|inner| Signed::abs(&inner))
    }

    fn abs_sub(&self, other: &Self) -> Self {
        self.zip_map(*other, |a, b| Signed::abs_sub(&a, &b))
    }

    fn signum(&self) -> Self {
        self.map_unchecked(|inner| Signed::signum(&inner))
    }

    fn is_positive(&self) -> bool {
        Signed::is_positive(self.as_ref())
    }

    fn is_negative(&self) -> bool {
        Signed::is_negative(self.as_ref())
    }
}

impl<T, C> Sub for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn sub(self, other: Self) -> Self::Output {
        self.zip_map(other, Sub::sub)
    }
}

impl<T, C> Sub<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Output = OutputOf<Self>;

    fn sub(self, other: T) -> Self::Output {
        self.map(|inner| inner - other)
    }
}

impl<T, C, E> SubAssign for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other
    }
}

impl<T, C, E> SubAssign<T> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn sub_assign(&mut self, other: T) {
        *self = self.map(|inner| inner - other)
    }
}

impl<T, C, E> Sum for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn sum<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(UnaryRealFunction::ZERO, |a, b| a + b)
    }
}

impl<T, C> ToCanonical for Constrained<T, C>
where
    T: Primitive,
{
    type Canonical = <T as ToCanonical>::Canonical;

    fn to_canonical(self) -> Self::Canonical {
        self.inner.to_canonical()
    }
}

impl<T, C> ToPrimitive for Constrained<T, C>
where
    T: Primitive + ToPrimitive,
{
    fn to_i8(&self) -> Option<i8> {
        self.as_ref().to_i8()
    }

    fn to_u8(&self) -> Option<u8> {
        self.as_ref().to_u8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.as_ref().to_i16()
    }

    fn to_u16(&self) -> Option<u16> {
        self.as_ref().to_u16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.as_ref().to_i32()
    }

    fn to_u32(&self) -> Option<u32> {
        self.as_ref().to_u32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.as_ref().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.as_ref().to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self.as_ref().to_isize()
    }

    fn to_usize(&self) -> Option<usize> {
        self.as_ref().to_usize()
    }

    fn to_f32(&self) -> Option<f32> {
        self.as_ref().to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.as_ref().to_f64()
    }
}

#[cfg(feature = "serde")]
impl<T, C> TryFrom<Serde<T>> for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    type Error = C::Error;

    fn try_from(container: Serde<T>) -> Result<Self, Self::Error> {
        Self::try_new(container.inner)
    }
}

#[cfg(feature = "approx")]
impl<T, C> UlpsEq for Constrained<T, C>
where
    T: Primitive + UlpsEq<Epsilon = T>,
    C: Constraint,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.into_inner()
            .ulps_eq(&other.into_inner(), epsilon.into_inner(), max_ulps)
    }
}

impl<T, C> UnaryRealFunction for Constrained<T, C>
where
    T: Primitive,
    C: Constraint,
{
    const ZERO: Self = Constrained::unchecked(UnaryRealFunction::ZERO);
    const ONE: Self = Constrained::unchecked(UnaryRealFunction::ONE);
    const E: Self = Constrained::unchecked(UnaryRealFunction::E);
    const PI: Self = Constrained::unchecked(UnaryRealFunction::PI);
    const FRAC_1_PI: Self = Constrained::unchecked(UnaryRealFunction::FRAC_1_PI);
    const FRAC_2_PI: Self = Constrained::unchecked(UnaryRealFunction::FRAC_2_PI);
    const FRAC_2_SQRT_PI: Self = Constrained::unchecked(UnaryRealFunction::FRAC_2_SQRT_PI);
    const FRAC_PI_2: Self = Constrained::unchecked(UnaryRealFunction::FRAC_PI_2);
    const FRAC_PI_3: Self = Constrained::unchecked(UnaryRealFunction::FRAC_PI_3);
    const FRAC_PI_4: Self = Constrained::unchecked(UnaryRealFunction::FRAC_PI_4);
    const FRAC_PI_6: Self = Constrained::unchecked(UnaryRealFunction::FRAC_PI_6);
    const FRAC_PI_8: Self = Constrained::unchecked(UnaryRealFunction::FRAC_PI_8);
    const SQRT_2: Self = Constrained::unchecked(UnaryRealFunction::SQRT_2);
    const FRAC_1_SQRT_2: Self = Constrained::unchecked(UnaryRealFunction::FRAC_1_SQRT_2);
    const LN_2: Self = Constrained::unchecked(UnaryRealFunction::LN_2);
    const LN_10: Self = Constrained::unchecked(UnaryRealFunction::LN_10);
    const LOG2_E: Self = Constrained::unchecked(UnaryRealFunction::LOG2_E);
    const LOG10_E: Self = Constrained::unchecked(UnaryRealFunction::LOG10_E);

    fn is_zero(self) -> bool {
        self.into_inner().is_zero()
    }

    fn is_one(self) -> bool {
        self.into_inner().is_zero()
    }

    fn sign(self) -> Sign {
        self.with_inner(|inner| inner.sign())
    }

    #[cfg(feature = "std")]
    fn abs(self) -> Self {
        self.map_unchecked(UnaryRealFunction::abs)
    }

    #[cfg(feature = "std")]
    fn floor(self) -> Self {
        self.map_unchecked(UnaryRealFunction::floor)
    }

    #[cfg(feature = "std")]
    fn ceil(self) -> Self {
        self.map_unchecked(UnaryRealFunction::ceil)
    }

    #[cfg(feature = "std")]
    fn round(self) -> Self {
        self.map_unchecked(UnaryRealFunction::round)
    }

    #[cfg(feature = "std")]
    fn trunc(self) -> Self {
        self.map_unchecked(UnaryRealFunction::trunc)
    }

    #[cfg(feature = "std")]
    fn fract(self) -> Self {
        self.map_unchecked(UnaryRealFunction::fract)
    }

    fn recip(self) -> Self::Codomain {
        self.map(UnaryRealFunction::recip)
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Codomain {
        self.map(|inner| UnaryRealFunction::powi(inner, n))
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Codomain {
        self.map(UnaryRealFunction::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map_unchecked(UnaryRealFunction::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self::Codomain {
        self.map(UnaryRealFunction::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Codomain {
        self.map(UnaryRealFunction::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Codomain {
        self.map(UnaryRealFunction::exp_m1)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self::Codomain {
        self.map(UnaryRealFunction::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self::Codomain {
        self.map(UnaryRealFunction::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self::Codomain {
        self.map(UnaryRealFunction::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self::Codomain {
        self.map(UnaryRealFunction::ln_1p)
    }

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Codomain {
        self.map(UnaryRealFunction::to_degrees)
    }

    #[cfg(feature = "std")]
    fn to_radians(self) -> Self {
        self.map_unchecked(UnaryRealFunction::to_radians)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map_unchecked(UnaryRealFunction::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map_unchecked(UnaryRealFunction::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self::Codomain {
        self.map(UnaryRealFunction::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self::Codomain {
        self.map(UnaryRealFunction::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self::Codomain {
        self.map(UnaryRealFunction::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map_unchecked(UnaryRealFunction::atan)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.into_inner().sin_cos();
        (Constrained::unchecked(sin), Constrained::unchecked(cos))
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        self.map_unchecked(UnaryRealFunction::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map_unchecked(UnaryRealFunction::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map_unchecked(UnaryRealFunction::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Codomain {
        self.map(UnaryRealFunction::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Codomain {
        self.map(UnaryRealFunction::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Codomain {
        self.map(UnaryRealFunction::atanh)
    }
}

impl<T, C> UpperExp for Constrained<T, C>
where
    T: UpperExp,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, C, E> Zero for Constrained<T, C>
where
    T: Primitive,
    C: Constraint<Error = E>,
    divergence::ContinueOf<C::Divergence>: NonResidual<Self, E>,
{
    fn zero() -> Self {
        UnaryRealFunction::ZERO
    }

    fn is_zero(&self) -> bool {
        self.as_ref().is_zero()
    }
}

macro_rules! impl_binary_operation_for_proxy {
    () => {
        with_binary_operations!(impl_binary_operation_for_proxy);
    };
    (operation => $trait:ident :: $method:ident) => {
        impl_binary_operation_for_proxy!(operation => $trait :: $method, |left, right| {
            right.map(|inner| $trait::$method(left, inner))
        });
    };
    (operation => $trait:ident :: $method:ident, |$left:ident, $right:ident| $f:block) => {
        macro_rules! impl_primitive_binary_operation_for_proxy {
            (primitive => $t:ty) => {
                impl<C> $trait<Constrained<$t, C>> for $t
                where
                    C: Constraint,
                {
                    type Output = OutputOf<Constrained<$t, C>>;

                    fn $method(self, other: Constrained<$t, C>) -> Self::Output {
                        let $left = self;
                        let $right = other;
                        $f
                    }
                }
            };
        }
        with_primitives!(impl_primitive_binary_operation_for_proxy);
    };
}
impl_binary_operation_for_proxy!();

/// Implements the `Real` trait from [`num-traits`](https://crates.io/crates/num-traits) for
/// non-`NaN` proxy types. Does nothing if the `std` feature is disabled.
///
/// A blanket implementation is not possible, because it conflicts with a very general blanket
/// implementation provided by [`num-traits`]. See the following issues:
///
/// - https://github.com/olson-sean-k/decorum/issues/10
/// - https://github.com/rust-num/num-traits/issues/49
macro_rules! impl_num_traits_real_for_proxy {
    () => {
        with_primitives!(impl_num_traits_real_for_proxy);
    };
    (primitive => $t:ty) => {
        impl_num_traits_real_for_proxy!(proxy => Real, primitive => $t);
        impl_num_traits_real_for_proxy!(proxy => ExtendedReal, primitive => $t);
    };
    (proxy => $p:ident, primitive => $t:ty) => {
        #[cfg(feature = "std")]
        impl<D> num_traits::real::Real for $p<$t, D>
        where
            D: Divergence,
            divergence::ContinueOf<D>: NonResidual<Self, ErrorOf<Self>>,
        {
            fn max_value() -> Self {
                BaseEncoding::MAX_FINITE
            }

            fn min_value() -> Self {
                BaseEncoding::MIN_FINITE
            }

            fn min_positive_value() -> Self {
                BaseEncoding::MIN_POSITIVE_NORMAL
            }

            fn epsilon() -> Self {
                BaseEncoding::EPSILON
            }

            fn min(self, other: Self) -> Self {
                self.zip_map(other, num_traits::real::Real::min)
            }

            fn max(self, other: Self) -> Self {
                self.zip_map(other, num_traits::real::Real::max)
            }

            fn is_sign_positive(self) -> bool {
                self.with_inner(num_traits::real::Real::is_sign_positive)
            }

            fn is_sign_negative(self) -> bool {
                self.with_inner(num_traits::real::Real::is_sign_negative)
            }

            fn signum(self) -> Self {
                self.map(num_traits::real::Real::signum)
            }

            fn abs(self) -> Self {
                self.map(num_traits::real::Real::abs)
            }

            fn floor(self) -> Self {
                self.map(num_traits::real::Real::floor)
            }

            fn ceil(self) -> Self {
                self.map(num_traits::real::Real::ceil)
            }

            fn round(self) -> Self {
                self.map(num_traits::real::Real::round)
            }

            fn trunc(self) -> Self {
                self.map(num_traits::real::Real::trunc)
            }

            fn fract(self) -> Self {
                self.map(num_traits::real::Real::fract)
            }

            fn recip(self) -> Self {
                self.map(num_traits::real::Real::recip)
            }

            fn mul_add(self, a: Self, b: Self) -> Self {
                let a = a.into_inner();
                let b = b.into_inner();
                self.map(|inner| inner.mul_add(a, b))
            }

            fn abs_sub(self, other: Self) -> Self {
                self.zip_map(other, num_traits::real::Real::abs_sub)
            }

            fn powi(self, n: i32) -> Self {
                self.map(|inner| num_traits::real::Real::powi(inner, n))
            }

            fn powf(self, n: Self) -> Self {
                self.zip_map(n, num_traits::real::Real::powf)
            }

            fn sqrt(self) -> Self {
                self.map(num_traits::real::Real::sqrt)
            }

            fn cbrt(self) -> Self {
                self.map(num_traits::real::Real::cbrt)
            }

            fn exp(self) -> Self {
                self.map(num_traits::real::Real::exp)
            }

            fn exp2(self) -> Self {
                self.map(num_traits::real::Real::exp2)
            }

            fn exp_m1(self) -> Self {
                self.map(num_traits::real::Real::exp_m1)
            }

            fn log(self, base: Self) -> Self {
                self.zip_map(base, num_traits::real::Real::log)
            }

            fn ln(self) -> Self {
                self.map(num_traits::real::Real::ln)
            }

            fn log2(self) -> Self {
                self.map(num_traits::real::Real::log2)
            }

            fn log10(self) -> Self {
                self.map(num_traits::real::Real::log10)
            }

            fn to_degrees(self) -> Self {
                self.map(num_traits::real::Real::to_degrees)
            }

            fn to_radians(self) -> Self {
                self.map(num_traits::real::Real::to_radians)
            }

            fn ln_1p(self) -> Self {
                self.map(num_traits::real::Real::ln_1p)
            }

            fn hypot(self, other: Self) -> Self {
                self.zip_map(other, num_traits::real::Real::hypot)
            }

            fn sin(self) -> Self {
                self.map(num_traits::real::Real::sin)
            }

            fn cos(self) -> Self {
                self.map(num_traits::real::Real::cos)
            }

            fn tan(self) -> Self {
                self.map(num_traits::real::Real::tan)
            }

            fn asin(self) -> Self {
                self.map(num_traits::real::Real::asin)
            }

            fn acos(self) -> Self {
                self.map(num_traits::real::Real::acos)
            }

            fn atan(self) -> Self {
                self.map(num_traits::real::Real::atan)
            }

            fn atan2(self, other: Self) -> Self {
                self.zip_map(other, num_traits::real::Real::atan2)
            }

            fn sin_cos(self) -> (Self, Self) {
                let (sin, cos) = self.with_inner(num_traits::real::Real::sin_cos);
                ($p::<_, D>::new(sin), $p::<_, D>::new(cos))
            }

            fn sinh(self) -> Self {
                self.map(num_traits::real::Real::sinh)
            }

            fn cosh(self) -> Self {
                self.map(num_traits::real::Real::cosh)
            }

            fn tanh(self) -> Self {
                self.map(num_traits::real::Real::tanh)
            }

            fn asinh(self) -> Self {
                self.map(num_traits::real::Real::asinh)
            }

            fn acosh(self) -> Self {
                self.map(num_traits::real::Real::acosh)
            }

            fn atanh(self) -> Self {
                self.map(num_traits::real::Real::atanh)
            }
        }
    };
}
impl_num_traits_real_for_proxy!();

// `TryFrom` cannot be implemented over an open type `T` and cannot be implemented for constraints
// in general, because it would conflict with the `From` implementation for `Total`.
macro_rules! impl_try_from_for_proxy {
    () => {
        with_primitives!(impl_try_from_for_proxy);
    };
    (primitive => $t:ty) => {
        impl_try_from_for_proxy!(proxy => Real, primitive => $t);
        impl_try_from_for_proxy!(proxy => ExtendedReal, primitive => $t);
    };
    (proxy => $p:ident, primitive => $t:ty) => {
        impl<D> TryFrom<$t> for $p<$t, D>
        where
            D: Divergence,
        {
            type Error = ErrorOf<Self>;

            fn try_from(inner: $t) -> Result<Self, Self::Error> {
                Self::try_new(inner)
            }
        }

        impl<'a, D> TryFrom<&'a $t> for &'a $p<$t, D>
        where
            D: Divergence,
        {
            type Error = ErrorOf<$p<$t, D>>;

            fn try_from(inner: &'a $t) -> Result<Self, Self::Error> {
                ConstraintOf::<$p<$t, D>>::check(*inner).map(|_| {
                    // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary
                    //         representation as its input type `T`. This means that it is safe to
                    //         transmute `T` to `Constrained<T>`.
                    unsafe { mem::transmute::<&'a $t, Self>(inner) }
                })
            }
        }

        impl<'a, D> TryFrom<&'a mut $t> for &'a mut $p<$t, D>
        where
            D: Divergence,
        {
            type Error = ErrorOf<$p<$t, D>>;

            fn try_from(inner: &'a mut $t) -> Result<Self, Self::Error> {
                ConstraintOf::<$p<$t, D>>::check(*inner).map(move |_| {
                    // SAFETY: `Constrained<T>` is `repr(transparent)` and has the same binary
                    //         representation as its input type `T`. This means that it is safe to
                    //         transmute `T` to `Constrained<T>`.
                    unsafe { mem::transmute::<&'a mut $t, Self>(inner) }
                })
            }
        }
    };
}
impl_try_from_for_proxy!();

#[cfg(test)]
mod tests {
    use crate::real::RealFunction;
    use crate::{ExtendedReal, InfinityEncoding, NanEncoding, Real, Total, E32, R32};

    #[test]
    fn total_no_panic_on_inf() {
        let x: Total<f32> = 1.0.into();
        let y = x / 0.0;
        assert!(InfinityEncoding::is_infinite(y));
    }

    #[test]
    fn total_no_panic_on_nan() {
        let x: Total<f32> = 0.0.into();
        let y = x / 0.0;
        assert!(NanEncoding::is_nan(y));
    }

    // This is the most comprehensive and general test of reference conversions, as there are no
    // failure conditions. Other similar tests focus solely on success or failure, not completeness
    // of the APIs under test. This test is an ideal Miri target.
    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    fn total_no_panic_from_ref_slice() {
        let x = 0.0f64 / 0.0;
        let y: &Total<_> = (&x).into();
        assert!(y.is_nan());

        let mut x = 0.0f64;
        let y: &mut Total<_> = (&mut x).into();
        *y = (0.0f64 / 0.0).into();
        assert!(y.is_nan());

        let xs = [0.0f64, 1.0];
        let ys = Total::from_slice(&xs);
        assert_eq!(ys, &[0.0f64, 1.0]);

        let xs = [0.0f64, 1.0];
        let ys = Total::from_slice(&xs);
        assert_eq!(ys, &[0.0f64, 1.0]);
    }

    #[test]
    fn notnan_no_panic_on_inf() {
        let x: E32 = 1.0.try_into().unwrap();
        let y = x / 0.0;
        assert!(InfinityEncoding::is_infinite(y));
    }

    #[test]
    #[should_panic]
    fn notnan_panic_on_nan() {
        let x: E32 = 0.0.try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    fn notnan_no_panic_from_inf_ref_slice() {
        let x = 1.0f64 / 0.0;
        let y: &ExtendedReal<_> = (&x).try_into().unwrap();
        assert!(y.is_infinite());

        let xs = [0.0f64, 1.0 / 0.0];
        let ys = ExtendedReal::<f64>::try_from_slice(&xs).unwrap();
        assert_eq!(ys, &[0.0f64, InfinityEncoding::INFINITY]);
    }

    #[test]
    #[should_panic]
    #[allow(clippy::zero_divided_by_zero)]
    fn notnan_panic_from_nan_ref() {
        let x = 0.0f64 / 0.0;
        let _: &ExtendedReal<_> = (&x).try_into().unwrap();
    }

    #[test]
    #[should_panic]
    #[allow(clippy::zero_divided_by_zero)]
    fn notnan_panic_from_nan_slice() {
        let xs = [1.0f64, 0.0f64 / 0.0];
        let _ = ExtendedReal::<f64>::try_from_slice(&xs).unwrap();
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_nan() {
        let x: R32 = 0.0.try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_inf() {
        let x: R32 = 1.0.try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_neg_inf() {
        let x: R32 = (-1.0).try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_from_inf_ref() {
        let x = 1.0f64 / 0.0;
        let _: &Real<_> = (&x).try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn finite_panic_from_inf_slice() {
        let xs = [1.0f64, 1.0f64 / 0.0];
        let _ = Real::<f64>::try_from_slice(&xs).unwrap();
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    fn total_nan_eq() {
        let x: Total<f32> = (0.0 / 0.0).into();
        let y: Total<f32> = (0.0 / 0.0).into();
        assert_eq!(x, y);

        let z: Total<f32> =
            (<f32 as InfinityEncoding>::INFINITY + <f32 as InfinityEncoding>::NEG_INFINITY).into();
        assert_eq!(x, z);

        #[cfg(feature = "std")]
        {
            use crate::real::UnaryRealFunction;

            let w: Total<f32> = (UnaryRealFunction::sqrt(-1.0f32)).into();
            assert_eq!(x, w);
        }
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    #[allow(invalid_nan_comparisons)]
    fn cmp_proxy_primitive() {
        // Compare a canonicalized `NaN` with a primitive `NaN` with a different representation.
        let x: Total<f32> = (0.0 / 0.0).into();
        assert_eq!(x, f32::sqrt(-1.0));

        // Compare a canonicalized `INF` with a primitive `NaN`.
        let y: Total<f32> = (1.0 / 0.0).into();
        assert!(y < (0.0 / 0.0));

        // Compare a proxy that disallows `INF` to a primitive `INF`.
        let z: R32 = 0.0.try_into().unwrap();
        assert_eq!(z.partial_cmp(&(1.0 / 0.0)), None);
    }

    #[test]
    fn sum() {
        let xs = [
            1.0.try_into().unwrap(),
            2.0.try_into().unwrap(),
            3.0.try_into().unwrap(),
        ];
        assert_eq!(xs.iter().cloned().sum::<R32>(), R32::assert(6.0));
    }

    #[test]
    fn product() {
        let xs = [
            1.0.try_into().unwrap(),
            2.0.try_into().unwrap(),
            3.0.try_into().unwrap(),
        ];
        assert_eq!(xs.iter().cloned().product::<R32>(), R32::assert(6.0),);
    }

    // TODO: This test is questionable.
    #[test]
    fn impl_traits() {
        fn as_infinite<T>(_: T)
        where
            T: InfinityEncoding,
        {
        }

        fn as_nan<T>(_: T)
        where
            T: NanEncoding,
        {
        }

        fn as_real<T>(_: T)
        where
            T: RealFunction,
        {
        }

        let finite = Real::<f32>::default();
        as_real(finite);

        let notnan = ExtendedReal::<f32>::default();
        as_infinite(notnan);
        as_real(notnan);

        let ordered = Total::<f32>::default();
        as_infinite(ordered);
        as_nan(ordered);
    }

    #[test]
    fn fmt() {
        let x: Total<f32> = 1.0.into();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", x);
        let y: ExtendedReal<f32> = 1.0.try_into().unwrap();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", y);
        let z: Real<f32> = 1.0.try_into().unwrap();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", z);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialize() {
        assert_eq!(
            R32::assert(1.0),
            serde_json::from_str::<R32>("1.0").unwrap()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    #[should_panic]
    fn deserialize_panic_on_violation() {
        // TODO: See `Serde`. This does not test a value that violates `E32`'s constraints;
        //       instead, this simply fails to deserialize `f32` from `"null"`.
        let _: E32 = serde_json::from_str("null").unwrap();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialize() {
        use crate::divergence::OrPanic;

        assert_eq!(
            "1.0",
            serde_json::to_string(&E32::<OrPanic>::assert(1.0)).unwrap()
        );
        // TODO: See `Serde`.
        assert_eq!(
            "null",
            serde_json::to_string(&E32::<OrPanic>::INFINITY).unwrap()
        );
    }
}
