<div align="center">
    <img alt="Decorum" src="https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.svg?sanitize=true" width="320"/>
</div>
<br/>

**Decorum** is a Rust library that provides total ordering, equivalence,
hashing, constraints, error handling, and more for IEEE 754 floating-point
representations. Decorum requires Rust 1.65.0 or higher and, except for specific
features, does **not** require the `std` nor `alloc` libraries.

[![GitHub](https://img.shields.io/badge/GitHub-olson--sean--k/decorum-8da0cb?logo=github&style=for-the-badge)](https://github.com/olson-sean-k/decorum)
[![docs.rs](https://img.shields.io/badge/docs.rs-decorum-66c2a5?logo=rust&style=for-the-badge)](https://docs.rs/decorum)
[![crates.io](https://img.shields.io/crates/v/decorum.svg?logo=rust&style=for-the-badge)](https://crates.io/crates/decorum)

## Basic Usage

Panic when a `NaN` is encountered:

```rust
use decorum::NotNan;

let x = NotNan::<f64>::assert(0.0);
let y = NotNan::<f64>::assert(0.0);
let z = x / y; // Panics.
```

Hash totally ordered IEEE 754 floating-point representations:

```rust
use decorum::real::UnaryRealFunction;
use decorum::Real;
use std::collections::HashMap;

let key = Real::<f64>::PI;
let mut xs: HashMap<_, _> = [(key, "pi")].into_iter().collect();
```

Configure the behavior of an IEEE 754 floating-point representation:

```rust
pub mod real {
    use decorum::constraint::IsReal;
    use decorum::divergence::{AsResult, OrError};
    use decorum::proxy::{OutputOf, Proxy};

    // A 64-bit floating point type that must represent a real number and returns
    // `Result`s from fallible operations.
    pub type Real = Proxy<f64, IsReal<OrError<AsResult>>>;
    pub type Result = OutputOf<Real>;
}

use real::Real;

pub fn f(x: Real) -> real::Result { ... }

let x = Real::assert(0.0);
let y = Real::assert(0.0);
let z = (x / y)?;
```

## Proxy Types

The primary API of Decorum is its `Proxy` types, which transparently wrap
primitive IEEE 754 floating-point types and configure their behavior. `Proxy`
types support many numeric features and operations and integrate with the
[`num-traits`] crate and others when [Cargo features](#cargo-features) are
enabled. Depending on its configuration, a proxy can be used as a drop-in
replacement for primitive floating-point types.

The following `Proxy` behaviors can be configured:

1. the allowed subset of IEEE 754 floating-point values
1. the output type of fallibe operations (that may produce non-member values
   w.r.t. a subset)
1. what happens when an error occurs (i.e., return an error value or panic)

Note that the output type of fallible operations and the error behavior are
independent. A `Proxy` type may return a `Result` and yet panic if an error
occurs, which can be useful for conditional compilation and builds wherein
**behavior** changes but types do not. The behavior of a `Proxy` type is
configured using two mechanisms: _constraints_ and _divergence_.

```rust
use decorum::constraint::IsReal;
use decorum::divergence::OrPanic;
use decorum::proxy::Proxy;

// `Real` must represent a real number and otherwise panics.
pub type Real = Proxy<f64, IsReal<OrPanic>>;
```

Constraints specify a subset of floating-point values that a proxy may
represent. IEEE 754 floating-point values are divided into three such subsets:

| Subset        | Example Member |
|---------------|----------------|
| real numbers  | `3.1459`       |
| infinities    | `+INF`         |
| not-a-numbers | `NaN`          |

Constraints can be used to strictly represent real numbers, extended reals, or
complete but totally ordered IEEE 754 types (i.e., no constraints). Available
constraints are summarized below:

| Constraint       | Members                                 | Fallible  |
|------------------|-----------------------------------------|-----------|
| `IsFloat`        | real numbers, infinities, not-a-numbers | no        |
| `IsExtendedReal` | real numbers, infinities                | yes       |
| `IsReal`         | real numbers                            | yes       |

`IsFloat` supports all IEEE 754 floating-point values and so applies no
constraint at all. As such, it has no fallible operations w.r.t. the constraint
and does not accept a divergence.

Many operations on members of these subsets may produce values from other
subsets that are illegal w.r.t. constraints, such as the addition of two real
numbers resulting in `+INF`. A _divergence type_ determines both the behavior
when an illegal value is encountered as well as the output type of such fallible
operations.

| Divergence | OK       | Error     | Default Output Kind |
|------------|----------|-----------|---------------------|
| `OrPanic`  | continue | **panic** | `AsSelf`            |
| `OrError`  | continue | break     | `AsExpression`      |

In the above table, _continue_ refers to returning a **non**-error value while
_break_ refers to returning an error value. If an illegal value is encountered,
then **the `OrPanic` divergence panics** while the `OrError` divergence
constructs a value that encodes the error. The output type of fallible
operations is determined by an _output kind_:

| Branch         | Type                  | Continue        | Break          |
|----------------|-----------------------|-----------------|----------------|
| `AsSelf`       | `Self`                | `Self`          |                |
| `AsOption`     | `Option<Self>`        | `Some(Self)`    | `None`         |
| `AsResult`     | `Result<Self, E>`     | `Ok(Self)`      | `Err(E)`       |
| `AsExpression` | `Expression<Self, E>` | `Defined(Self)` | `Undefined(E)` |

In the table above, `Self` refers to a `Proxy` type and `E` refers to the
associated error type of its constraint. Note that only the `OrPanic` divergence
supports `AsSelf` and can output the same type as its input type for fallible
operations (just like primitive IEEE 754 floating-point types).

With the sole exception of `AsSelf`, the output type of fallible operations is
extrinsic: fallible operations produce types that differ from their input types.
The `Expression` type, which somewhat resembles the standard `Result` type,
improves the ergonomics of error handling by implementing mathematical traits
such that it can be used directly in expressions and defer error checking.

```rust
use decorum::constraint::IsReal;
use decorum::divergence::{AsExpression, OrError};
use decorum::proxy::{OutputOf, Proxy};
use decorum::real::UnaryRealFunction;
use decorum::try_expression;

pub type Real = Proxy<f64, IsReal<OrError<AsExpression>>>;
pub type Expr = OutputOf<Real>;

pub fn f(x: Real, y: Real) -> Expr {
    let sum = x + y;
    sum * g(x)
}

pub fn g(x: Real) -> Expr {
    x + Real::ONE
}

let x: Real = try_expression! { f(Real::E, -Real::ONE) };
// ...
```

When using a nightly Rust toolchain with the `unstable` [Cargo
feature](#cargo-features) enabled, `Expression` also supports the (at time of
writing) unstable `Try` trait and try operator `?`.

```rust
// As above, but using the try operator `?`.
let x: Real = f(Real::E, -Real::ONE)?;
```

`Proxy` types support numerous constructions and conversions depending on
configuration, including conversions for references, slices, subsets, supersets,
and more. Conversions are provided via inherent functions and implementations of
the standard `From` and `TryFrom` traits. The following inherent functions are
supported by all `Proxy` types, though some more bespoke constructions are
available for specific configurations.

| Proxy Method           | Input     | Output    | Error         |
|------------------------|-----------|-----------|---------------|
| `new`                  | primitive | proxy     | break         |
| `assert`               | primitive | proxy     | **panic**     |
| `try_new`              | primitive | proxy     | `Result::Err` |
| `try_from_{mut_}slice` | primitive | proxy     | `Result::Err` |
| `into_inner`           | proxy     | primitive |               |
| `from_subset`          | proxy     | proxy     |               |
| `into_superset`        | proxy     | proxy     |               |

The following type definitions provide common proxy configurations. Each type
implements different traits that describe the supported encoding and elements of
IEEE 754 floating-point based on its constraints.

| Type Definition | Sized Aliases | Trait Implementations                           | Illegal Values        |
|-----------------|---------------|-------------------------------------------------|-----------------------|
| `Total`         |               | `BaseEncoding + InfinityEncoding + NanEncoding` |                       |
| `ExtendedReal`  | `E32`, `E64`  | `BaseEncoding + InfinityEncoding`               | `NaN`                 |
| `Real`          | `R32`, `R64`  | `BaseEncoding`                                  | `NaN`, `-INF`, `+INF` |

## Relations and Total Ordering

Decorum provides the following non-standard total ordering for IEEE 754
floating-point representations:

```
-INF < ... < 0 < ... < +INF < NaN
```

IEEE 754 floating-point encoding has multiple representations of zero (`-0` and
`+0`) and `NaN`. This ordering and equivalence relations consider all zero and
`NaN` representations equal, which differs from the [standard partial
ordering](https://en.wikipedia.org/wiki/NaN#Comparison_with_NaN).

Some proxy types disallow unordered `NaN` values and therefore support a total
ordering based on the ordered subset of non-`NaN` floating-point values. Proxy
types that use `IsFloat` (such as the `Total` type definition) support `NaN` but
use the total ordering described above to implement the standard `Eq`, `Hash`,
and `Ord` traits.

The following traits can be used to compare and hash primitive floating-point
values (including slices) using this non-standard relation.

| Floating-Point Trait | Standard Trait   |
|----------------------|------------------|
| `CanonicalEq`        | `Eq`             |
| `CanonicalHash`      | `Hash`           |
| `CanonicalOrd`       | `Ord`            |

```rust
use decorum::cmp::CanonicalEq;

let x = 0.0f64 / 0.0f64; // `NaN`.
let y = f64::INFINITY + f64::NEG_INFINITY; // `NaN`.
assert!(x.eq_canonical_bits(&y));
```

Decorum also provides the `IntrinsicOrd` trait and the `min_or_undefined` and
`max_or_undefined` functions. These pairwise comparisons can be used with
partially ordered types that have an intrinsic representation for undefined,
such as `Option` (`None`) and IEEE 754 floating-point representations (`NaN`).
For floating-point representations, this provides an ergonomic method for
comparison that naturally propogates `NaN`s just like floating-point operations
do (unlike `f64::max`, etc.).

```rust
use decorum::cmp;
use decorum::real::{Endofunction, RealFunction, UnaryRealFunction};

pub fn f<T>(x: T, y: T) -> T
where
    T: Endofunction + RealFunction,
{
    // If the comparison is undefined, then `min` is assigned some
    // representation of undefined. For floating-point types, `NaN` represents
    // undefined and cannot be compared, so `min` is `NaN` if `x` or `y` is
    // `NaN`.
    let min = cmp::min_or_undefined(x, y);
    min * T::PI
}
```

## Mathematical Traits

The `real` module provides various traits that describe real numbers and
constructions via IEEE 754 floating-point types. These traits model functions
and operations on real numbers and specify a codomain for functions where the
output is not mathematically confined to the reals or a floating-point exception
may yield a non-real approximation or error. For example, the logarithm of zero
is undefined and the sum of two very large reals results in an infinity in IEEE
754. For proxy types, the codomain is the same as the branch type of its
divergence (see above).

Real number and IEEE 754 encoding traits can both be used for generic
programming. The following code demonstrates a function that accepts types that
support floating-point infinities and real functions.

```rust
use decorum::real::{Endofunction, RealFunction};
use decorum::InfinityEncoding;

fn f<T>(x: T, y: T) -> T
where
    T: Endofunction + InfinityEncoding + RealFunction,
{
    let z = x / y;
    if z.is_infinite() {
        x + y
    }
    else {
        z + y
    }
}
```

## Cargo Features

Decorum supports the following feature flags.

| Feature    | Default | Description                                                  |
|------------|---------|--------------------------------------------------------------|
| `approx`   | yes     | Implements traits from [`approx`] for `Proxy` types.         |
| `serde`    | yes     | Implements traits from [`serde`] for `Proxy` types.          |
| `std`      | yes     | Integrates the `std` library and enables dependent features. |
| `unstable` | no      | Enables features that require an unstable compiler.          |

[`approx`]: https://crates.io/crates/approx
[`num-traits`]: https://crates.io/crates/num-traits
[`serde`]: https://crates.io/crates/serde
