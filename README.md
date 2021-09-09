<div align="center">
    <img alt="Decorum" src="https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.svg?sanitize=true" width="320"/>
</div>
<br/>

**Decorum** is a Rust library that provides total ordering, equivalence,
hashing, and constraints for floating-point representations. Decorum requires
Rust 1.43.0 or higher and does not require the `std` library.

[![GitHub](https://img.shields.io/badge/GitHub-olson--sean--k/decorum-8da0cb?logo=github&style=for-the-badge)](https://github.com/olson-sean-k/decorum)
[![docs.rs](https://img.shields.io/badge/docs.rs-decorum-66c2a5?logo=rust&style=for-the-badge)](https://docs.rs/decorum)
[![crates.io](https://img.shields.io/crates/v/decorum.svg?logo=rust&style=for-the-badge)](https://crates.io/crates/decorum)

## Total Ordering

The following total ordering is exposed via traits for primitive types and proxy
types that implement `Ord`:

```
[ -INF < ... < 0 < ... < +INF < NaN ]
```

IEEE-754 floating-point encoding provides multiple representations of zero (`-0`
and `+0`) and `NaN`. This ordering considers all zero and `NaN` representations
equal, which differs from the [standard partial
ordering](https://en.wikipedia.org/wiki/NaN#Comparison_with_NaN).

Some proxy types disallow unordered `NaN` values and therefore support a total
ordering based on the ordered subset of non-`NaN` floating-point values (see
below).

## Proxy Types

Decorum exposes several proxy (wrapper) types. Proxy types provide two primary
features: they implement total ordering and equivalence via the `Eq`, `Ord`, and
`Hash` traits and they constrain the set of floating-point values they can
represent. Different type definitions apply different constraints, with the
`Total` type applying no constraints at all.

| Type     | Aliases      | Trait Implementations                      | Disallowed Values     |
|----------|--------------|--------------------------------------------|-----------------------|
| `Total`  |              | `Encoding + Real + Infinite + Nan + Float` |                       |
| `NotNan` | `N32`, `N64` | `Encoding + Real + Infinite`               | `NaN`                 |
| `Finite` | `R32`, `R64` | `Encoding + Real`                          | `NaN`, `-INF`, `+INF` |


Proxy types implement common operation traits, such as `Add` and `Mul`. These
types also implement numeric traits from the [`num-traits`] crate (such as
`Float`, `Num`, `NumCast`, etc.), in addition to more targeted traits like
`Real` and `Nan` provided by Decorum.

Constraint violations cause panics in numeric operations. For example, `NotNan`
is useful for avoiding or tracing sources of `NaN`s in computation, while
`Total` provides useful features without introducing any panics at all, because
it allows any IEEE-754 floating-point values.

Proxy types should work as a drop-in replacement for primitive types in most
applications with the most common exception being initialization (because it
requires a conversion). Serialization is optionally supported with [`serde`] and
approximate comparisons are optionally supported with [`approx`] via the
`serialize-serde` and `approx` features, respectively.

## Traits

Traits are essential for generic programming, but the constraints used by some
proxy types prevent them from implementing the `Float` trait, because it implies
the presence of `-INF`, `+INF`, and `NaN` (and their corresponding trait
implementations).

Decorum provides more granular traits that separate these APIs: `Real`,
`Infinite`, `Nan`, and `Encoding`. Primitive floating-point types implement all
of these traits and proxy types implement traits that are consistent with their
constraints.

For example, code that wishes to be generic over floating-point types
representing real numbers and infinities can use a bound on the `Infinite` and
`Real` traits:

```rust
use decorum::{Infinite, Real};

fn f<T>(x: T, y: T) -> T
where
    T: Infinite + Real,
{
    let z = x / y;
    if z.is_infinite() {
        y
    }
    else {
        z
    }
}
```

Both Decorum and [`num-traits`] provide `Real` and `Float` traits. These traits
are somewhat different and are not always interchangeable. Traits from both
crates are implemented by Decorum where possible. For example, `Total`
implements `Float` from both Decorum and [`num-traits`].

## Construction and Conversions

Proxy types are used via constructors and conversions from and into primitive
floating-point types and other compatible proxy types. Unlike numeric
operations, these functions do not necessarily panic if a constraint is
violated.

| Method          | Input     | Output    | Violation |
|-----------------|-----------|-----------|-----------|
| `new`           | primitive | proxy     | error     |
| `assert`        | primitive | proxy     | panic     |
| `into_inner`    | proxy     | primitive | n/a       |
| `from_subset`   | proxy     | proxy     | n/a       |
| `into_superset` | proxy     | proxy     | n/a       |

The `new` constructor and `into_inner` conversion move primitive floating-point
values into and out of proxies and are the most basic way to construct and
deconstruct proxies. Note that for `Total`, which has no constraints, the error
type is `Infallible`.

The `assert` constructor panics if the given primitive floating-point value
violates the proxy's constraints. This is equivalent to unwrapping the output of
`new`.

Finally, the `into_superset` and `from_subset` conversions provide an
inexpensive way to convert between proxy types with different but compatible
constraints.

```rust
use decorum::R64;

fn f(x: R64) -> R64 {
    x * 3.0
}

let y = R64::assert(3.1459);
let z = f(R64::new(2.7182).unwrap());
let w = z.into_inner();
```

All conversions also support the standard `From`/`Into` and `TryFrom`/`TryInto`
traits, which can also be applied to primitives and literals.

```rust
use core::convert::{TryFrom, TryInto};
use decorum::R64;

fn f(x: R64) -> R64 {
    x * 2.0
}

let y: R64 = 3.1459.try_into().unwrap();
let z = f(R64::try_from(2.7182).unwrap());
let w: f32 = z.into();
```

## Hashing and Comparing Primitives

Proxy types implement `Eq`, `Hash`, and `Ord`, but sometimes it is not possible
or ergonomic to use such a type. Traits can be used with primitive
floating-point values for ordering, equivalence, and hashing instead.

| Floating-Point Trait | Standard Trait   |
|----------------------|------------------|
| `FloatEq`            | `Eq`             |
| `FloatHash`          | `Hash`           |
| `FloatOrd`           | `Ord`            |

These traits use the same total ordering and equivalence rules that proxy types
do. They are implemented for primitive types like `f64` as well as slices like
`[f64]`.

```rust
use decorum::cmp::FloatEq;

let x = 0.0f64 / 0.0f64; // `NaN`.
let y = f64::INFINITY + f64::NEG_INFINITY; // `NaN`.
assert!(x.float_eq(&y));

let xs = [1.0f64, f64::NAN, f64::INFINITY];
let ys = [1.0f64, f64::NAN, f64::INFINITY];
assert!(xs.float_eq(&ys));
```

[`approx`]: https://crates.io/crates/approx
[`num-traits`]: https://crates.io/crates/num-traits
[`serde`]: https://crates.io/crates/serde
