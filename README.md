<div align="center">
    <img alt="Decorum" src="https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.svg?sanitize=true" width="320"/>
</div>
<br/>

**Decorum** is a Rust library that provides total ordering, equivalence,
hashing, and constraints for floating-point representations. Decorum does not
require `std` and can be used in `#[no_std]` environments (see the `std`
feature).

[![CI](https://github.com/olson-sean-k/decorum/workflows/CI/badge.svg)](https://github.com/olson-sean-k/decorum/actions)
[![Documentation](https://docs.rs/decorum/badge.svg)](https://docs.rs/decorum)
[![Crate](https://img.shields.io/crates/v/decorum.svg)](https://crates.io/crates/decorum)

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

## Proxy Types

Decorum exposes several proxy (wrapper) types. Proxy types provide two primary
features: they implement a total ordering with `Eq`, `Ord`, and `Hash` and they
constrain the class of values they can represent. Different type definitions
apply different constraints, with the `Total` type applying no constraints at
all.

| Type     | Aliases      | Trait Implementations                      | Disallowed Values     |
|----------|--------------|--------------------------------------------|-----------------------|
| `Total`  |              | `Encoding + Real + Infinite + Nan + Float` |                       |
| `NotNan` | `N32`, `N64` | `Encoding + Real + Infinite`               | `NaN`                 |
| `Finite` | `R32`, `R64` | `Encoding + Real`                          | `NaN`, `-INF`, `+INF` |


Proxy types implement common operation traits, such as `Add` and `Mul`. These
types also implement numeric traits from the
[`num-traits`](https://crates.io/crate/num-traits) crate (such as `Float`,
`Num`, `NumCast`, etc.), in addition to more targeted traits like `Real` and
`Nan` provided by Decorum.

Constraint violations cause panics. For example, `NotNan` is useful for avoiding
or tracing sources of `NaN`s in computation, while `Total` provides useful
features without introducing any panics at all, because it allows any IEEE-754
floating-point values.

Proxy types should work as a drop-in replacement for primitive types in most
applications, with the most common exception being initialization (because it
requires a conversion). Serialization is optionally supported via
[serde](https://crates.io/crates/serde) (see the `serialize-serde` feature).

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

Both Decorum and [`num-traits`](https://crates.io/crate/num-traits) provide
`Real` and `Float` traits. These traits are somewhat different and are not
always interchangeable. Traits from both crates are implemented by Decorum where
possible. For example, `Total` implements `Float` from both Decorum and
`num-traits`.

## Conversions

Proxy types are used via conversions to and from primitive floating-point
types and other proxy types.

| Conversion      | Input     | Output    | Violation |
|-----------------|-----------|-----------|-----------|
| `from_inner`    | primitive | proxy     | panic     |
| `into_inner`    | proxy     | primitive | n/a       |
| `from_subset`   | proxy     | proxy     | n/a       |
| `into_superset` | proxy     | proxy     | n/a       |

The `from_inner` and `into_inner` conversions move primitive floating-point
values into and out of proxies. The `into_superset` and `from_subset`
conversions provide an inexpensive way to convert between proxy types with
different but compatible constraints. All conversions also support the standard
`From` and `Into` traits, which can also be applied to literals:

```rust
use decorum::R32;

fn f(x: R32) -> R32 {
    x * 2.0
}
let y: R32 = 3.1459.into();
let z = f(2.7182.into());
let w: f32 = z.into();
```

## Primitives

Proxy types implement `Eq`, `Hash`, and `Ord`, but sometimes it is not
possible or ergonomic to use such a type. Traits can be used with primitive
floating-point values for ordering, equivalence, and hashing instead.

| Trait       | Analogous Trait  |
|-------------|------------------|
| `FloatEq`   | `Eq`             |
| `FloatHash` | `Hash`           |
| `FloatOrd`  | `Ord`            |

These traits use the same total ordering and equivalence rules that proxy types
do. They are implemented for base types as well as slices:

```rust
use decorum::cmp::FloatEq;

let x = 0.0f64 / 0.0f64; // `NaN`.
let y = f64::INFINITY + f64::NEG_INFINITY; // `NaN`.
assert!(x.float_eq(&y));

let xs = [1.0f64, f64::NAN, f64::INFINITY];
let ys = [1.0f64, f64::NAN, f64::INFINITY];
assert!(xs.float_eq(&ys));
```
