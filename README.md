![Decorum](https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.png)

**Decorum** is a Rust library that provides ordering, equality, hashing, and
constraints for floating-point types.

[![Build Status](https://travis-ci.org/olson-sean-k/decorum.svg?branch=master)](https://travis-ci.org/olson-sean-k/decorum)
[![Build Status](https://ci.appveyor.com/api/projects/status/3630cscs05c6ux86?svg=true)](https://ci.appveyor.com/project/olson-sean-k/decorum)
[![Documentation](https://docs.rs/decorum/badge.svg)](https://docs.rs/decorum)
[![Crate](https://img.shields.io/crates/v/decorum.svg)](https://crates.io/crates/decorum)

## Proxy Types

Decorum exposes several proxy (wrapper) types. Proxy types provide two primary
features: they canonicalize floating-point values to support `Eq`, `Hash`, and
`Ord`, and they constrain the values they support. Different types place
different constraints on the values that they can represent, with the `Ordered`
type applying no constraints (only ordering).

| Type      | Aliases      | Numeric Traits                             | Disallowed Values    |
|-----------|--------------|--------------------------------------------|----------------------|
| `Ordered` | none         | `Encoding + Real + Infinite + Nan + Float` | n/a                  |
| `NotNan`  | `N32`, `N64` | `Encoding + Real + Infinite`               | `NaN`                |
| `Finite`  | `R32`, `R64` | `Encoding + Real`                          | `NaN`, `-INF`, `INF` |

All proxy types implement the expected operation traits, such as `Add` and
`Mul`. These types also implement numeric traits from the
[num-traits](https://crates.io/crate/num-traits) crate (such as `Float`, `Num`,
`NumCast`, etc.), in addition to more targeted traits like `Real` and `Nan`
provided by Decorum.

Proxy types should work as a drop-in replacement for primitive types in most
applications, with the most common exception being initialization (because it
may require a conversion).

## Ordering

`NaN` and zero are canonicalized to a single representation (called `CNaN` and
`C0` respectively) to provide the following total ordering for all proxy types
and ordering functions:

```
[ -INF | ... | C0 | ... | +INF | CNaN ]
```

Note that `NaN` is canonicalized to `CNaN`, which has a single representation
and supports the relations `CNaN = CNaN` and `CNaN > x ∋ x ≠ CNaN`. `+0` and
`-0` are also canonicalized to `C0`, which is equivalent to `+0`.

## Constraints

The `NotNan` and `Finite` types wrap raw floating-point values and disallow
certain values like `NaN`, `INF`, and `-INF`. They will panic if an operation
or conversion invalidates these constraints. The `Ordered` type allows any
valid IEEE-754 value (there are no constraints). For most use cases, either
`Ordered` or `NotNan` are appropriate.

## Traits

Numeric traits are essential for generic programming, but the constraints used
by some proxy types prevent them from implementing the ubiquitous `Float`
trait, because it implies the presence of `-INF`, `INF`, and `NaN`.

Decorum provides more granular traits that separate these APIs: `Real`,
`Infinite`, `Nan`, and `Encoding`. These traits are monkey-patched using
blanket implementations so that the trait bounds `T: Float` and `T: Encoding +
Infinite + Nan + Real` are equivalent, and for all primitive and proxy types `T: Float ⇒ T:
Encoding + Infinite + Nan + Real`.

For example, code that wishes to be generic over floating-point types
representing real numbers can use a bound on the `Real` trait:

```rust
use decorum::Real;

fn f<T>(x: T, y: T) -> T
where
    T: Real,
{
    x + y
}
```

Both Decorum and [num-traits](https://crates.io/crate/num-traits) expose a
`Real` trait. Due to some subtle differences and [an
issue](https://github.com/rust-num/num-traits/issues/49) with the num-traits API
that makes implementing `Real` difficult, Decorum continues to vendor its own.
However, both traits are implemented for proxy types.

## Conversions

Proxy types are used via conversions to and from primitive floating-point
values and other proxy types.

| Conversion      | Input     | Output    | Failure |
|-----------------|-----------|-----------|---------|
| `from_inner`    | primitive | proxy     | panic   |
| `into_inner`    | proxy     | primitive | n/a     |
| `from_subset`   | proxy     | proxy     | n/a     |
| `into_superset` | proxy     | proxy     | n/a     |

The `from_inner` and `into_inner` conversions move primitive floating-point
values into and out of proxies. The `into_superset` and `from_subset`
conversions provide an inexpensive way to convert between proxy types with
different (and compatible) constraints. All conversions also support the
standard `From` and `Into` traits, which can also be applied to literals:

```rust
use decorum::R32;

fn f(x: R32) -> R32 {
    x * 2.0
}
let y: R32 = 3.1459.into();
let z = f(2.7182.into());
let w: f32 = z.into();
```

## Functions

All proxy types implement `Eq`, `Hash`, and `Ord`, but sometimes it is not
possible or ergonomic to use a proxy type. Functions accepting raw floating
point values can be used for equality, hashing, and ordering instead.

| Function     | Analogous Trait  |
|--------------|------------------|
| `eq_float`   | `Eq`             |
| `hash_float` | `Hash`           |
| `ord_float`  | `Ord`            |

These functions all canonicalize their inputs and the output is equivalent to
wrapping the input values in the `Ordered` proxy and using `Eq`, `Hash`, or
`Ord`.

Each basic function has a variant for arrays and slices, such as
`eq_float_slice` and `ord_float_array`. Arrays up to length 16 are supported.
When comparing sequences, longer sequences are always greater than shorter
sequences. For sequences of the same length, order is determined by the first
corresponding elements that differ, and if no such elements exist then the
sequences are equal.

For example, with the [derivative](https://crates.io/crates/derivative) crate,
floating-point fields can be hashed easily using one of these functions when
deriving `Hash`. A `Vertex` type used by a rendering pipeline could use this
for floating-point fields:

```rust
use decorum;

#[derive(Derivative)]
#[derivative(Hash)]
pub struct Vertex {
    #[derivative(Hash(hash_with = "decorum::hash_float_array"))]
    pub position: [f32; 3],
    ...
}
```
