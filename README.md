![Decorum](https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.png)

**Decorum** is a Rust library that provides ordering, equality, and hashing for
floating point types.

[![Build Status](https://travis-ci.org/olson-sean-k/decorum.svg?branch=master)](https://travis-ci.org/olson-sean-k/decorum)
[![Build Status](https://ci.appveyor.com/api/projects/status/3630cscs05c6ux86?svg=true)](https://ci.appveyor.com/project/olson-sean-k/decorum)
[![Documentation](https://docs.rs/decorum/badge.svg)](https://docs.rs/decorum)
[![Crate](https://img.shields.io/crates/v/decorum.svg)](https://crates.io/crates/decorum)

## Proxy Types

Several proxy (wrapper) types are provided. Proxy types provide two primary
features: they normalize floating point values to support `Eq`, `Hash`, and
`Ord`, and they constrain the values they support. Different types place
different constraints on the values that they can represent.

| Type      | Numeric Traits                  | Disallowed Values  |
|-----------|---------------------------------|--------------------|
| `Ordered` | `Infinite` `Float` `Nan` `Real` |                    |
| `NotNan`  | `Infinite` `Real`               | `NaN`              |
| `Finite`  | `Real`                          | `-INF` `INF` `NaN` |

All proxy types implement the expected operation traits, such as `Add` and
`Mul`. These types also implement numeric traits from the
[num-traits](https://crates.io/crate/num-traits) crate (such as `Float`, `Num`,
`NumCast`, etc.), in addition to more targeted traits like `Real` and `Nan`.

Normalization and special behavior for `NaN` provide a total ordering for all
proxy types and ordering functions. `NaN` values are considered greater than
non-`NaN` values, and `NaN`s are considered equal regardless of their internal
representation.

## Constraints

The `NotNan` and `Finite` types wrap raw floating point values and disallow
certain values like `NaN`, `INF`, and `-INF`. They will panic if an operation
or conversion invalidates these constraints. The `Ordered` type allows any
valid IEEE-754 value (there are no constraints).

## Conversions

Proxy types are used via conversions to and from primitive floating point
values and other proxy types.

| Conversion      | Failure | Description                          |
|-----------------|---------|--------------------------------------|
| `from_inner`    | Panic   | Creates a proxy from a primitive.    |
| `into_inner`    |         | Converts a proxy into a primitive.   |
| `into_superset` |         | Converts a proxy into another proxy. |
| `from_subset`   |         | Creates a proxy from another proxy.  |

The `from_inner` and `into_inner` conversions are exposed by the `FloatProxy`
trait, which can be used in generic code to support different proxy types.

The `into_superset` and `from_subset` conversions provide an inexpensive way to
convert between proxy types with different (and compatible) constraints.

## Functions

All proxy types implement `Eq`, `Hash`, and `Ord`, but sometimes it is not
possible or ergonomic to use a proxy type. Functions accepting raw floating
point values can be used for equality, hashing, and ordering instead.

| Function     | Description                                   |
|--------------|-----------------------------------------------|
| `eq_float`   | Determines if normalized values are equal.    |
| `hash_float` | Hashes a normalized value.                    |
| `ord_float`  | Determines the ordering of normalized values. |

Each basic function has a variant for arrays and slices, such as
`eq_float_slice` and `ord_float_array`.

For example, with the [derivative](https://crates.io/crates/derivative) crate,
floating point fields can be hashed using one of these functions when deriving
`Hash`. A `Vertex` type used by a rendering pipeline could use this for
floating point fields:

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

Scalar values, slices, and arrays up to length 16 are supported.
