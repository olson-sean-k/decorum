![Decorum](https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.png)

**Decorum** is a Rust library that provides ordering, equality, and hashing for
floating point types.

[![Build Status](https://travis-ci.org/olson-sean-k/decorum.svg?branch=master)](https://travis-ci.org/olson-sean-k/decorum)
[![Build Status](https://ci.appveyor.com/api/projects/status/3630cscs05c6ux86?svg=true)](https://ci.appveyor.com/project/olson-sean-k/decorum)
[![Documentation](https://docs.rs/decorum/badge.svg)](https://docs.rs/decorum)
[![Crate](https://img.shields.io/crates/v/decorum.svg)](https://crates.io/crates/decorum)

## Proxy Types

Several proxy (wrapper) types are provided. Every proxy is normalized to
implement `Eq`, `Hash`, and `Ord`, but different types place different
constraints on the values that can be represented.

| Type      | Numeric Traits                  | Disallowed Values  |
|-----------|---------------------------------|--------------------|
| `Ordered` | `Infinite` `Float` `Nan` `Real` | None               |
| `NotNan`  | `Infinite` `Real`               | `NaN`              |
| `Finite`  | `Real`                          | `-INF` `INF` `NaN` |

All proxy types implement the expected operation traits, such as `Add` and
`Mul`. These types also implement numeric traits from the
[num-traits](https://crates.io/crate/num-traits) crate (such as `Float`, `Num`,
`NumCast`, etc.), in addition to more targeted traits like `Real` and `Nan`.

Note that `Ordered`, which allows `NaN` values, is still ordered. `NaN` values
are considered greater than non-`NaN` values, and `NaN`s are considered equal
regardless of their internal representation.

## Constraints

The `NotNan` and `Finite` types wrap raw floating point values and disallow
certain values like `NaN`, `INF`, and `-INF`. They will panic if an operation
or conversion invalidates these constraints and checking is enabled.

Constraint checking can be toggled with the `enforce-constraints` feature. This
is useful if code would like to enforce constraints for some builds but not
others (e.g., debug vs. release builds). Constraint checking is enabled by
default. If checking is disabled, `from_raw_float` will never panic.

## Conversions

Proxy types are used via conversions to and from primitive floating point
values and other proxy types.

| Conversion           | Failure | Description                          |
|----------------------|---------|--------------------------------------|
| `from_raw_float`     | Panic   | Creates a proxy from a primitive.    |
| `into_raw_float`     | None    | Converts a proxy into a primitive.   |
| `into_superset`      | None    | Converts a proxy into another proxy. |
| `from_subset`        | None    | Creates a proxy from another proxy.  |

The `from_raw_float` and `into_raw_float` conversions are exposed by the
`FloatProxy` trait, which can be used in generic code to support different
proxy types.

## Hashing Functions

All proxy types implement `Hash`, but sometimes it is not possible or ergonomic
to use these. Hashing functions for raw floating point values can be used
instead.

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
