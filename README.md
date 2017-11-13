![Decorum](https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.png)

**Decorum** is a Rust library that provides ordering, equality, and hashing for
floating point types.

[![Build Status](https://travis-ci.org/olson-sean-k/decorum.svg?branch=master)](https://travis-ci.org/olson-sean-k/decorum)
[![Documentation](https://docs.rs/decorum/badge.svg)](https://docs.rs/decorum)
[![Crate](https://img.shields.io/crates/v/decorum.svg)](https://crates.io/crates/decorum)

## Proxy Types

Several proxy (wrapper) types are provided. Every proxy is normalized to
implement `Hash`, but different types place different constraints on the values
that can be represented. This influences which numeric and ordering traits are
implemented.

| Type      | Numeric Traits                  | Operation Traits    |
|-----------|---------------------------------|---------------------|
| `Ordered` | `Infinite` `Float` `Nan` `Real` | `Eq` `Hash` `Ord`   |
| `NotNan`  | `Infinite` `Real`               | `Eq` `Hash` `Ord`   |
| `Finite`  | `Real`                          | `Eq` `Hash` `Ord`   |

All proxy types implement the expected operation traits, such as `Add` and
`Mul`. These types also implement numeric traits from the
[num-traits](https://crates.io/crate/num-traits) crate (such as `Float`, `Num`,
`NumCast`, etc.), in addition to more targeted traits like `Real` and `Nan`.

Note that `Ordered`, which allows `NaN` values, is still ordered. `NaN` values
are considered greater than non-`NaN` values, and `NaN`s are considered equal
regardless of their internal representation.

## Constrained Proxy Types

The `NotNan` and `Finite` types wrap raw floating point values and disallow
certain values like `NaN`, `INF`, and `-INF`. They will panic if an operation
invalidates these constraints. Depending on configuration, the `from_raw_float`
function may panic or continue silently. The `try_from_raw_float` function
always checks constraints, returning a `Result`.

Constraint checking can be toggled with the `enforce-constraints` feature. This
is useful if code would like to enforce constraints for some builds but not
others (e.g., debug vs. release builds). Constraint checking is enabled by
default. If checking is disabled, `from_raw_float` will never panic.

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
