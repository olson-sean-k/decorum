//! IEEE 754 floating-point proxy types that apply ordering and configurable contraints and
//! divergence.
//!
//! [`Constrained`] types wrap primitive floating-point types and change their behavior with
//! respect to ordering, equivalence, hashing, supported values, and error behaviors. These types
//! have the same representation as primitive floating-point types and can often be used as drop-in
//! replacements.
//!
//! [Constraints][`constraint`] configure the behavior of [`Constrained`]s by determining the set
//! of IEEE
//! 754 floating-point values that can be represented and how to [diverge][`divergence`] if a value
//!     that is not in this set is encountered. The following table summarizes the proxy type
//!     definitions and their constraints:
//!
//! | Type Definition  | Sized Definitions | Trait Implementations                           | Disallowed Values     |
//! |------------------|-------------------|-------------------------------------------------|-----------------------|
//! | [`Total`]        |                   | `BaseEncoding + InfinityEncoding + NanEncoding` |                       |
//! | [`ExtendedReal`] | `E32`, `E64`      | `BaseEncoding + InfinityEncoding`               | `NaN`                 |
//! | [`Real`]         | `R32`, `R64`      | `BaseEncoding`                                  | `NaN`, `-INF`, `+INF` |
//!
//! The [`ExtendedReal`] and [`Real`] types disallow values that represent not-a-number, $\infin$,
//! and $-\infin$. These types diverge if such a value is encountered, which may result in an error
//! encoding output (e.g., a `Result::Err`) or even a panic. Notably, the [`Total`] type applies no
//! constraints and is infallible (never diverges).
//!
//! [`constraint`]: crate::constraint
//! [`divergence`]: crate::divergence
//! [`ExtendedReal`]: crate::ExtendedReal
//! [`Real`]: crate::Real
//! [`Result::Err`]: core::result::Result::Err
//! [`Total`]: crate::Total

mod constrained;
mod nan;

#[cfg(feature = "serde")]
use serde_derive::{Deserialize, Serialize};

use crate::constraint::Constraint;
use crate::Primitive;

pub use constrained::{Constrained, ErrorOf, ExpressionOf, OutputOf};
pub use nan::Nan;

/// An IEEE 754 floating-point proxy type.
pub trait Proxy: Sized {
    type Primitive: Primitive;
    type Constraint: Constraint;
}

// TODO: By default, Serde serializes floating-point primitives representing `NaN` and infinities
//       as `"null"`. Moreover, Serde cannot deserialize `"null"` as a floating-point primitive.
//       This means that information is lost when serializing and deserializing is impossible for
//       non-real values.
/// Serialization container.
///
/// This type is represented and serialized transparently as its inner type `T`. `Constrained` uses
/// this type for its own serialization and deserialization. Importantly, this uses a conversion
/// when deserializing that upholds the constraints on proxy types, so it is not possible to
/// deserialize a floating-point value into a proxy type that does not support that value.
///
/// See the following for more context and details:
///
/// - https://github.com/serde-rs/serde/issues/642
/// - https://github.com/serde-rs/serde/issues/939
#[cfg(feature = "serde")]
#[derive(Deserialize, Serialize)]
#[serde(transparent)]
#[derive(Clone, Copy)]
#[repr(transparent)]
struct Serde<T> {
    inner: T,
}
