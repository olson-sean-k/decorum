/// A primitive floating-point value.
///
/// This trait differentiates types that implement floating-point traits but
/// may not be primitive types.
pub trait Primitive: Copy + Sized {}

impl Primitive for f32 {}
impl Primitive for f64 {}
