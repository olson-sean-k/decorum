use crate::Primitive;

/// An incomparable primitive IEEE 754 floating-point `NaN`.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Nan<T>
where
    T: Primitive,
{
    inner: T,
}

impl<T> Nan<T>
where
    T: Primitive,
{
    pub(crate) const fn unchecked(inner: T) -> Self {
        Nan { inner }
    }

    pub const fn into_inner(self) -> T {
        self.inner
    }
}

impl From<Nan<f32>> for f32 {
    fn from(nan: Nan<f32>) -> Self {
        nan.into_inner()
    }
}

impl From<Nan<f64>> for f64 {
    fn from(nan: Nan<f64>) -> Self {
        nan.into_inner()
    }
}
