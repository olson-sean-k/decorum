use num_traits::Float;
use std::marker::PhantomData;

// TODO: Allow policies to be disable in release builds and respect the
//       "always-enforce-policy" feature. This may require larger code changes,
//       because omitting policy checks in release builds will still require
//       examining an `Option`.

pub trait FloatPolicy<T>: Copy + PartialEq + PartialOrd + Sized
where
    T: Float,
{
    fn evaluate(value: T) -> Option<T>;
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct NotNanPolicy<T>
where
    T: Float,
{
    phantom: PhantomData<T>,
}

impl<T> FloatPolicy<T> for NotNanPolicy<T>
where
    T: Float,
{
    fn evaluate(value: T) -> Option<T> {
        if value.is_nan() {
            None
        }
        else {
            Some(value)
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct FinitePolicy<T>
where
    T: Float,
{
    phantom: PhantomData<T>,
}

impl<T> FloatPolicy<T> for FinitePolicy<T>
where
    T: Float,
{
    fn evaluate(value: T) -> Option<T> {
        if value.is_nan() | value.is_infinite() {
            None
        }
        else {
            Some(value)
        }
    }
}
