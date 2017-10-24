use num_traits::Float;

// TODO: Implement this. It provides ordering and is much like `NotNan`, but
//       can be NaN.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct OrderedFloat<T>(T)
where
    T: Float;
