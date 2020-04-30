use core::hash::{Hash, Hasher};

use crate::canonical::ToCanonicalBits;
use crate::primitive::Primitive;
use crate::{Encoding, Nan};

pub trait FloatHash {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher;
}

impl<T> FloatHash for T
where
    T: Encoding + Nan + Primitive,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.to_canonical_bits().hash(state);
    }
}

impl<T> FloatHash for [T]
where
    T: Encoding + Nan + Primitive,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for item in self.iter() {
            item.hash(state);
        }
    }
}
