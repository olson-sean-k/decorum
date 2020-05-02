use core::num::FpCategory;
use core::ops::Neg;
use num_traits::{Num, NumCast};

use crate::{Encoding, Infinite, Nan, Real};

/// Primitive floating-point types.
pub trait Primitive: Copy + Neg<Output = Self> + Num + NumCast + PartialOrd {}

macro_rules! impl_primitive {
    (primitive => $t:ident) => {
        impl Infinite for $t {
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;

            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }

        impl Nan for $t {
            const NAN: Self = <$t>::NAN;

            fn is_nan(self) -> bool {
                self.is_nan()
            }
        }

        impl Primitive for $t {}

        impl Real for $t {
            // TODO: The propagation from a constant in a module requires that
            //       this macro accept an `ident` token rather than a `ty`
            //       token. Use `ty` if these constants become associated
            //       constants of the primitive types.
            const E: Self = core::$t::consts::E;
            const PI: Self = core::$t::consts::PI;
            const FRAC_1_PI: Self = core::$t::consts::FRAC_1_PI;
            const FRAC_2_PI: Self = core::$t::consts::FRAC_2_PI;
            const FRAC_2_SQRT_PI: Self = core::$t::consts::FRAC_2_SQRT_PI;
            const FRAC_PI_2: Self = core::$t::consts::FRAC_PI_2;
            const FRAC_PI_3: Self = core::$t::consts::FRAC_PI_3;
            const FRAC_PI_4: Self = core::$t::consts::FRAC_PI_4;
            const FRAC_PI_6: Self = core::$t::consts::FRAC_PI_6;
            const FRAC_PI_8: Self = core::$t::consts::FRAC_PI_8;
            const SQRT_2: Self = core::$t::consts::SQRT_2;
            const FRAC_1_SQRT_2: Self = core::$t::consts::FRAC_1_SQRT_2;
            const LN_2: Self = core::$t::consts::LN_2;
            const LN_10: Self = core::$t::consts::LN_10;
            const LOG2_E: Self = core::$t::consts::LOG2_E;
            const LOG10_E: Self = core::$t::consts::LOG10_E;

            fn is_sign_positive(self) -> bool {
                <$t>::is_sign_positive(self)
            }

            fn is_sign_negative(self) -> bool {
                <$t>::is_sign_negative(self)
            }

            fn signum(self) -> Self {
                <$t>::signum(self)
            }

            fn abs(self) -> Self {
                <$t>::abs(self)
            }

            fn floor(self) -> Self {
                <$t>::floor(self)
            }

            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }

            fn round(self) -> Self {
                <$t>::round(self)
            }

            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }

            fn fract(self) -> Self {
                <$t>::fract(self)
            }

            fn recip(self) -> Self {
                <$t>::recip(self)
            }

            #[cfg(feature = "std")]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$t>::mul_add(self, a, b)
            }

            #[cfg(feature = "std")]
            fn powi(self, n: i32) -> Self {
                <$t>::powi(self, n)
            }

            #[cfg(feature = "std")]
            fn powf(self, n: Self) -> Self {
                <$t>::powf(self, n)
            }

            #[cfg(feature = "std")]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }

            #[cfg(feature = "std")]
            fn cbrt(self) -> Self {
                <$t>::cbrt(self)
            }

            #[cfg(feature = "std")]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }

            #[cfg(feature = "std")]
            fn exp2(self) -> Self {
                <$t>::exp2(self)
            }

            #[cfg(feature = "std")]
            fn exp_m1(self) -> Self {
                <$t>::exp_m1(self)
            }

            #[cfg(feature = "std")]
            fn log(self, base: Self) -> Self {
                <$t>::log(self, base)
            }

            #[cfg(feature = "std")]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            #[cfg(feature = "std")]
            fn log2(self) -> Self {
                <$t>::log2(self)
            }

            #[cfg(feature = "std")]
            fn log10(self) -> Self {
                <$t>::log10(self)
            }

            #[cfg(feature = "std")]
            fn ln_1p(self) -> Self {
                <$t>::ln_1p(self)
            }

            #[cfg(feature = "std")]
            fn hypot(self, other: Self) -> Self {
                <$t>::hypot(self, other)
            }

            #[cfg(feature = "std")]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }

            #[cfg(feature = "std")]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }

            #[cfg(feature = "std")]
            fn tan(self) -> Self {
                <$t>::tan(self)
            }

            #[cfg(feature = "std")]
            fn asin(self) -> Self {
                <$t>::asin(self)
            }

            #[cfg(feature = "std")]
            fn acos(self) -> Self {
                <$t>::acos(self)
            }

            #[cfg(feature = "std")]
            fn atan(self) -> Self {
                <$t>::atan(self)
            }

            #[cfg(feature = "std")]
            fn atan2(self, other: Self) -> Self {
                <$t>::atan2(self, other)
            }

            #[cfg(feature = "std")]
            fn sin_cos(self) -> (Self, Self) {
                <$t>::sin_cos(self)
            }

            #[cfg(feature = "std")]
            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }

            #[cfg(feature = "std")]
            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }

            #[cfg(feature = "std")]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }

            #[cfg(feature = "std")]
            fn asinh(self) -> Self {
                <$t>::asinh(self)
            }

            #[cfg(feature = "std")]
            fn acosh(self) -> Self {
                <$t>::acosh(self)
            }

            #[cfg(feature = "std")]
            fn atanh(self) -> Self {
                <$t>::atanh(self)
            }
        }
    };
}
impl_primitive!(primitive => f32);
impl_primitive!(primitive => f64);

impl Encoding for f32 {
    const MAX: Self = f32::MAX;
    const MIN: Self = f32::MIN;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const EPSILON: Self = f32::EPSILON;

    fn classify(self) -> FpCategory {
        self.classify()
    }

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.to_bits();
        let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
        let exponent: i16 = ((bits >> 23) & 0xff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x7f_ffff) << 1
        }
        else {
            (bits & 0x7f_ffff) | 0x800_000
        };
        (mantissa as u64, exponent - (127 + 23), sign)
    }
}

impl Encoding for f64 {
    const MAX: Self = f64::MAX;
    const MIN: Self = f64::MIN;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const EPSILON: Self = f64::EPSILON;

    fn classify(self) -> FpCategory {
        self.classify()
    }

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xf_ffff_ffff_ffff) << 1
        }
        else {
            (bits & 0xf_ffff_ffff_ffff) | 0x10_0000_0000_0000
        };
        (mantissa, exponent - (1023 + 52), sign)
    }
}
