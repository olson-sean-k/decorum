use core::num::FpCategory;

use crate::{Encoding, Infinite, Nan, Real};

// TODO: Be consistent with the use of associated constants (use them
//       everywhere).

/// A primitive floating-point value.
///
/// This trait differentiates types that implement floating-point traits but
/// may not be primitive types.
pub trait Primitive: Copy + Sized {}

macro_rules! impl_primitive {
    (primitive => $T:ident) => {
        impl Infinite for $T {
            fn infinity() -> Self {
                $T::INFINITY
            }

            fn neg_infinity() -> Self {
                $T::NEG_INFINITY
            }

            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }

        impl Nan for $T {
            fn nan() -> Self {
                $T::NAN
            }

            fn is_nan(self) -> bool {
                self.is_nan()
            }
        }

        impl Primitive for $T {}

        impl Real for $T {
            const E: Self = core::$T::consts::E;
            const PI: Self = core::$T::consts::PI;
            const FRAC_1_PI: Self = core::$T::consts::FRAC_1_PI;
            const FRAC_2_PI: Self = core::$T::consts::FRAC_2_PI;
            const FRAC_2_SQRT_PI: Self = core::$T::consts::FRAC_2_SQRT_PI;
            const FRAC_PI_2: Self = core::$T::consts::FRAC_PI_2;
            const FRAC_PI_3: Self = core::$T::consts::FRAC_PI_3;
            const FRAC_PI_4: Self = core::$T::consts::FRAC_PI_4;
            const FRAC_PI_6: Self = core::$T::consts::FRAC_PI_6;
            const FRAC_PI_8: Self = core::$T::consts::FRAC_PI_8;
            const SQRT_2: Self = core::$T::consts::SQRT_2;
            const FRAC_1_SQRT_2: Self = core::$T::consts::FRAC_1_SQRT_2;
            const LN_2: Self = core::$T::consts::LN_2;
            const LN_10: Self = core::$T::consts::LN_10;
            const LOG2_E: Self = core::$T::consts::LOG2_E;
            const LOG10_E: Self = core::$T::consts::LOG10_E;

            fn min(self, other: Self) -> Self {
                $T::min(self, other)
            }

            fn max(self, other: Self) -> Self {
                $T::max(self, other)
            }

            fn is_sign_positive(self) -> bool {
                $T::is_sign_positive(self)
            }

            fn is_sign_negative(self) -> bool {
                $T::is_sign_negative(self)
            }

            fn signum(self) -> Self {
                $T::signum(self)
            }

            fn abs(self) -> Self {
                $T::abs(self)
            }

            fn floor(self) -> Self {
                $T::floor(self)
            }

            fn ceil(self) -> Self {
                $T::ceil(self)
            }

            fn round(self) -> Self {
                $T::round(self)
            }

            fn trunc(self) -> Self {
                $T::trunc(self)
            }

            fn fract(self) -> Self {
                $T::fract(self)
            }

            fn recip(self) -> Self {
                $T::recip(self)
            }

            #[cfg(feature = "std")]
            fn mul_add(self, a: Self, b: Self) -> Self {
                $T::mul_add(self, a, b)
            }

            #[cfg(feature = "std")]
            fn powi(self, n: i32) -> Self {
                $T::powi(self, n)
            }

            #[cfg(feature = "std")]
            fn powf(self, n: Self) -> Self {
                $T::powf(self, n)
            }

            #[cfg(feature = "std")]
            fn sqrt(self) -> Self {
                $T::sqrt(self)
            }

            #[cfg(feature = "std")]
            fn cbrt(self) -> Self {
                $T::cbrt(self)
            }

            #[cfg(feature = "std")]
            fn exp(self) -> Self {
                $T::exp(self)
            }

            #[cfg(feature = "std")]
            fn exp2(self) -> Self {
                $T::exp2(self)
            }

            #[cfg(feature = "std")]
            fn exp_m1(self) -> Self {
                $T::exp_m1(self)
            }

            #[cfg(feature = "std")]
            fn log(self, base: Self) -> Self {
                $T::log(self, base)
            }

            #[cfg(feature = "std")]
            fn ln(self) -> Self {
                $T::ln(self)
            }

            #[cfg(feature = "std")]
            fn log2(self) -> Self {
                $T::log2(self)
            }

            #[cfg(feature = "std")]
            fn log10(self) -> Self {
                $T::log10(self)
            }

            #[cfg(feature = "std")]
            fn ln_1p(self) -> Self {
                $T::ln_1p(self)
            }

            #[cfg(feature = "std")]
            fn hypot(self, other: Self) -> Self {
                $T::hypot(self, other)
            }

            #[cfg(feature = "std")]
            fn sin(self) -> Self {
                $T::sin(self)
            }

            #[cfg(feature = "std")]
            fn cos(self) -> Self {
                $T::cos(self)
            }

            #[cfg(feature = "std")]
            fn tan(self) -> Self {
                $T::tan(self)
            }

            #[cfg(feature = "std")]
            fn asin(self) -> Self {
                $T::asin(self)
            }

            #[cfg(feature = "std")]
            fn acos(self) -> Self {
                $T::acos(self)
            }

            #[cfg(feature = "std")]
            fn atan(self) -> Self {
                $T::atan(self)
            }

            #[cfg(feature = "std")]
            fn atan2(self, other: Self) -> Self {
                $T::atan2(self, other)
            }

            #[cfg(feature = "std")]
            fn sin_cos(self) -> (Self, Self) {
                $T::sin_cos(self)
            }

            #[cfg(feature = "std")]
            fn sinh(self) -> Self {
                $T::sinh(self)
            }

            #[cfg(feature = "std")]
            fn cosh(self) -> Self {
                $T::cosh(self)
            }

            #[cfg(feature = "std")]
            fn tanh(self) -> Self {
                $T::tanh(self)
            }

            #[cfg(feature = "std")]
            fn asinh(self) -> Self {
                $T::asinh(self)
            }

            #[cfg(feature = "std")]
            fn acosh(self) -> Self {
                $T::acosh(self)
            }

            #[cfg(feature = "std")]
            fn atanh(self) -> Self {
                $T::atanh(self)
            }
        }
    };
}
impl_primitive!(primitive => f32);
impl_primitive!(primitive => f64);

impl Encoding for f32 {
    fn max_value() -> Self {
        f32::MAX
    }

    fn min_value() -> Self {
        f32::MIN
    }

    fn min_positive_value() -> Self {
        f32::MIN_POSITIVE
    }

    fn epsilon() -> Self {
        f32::EPSILON
    }

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
    fn max_value() -> Self {
        f64::MAX
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn min_positive_value() -> Self {
        f64::MIN_POSITIVE
    }

    fn epsilon() -> Self {
        f64::EPSILON
    }

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
