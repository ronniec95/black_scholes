asdasd SDA #[allow(non_snake_case)]
use crate::bs::OptionDir;
use wide::*;

fn erf_f32x8(x: f32x8) -> f32x8 {
    let t = x.sign_bit();
    let e = x.abs();
    let n: f32x8 = f32x8::splat(0.3275911);
    let a: f32x8 = f32x8::splat(0.254829592);
    let r: f32x8 = f32x8::splat(-0.284496736);
    let i: f32x8 = f32x8::splat(1.421413741);
    let l: f32x8 = f32x8::splat(-1.453152027);
    let d: f32x8 = f32x8::splat(1.061405429);
    let u = f32x8::ONE / e.mul_add(n, f32x8::ONE);
    let eu = u * (-e * e).exp();
    let m = eu.mul_neg_add(
        u.mul_add(u.mul_add(u.mul_add(d.mul_add(u, l), i), r), a),
        f32x8::ONE,
    );
    t.blend(-m, m)
}

fn phi_f32x8(e: f32x8) -> f32x8 {
    let v = f32x8::HALF * (f32x8::ONE + erf_f32x8(e / f32x8::SQRT_2));
    let min: f32x8 = f32x8::splat(-1.0e5);
    let max: f32x8 = f32x8::splat(1.0e5);

    let zero_mask = e.cmp_lt(min);
    let one_mask = e.cmp_gt(max);
    let v = zero_mask.blend(f32x8::ZERO, v);
    let v = one_mask.blend(f32x8::ONE, v);
    v
}

// fn phi_f32x8(e: f32x8) -> f32x8 {
//     const C: f32 = 0.3989422804014330;
//     (0.5 * e * e).exp() * C
// }

pub(crate) fn pdf_f32x8(x: f32x8, mu: f32x8, sigma: f32x8) -> f32x8 {
    const P: f32 = 2.506628274631000502415765284811;
    (-1.0 * (x - mu) * (x - mu) / (2.0 * sigma * sigma)).exp() / (sigma * P)
}

// t - spot
// n - strike
// r - volatility
// a - years_to_expiry
// i - interest
// l - dividend
pub(crate) fn call_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    // Call specific
    let o = phi_f32x8(d1);
    let c = phi_f32x8(d2);
    o * spot * la - c * g
}

pub(crate) fn call_delta_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let la = (-dividend_yield * years_to_expiry).exp();
    // Call specific
    let o = phi_f32x8(d1);
    la * o
}

pub(crate) fn put_delta_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let la = (-dividend_yield * years_to_expiry).exp();
    //let ia = (-risk_free_rate * years_to_expiry).exp();
    // Call specific
    let o = phi_f32x8(-d1);
    -la * o
}

pub(crate) fn gamma_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let v = pdf_f32x8(d1, f32x8::ZERO, f32x8::ONE);
    let la = (-dividend_yield * years_to_expiry).exp();
    -la * v / (spot * volatility * d)
}

pub(crate) fn vega_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let v = pdf_f32x8(d1, f32x8::ZERO, f32x8::ONE);
    let la = (-dividend_yield * years_to_expiry).exp();
    spot * la * v * d
}
pub(crate) fn call_theta_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let v = pdf_f32x8(d1, f32x8::ZERO, f32x8::ONE);
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    // Call specific
    let o = phi_f32x8(d1);
    let c = phi_f32x8(d2);
    -la * spot * v * volatility / (2.0 * d) - risk_free_rate * g * c
        + dividend_yield * spot * la * o
}

pub(crate) fn put_theta_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let v = pdf_f32x8(d1, f32x8::ZERO, f32x8::ONE);
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    let o = phi_f32x8(-d1);
    let c = phi_f32x8(-d2);
    -la * spot * v * volatility / (2.0 * d) + risk_free_rate * g * c
        - dividend_yield * spot * la * o
}
pub(crate) fn call_rho_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    // Call specific
    let c = phi_f32x8(d2);
    g * years_to_expiry * c
}

/// Calculate rho for a wide set of values
pub(crate) fn put_rho_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    let c = phi_f32x8(-d2);
    -g * years_to_expiry * c
}

pub(crate) fn put_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    //let v = pdf_f32x8(d1,f32x8::ZERO,f32x8::ONE);
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    // Put specific
    let o = phi_f32x8(-d1);
    let c = phi_f32x8(-d2);
    c * g - o * spot * la
}

/// Black Scholes single option pricing
pub(crate) fn price_f32x8(
    dir: OptionDir,
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    match dir {
        OptionDir::CALL => call_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
        OptionDir::PUT => put_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
    }
}

/// Delta calculator
pub(crate) fn delta(
    option_dir: OptionDir,
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    match option_dir {
        OptionDir::CALL => call_delta_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
        OptionDir::PUT => put_delta_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
    }
}

pub(crate) fn theta(
    option_dir: OptionDir,
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    match option_dir {
        OptionDir::CALL => call_theta_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
        OptionDir::PUT => put_theta_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
    }
}

pub(crate) fn rho(
    option_dir: OptionDir,
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    match option_dir {
        OptionDir::CALL => call_rho_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
        OptionDir::PUT => put_rho_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
    }
}

pub(crate) fn implied_vol_f32x8(
    option_dir: OptionDir,
    price: f32x8,
    spot: f32x8,
    strike: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let mut volatility = f32x8::splat(0.2);
    let mut count = 0;
    loop {
        let option_value = price_f32x8(
            option_dir,
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        let diff = option_value - price;
        let mask = diff.abs().cmp_lt(f32x8::splat(0.001));
        if mask.all() {
            break;
        }
        let derivative = vega_f32x8(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        let derivative = derivative.max(f32x8::ONE);
        let bump_value = diff / derivative;
        volatility = volatility - bump_value;
        if count > 100 {
            break;
        } else {
            count = count + 1
        }
    }
    let vol_mask = volatility.cmp_gt(f32x8::ZERO);
    volatility = vol_mask.blend(volatility, f32x8::ZERO);
    let price_mask = price.cmp_eq(f32x8::ZERO);
    volatility = price_mask.blend(f32x8::ZERO, volatility);
    volatility
}

pub(crate) fn implied_ir_f32x8(
    option_dir: OptionDir,
    price: f32x8,
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let mut risk_free_rate: f32x8 = 0.05.into();
    let mut count = 0;
    loop {
        let option_value = price_f32x8(
            option_dir,
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );

        let diff = option_value - price;
        let mask: f32x8 = diff.abs();
        let mask: f32x8 = mask.cmp_lt(0.0001);
        if mask.all() {
            break;
        }
        let derivative = rho(
            option_dir,
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );

        let derivative = derivative.max(f32x8::ONE);
        let bump_value = diff / derivative;
        risk_free_rate = risk_free_rate - bump_value;
        // Extremes
        risk_free_rate = risk_free_rate
            .cmp_lt(f32x8::ZERO)
            .blend(f32x8::ZERO, risk_free_rate);
        risk_free_rate = risk_free_rate.cmp_gt(2.0);
        risk_free_rate = risk_free_rate.blend(2.0.into(), risk_free_rate);

        if count > 50 {
            break;
        } else {
            count = count + 1
        }
    }
    let price_mask = price.cmp_eq(f32x8::ZERO);
    risk_free_rate = price_mask.blend(f32x8::ZERO, risk_free_rate);
    risk_free_rate
}

pub(crate) fn call_strike_from_delta_f32x8(
    delta: f32x8,
    spot: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let tsq = years_to_expiry.sqrt();
    let d1 = delta * (risk_free_rate * years_to_expiry).exp();
    let t_0 = -phi_f32x8(d1) * volatility * tsq + (volatility * volatility) / 2.0 * years_to_expiry;
    spot * t_0.abs()
}

pub(crate) fn put_strike_from_delta_f32x8(
    delta: f32x8,
    spot: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    years_to_expiry: f32x8,
) -> f32x8 {
    let tsq = years_to_expiry.sqrt();
    let d1 = delta * (risk_free_rate * years_to_expiry).exp();
    let t_0 = phi_f32x8(d1) * volatility * tsq + (volatility * volatility) / 2.0 * years_to_expiry;
    spot * t_0.abs()
}

pub struct Greek {
    pub pv: f32x8,
    pub delta: f32x8,
    pub theta: f32x8,
    pub gamma: f32x8,
    pub rho: f32x8,
    pub vega: f32x8,
}

pub(crate) fn call_greeks_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> Greek {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    let v = pdf_f32x8(d1, f32x8::ZERO, f32x8::ONE);
    // Call specific
    let o = phi_f32x8(d1);
    let c = phi_f32x8(d2);
    let pv = o * spot * la - c * g;
    let delta = la * o;
    let gamma = -la * v / (spot * volatility * d);
    let vega = spot * la * v * d;
    let theta = -la * spot * v * volatility / (2.0 * d) - risk_free_rate * g * c
        + dividend_yield * spot * la * o;
    let rho = g * years_to_expiry * c;
    Greek {
        pv,
        delta,
        theta,
        gamma,
        rho,
        vega,
    }
}

pub(crate) fn put_greeks_f32x8(
    spot: f32x8,
    strike: f32x8,
    volatility: f32x8,
    risk_free_rate: f32x8,
    dividend_yield: f32x8,
    years_to_expiry: f32x8,
) -> Greek {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = f32x8::ONE / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    //let v = pdf_f32x8(d1,f32x8::ZERO,f32x8::ONE);
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    let v = pdf_f32x8(d1, f32x8::ZERO, f32x8::ONE);

    // Put specific
    let o = phi_f32x8(-d1);
    let c = phi_f32x8(-d2);
    let pv = c * g - o * spot * la;
    let delta = -la * o;
    let gamma = -la * v / (spot * volatility * d);
    let vega = spot * la * v * d;
    let theta = -la * spot * v * volatility / (2.0 * d) + risk_free_rate * g * c
        - dividend_yield * spot * la * o;
    let rho = -g * years_to_expiry * c;
    Greek {
        pv,
        delta,
        theta,
        gamma,
        rho,
        vega,
    }
}

/*
NOTE: TBD

pub fn american_put(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    const BINOMIAL_ITER: f32 = 16f32;

    let delta_t = years_to_expiry / BINOMIAL_ITER;
    let up = (volatility * delta_t.sqrt()).exp();
    let discount_rate = ((risk_free_rate - dividend_yield) * delta_t).exp();
    let d = 1.0 / discount_rate;
    let pu = (discount_rate - d) / (up - d);
    let mut v = vec![0.0; BINOMIAL_ITER as usize];
    for j in 0..(BINOMIAL_ITER as usize) {
        let upow = up.powf(2.0 * j as f32 - BINOMIAL_ITER as f32);
        v[j] = f32::max(0.0, strike - (spot * upow));
    }

    let p0: f32 = 1.0 - pu;
    for j in ((BINOMIAL_ITER as usize + 1) - 1..0).rev() {
        for k in 0..j {
            v[k] = (p0 * v[k] + pu * v[k + 1]) / discount_rate;
        }
    }
    v[0]
}

pub fn american_put_simd(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    const BINOMIAL_ITER: f32 = 16f32;

    let delta_t = years_to_expiry / BINOMIAL_ITER;
    let up = (volatility * delta_t.sqrt()).exp();
    let discount_rate = ((risk_free_rate - dividend_yield) * delta_t).exp();
    let d = 1.0 / discount_rate;
    let pu = (discount_rate - d) / (up - d);
    let mut v = vec![0.0; BINOMIAL_ITER as usize];
    for j in 0..(BINOMIAL_ITER as usize) {
        let upow = up.powf(2.0 * j as f32 - BINOMIAL_ITER as f32);
        v[j] = f32::max(0.0, strike - (spot * upow));
    }

    let p0: f32 = 1.0 - pu;
    for j in ((BINOMIAL_ITER as usize + 1) - 1..0).rev() {
        for k in 0..j {
            v[k] = (p0 * v[k] + pu * v[k + 1]) / discount_rate;
        }
    }
    v[0]
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bs_single::*;
    use bytemuck::cast;
    #[test]
    fn erf_check() {
        for i in (-100..100).step_by(1) {
            let expected = erf(i as f32 / 100.0);
            let actual: [f32; 8] = cast(erf_f32x8((i as f32 / 100.0).into()));
            assert!((actual[0] - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn pdf_check() {
        for i in (-100..100).step_by(1) {
            let expected = pdf(i as f32 / 100.0, 0.0, 1.0);
            let actual: [f32; 8] = cast(pdf_f32x8(
                (i as f32 / 100.0).into(),
                f32x8::ZERO,
                f32x8::ONE,
            ));
            assert!((actual[0] - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn ncd_check() {
        for i in (-100..100).step_by(1) {
            let expected = phi(i as f32 / 100.0);
            let actual: [f32; 8] = cast(phi_f32x8((i as f32 / 100.0).into()));
            assert!((actual[0] - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn ncd_perf() {
        let now = std::time::Instant::now();
        const F: f32 = 0.2;
        for _ in 0..(120 / 8) {
            phi_f32x8(F.into());
        }
        let duration = now.elapsed().as_millis();
        println!("Time take {}ms", duration);
    }

    #[test]
    fn call_check() {
        for i in (50..200).step_by(1) {
            let spot = 60.0;
            let strike = i as f32;
            let years_to_expiry = (i as f32 / 4.0) / 252.0;
            let risk_free_rate = 0.02;
            let volatility = 0.2;
            let dividend_yield = 0.01 + (i as f32 / 100.0);

            // Basic call/put test
            let expected = call(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(call_f32x8(
                f32x8::splat(spot),
                f32x8::splat(strike),
                f32x8::splat(volatility),
                f32x8::splat(risk_free_rate),
                f32x8::splat(dividend_yield),
                f32x8::splat(years_to_expiry),
            ));
            assert!((actual[0] - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn put_check() {
        for i in (50..160).step_by(1) {
            let spot = 150.0;
            let strike = i as f32;
            let years_to_expiry = (i as f32 / 4.0) / 252.0;
            let risk_free_rate = 0.02;
            let volatility = 0.2;
            let dividend_yield = 0.01 + (i as f32 / 100.0);

            // Basic call/put test
            let expected = put(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(put_f32x8(
                f32x8::splat(spot),
                f32x8::splat(strike),
                f32x8::splat(volatility),
                f32x8::splat(risk_free_rate),
                f32x8::splat(dividend_yield),
                f32x8::splat(years_to_expiry),
            ));
            assert!((actual[0] - expected).abs() < 0.0001);
        }
    }

    #[test]
    fn vega_check() {
        for i in (50..90).step_by(1) {
            let spot = 50.0;
            let strike = i as f32;
            let years_to_expiry = 1.0;
            let risk_free_rate = 0.02;
            let volatility = 0.2;
            let dividend_yield = 0.01;

            // Basic call/put test
            let expected = vega(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(vega_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - expected).abs() < 100.0);
        }
    }

    #[test]
    fn gamma_check() {
        for i in (50..90).step_by(1) {
            let spot = 50.0;
            let strike = i as f32;
            let years_to_expiry = 1.0;
            let risk_free_rate = 0.02;
            let volatility = 0.2;
            let dividend_yield = 0.01;

            // Basic call/put test
            let expected = gamma(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(gamma_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - -expected).abs() < 0.00001);
        }
    }
    #[test]
    fn rho_check() {
        for i in (50..90).step_by(1) {
            let spot = 50.0;
            let strike = i as f32;
            let years_to_expiry = 1.0;
            let risk_free_rate = 0.02;
            let volatility = 0.2;
            let dividend_yield = 0.01;

            // Basic call/put test
            let expected = put_rho(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(put_rho_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - expected).abs() < 0.00001);

            let expected = call_rho(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(call_rho_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn delta_check() {
        for i in (50..90).step_by(1) {
            let spot = 50.0;
            let strike = i as f32;
            let years_to_expiry = 1.0;
            let risk_free_rate = 0.02;
            let volatility = 0.2;
            let dividend_yield = 0.01;

            // Basic call/put test
            let expected = put_delta(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(put_delta_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - expected).abs() < 0.00001);

            let expected = call_delta(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(call_delta_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn theta_check() {
        for i in (50..90).step_by(1) {
            let spot = 50.0;
            let strike = i as f32;
            let years_to_expiry = 1.0;
            let risk_free_rate = 0.02;
            let volatility = 0.2;
            let dividend_yield = 0.01;

            // Basic call/put test
            let expected = put_theta(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(put_theta_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - expected).abs() < 0.00001);

            let expected = call_theta(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );

            let actual: [f32; 8] = cast(call_theta_f32x8(
                spot.into(),
                strike.into(),
                volatility.into(),
                risk_free_rate.into(),
                dividend_yield.into(),
                years_to_expiry.into(),
            ));
            assert!((actual[0] - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn check_iv_from_price_f32x8() {
        let spot = 131.0;
        let strike = 115.0;
        let years_to_expiry = 24.0 / 252.0;
        let risk_free_rate = 0.001;
        let volatility = 0.419;
        let dividend_yield = 0.00625 * 12.0;

        // Basic call/put test
        let call_s = put(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        let v: [f32; 8] = cast(implied_vol_f32x8(
            OptionDir::PUT,
            call_s.into(),
            spot.into(),
            strike.into(),
            risk_free_rate.into(),
            dividend_yield.into(),
            years_to_expiry.into(),
        ));
        println!("Put {} IV {:?}", call_s, v);
        assert!((v[0] - volatility).abs() < 0.001);
    }

    #[test]
    fn check_ir_from_price_f32x8() {
        let spot = 100.0;
        let strike = 100.0;
        let years_to_expiry = 24.0 / 252.0;
        let risk_free_rate = 0.02;
        let volatility = 0.18;
        let dividend_yield = 0.00;

        // Basic call/put test
        let call_s = call(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        let v: [f32; 8] = cast(implied_ir_f32x8(
            OptionDir::CALL,
            call_s.into(),
            spot.into(),
            strike.into(),
            volatility.into(),
            dividend_yield.into(),
            years_to_expiry.into(),
        ));
        assert!((v[0] - risk_free_rate).abs() < 0.001);
    }

    #[test]
    fn put_perf() {
        let spot = 150.0.into();
        let strike = 156.0.into();
        let years_to_expiry = (24.0 / 252.0).into();
        let risk_free_rate = 0.02.into();
        let volatility = 0.2.into();
        let dividend_yield = 0.01.into();

        let now = std::time::Instant::now();

        for _ in 0..120 / 8 {
            // Basic call/put test
            let _ = put_f32x8(
                spot,
                strike,
                years_to_expiry,
                risk_free_rate,
                volatility,
                dividend_yield,
            );
        }
        let duration = now.elapsed().as_millis();
        println!("Time take {}ms", duration);
    }

    #[test]
    fn cdf_perf() {
        let now = std::time::Instant::now();
        const F: f32 = 0.2;
        for _ in 0..120 / 8 {
            phi_f32x8(F.into());
        }
        let duration = now.elapsed().as_millis();
        println!("Time take {}ms", duration);
    }

    #[test]
    fn gamma_qcheck() {
        let spot = [110.0, 110.0, 110.0, 110.0, 110.0, 110.0];
        let strike = [120.0, 121.0, 122.0, 123.0, 124.0, 125.0];
        let y2e = 20.0 / 252.0;
        let years_to_expiry = [y2e, y2e, y2e, y2e, y2e, y2e];
        let risk_free_rate = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        let volatility = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15];
        let dividend_yield = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05];

        let gam = put_theta_f32x8(
            f32x8::from(&spot[..]),
            f32x8::from(&strike[..]),
            f32x8::from(&volatility[..]),
            f32x8::from(&risk_free_rate[..]),
            f32x8::from(&dividend_yield[..]),
            f32x8::from(&years_to_expiry[..]),
        );
        dbg!(gam);
    }
}
