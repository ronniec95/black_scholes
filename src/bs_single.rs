use crate::bs::OptionDir;
const C: f32 = 0.3989422804014330;

pub(crate) fn erf(x: f32) -> f32 {
    let t = x.signum();
    //if x > 0.0 { 1.0 } else { -1.0 };
    let e = x.abs();
    const N: f32 = 0.3275911;
    const A: f32 = 0.254829592;
    const R: f32 = -0.284496736;
    const I: f32 = 1.421413741;
    const L: f32 = -1.453152027;
    const D: f32 = 1.061405429;
    let u = 1.0 / (1.0 + N * e);
    let m = 1.0 - ((((D * u + L) * u + I) * u + R) * u + A) * u * (-e * e).exp();
    t * m
}

pub(crate) fn ncd(e: f32) -> f32 {
    if e < -1.0e5 {
        0.0
    } else if e > 1.0e5 {
        1.0
    } else {
        0.5 * (1.0 + erf(e / 2.0f32.sqrt()))
    }
}

pub(crate) fn npd(e: f32) -> f32 {
    C * (0.5 * e * e).exp()
}

// t - spot
// n - strike
// r - volatility
// a - years_to_expiry
// i - interest
// l - dividend
pub fn call(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let years_sqrt = years_to_expiry.sqrt();
    let rd = volatility * years_sqrt;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    //let v = npd(d1);
    let dividend_years_exp = (-dividend_yield * years_to_expiry).exp();
    let risk_free_years_exp = (-risk_free_rate * years_to_expiry).exp();
    let strike_x_risk_free_years_exp = strike * risk_free_years_exp;
    // Call specific
    let ncd_d1 = ncd(d1);
    let ncd_d2 = ncd(d2);
    ncd_d1 * spot * dividend_years_exp - ncd_d2 * strike_x_risk_free_years_exp
}

pub fn call_delta(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    //let d2 = d1 - rd;
    //let v = npd(d1);
    let la = (-dividend_yield * years_to_expiry).exp();
    //let ia = (-risk_free_rate * years_to_expiry).exp();
    //let g = strike * ia;
    // Call specific
    let o = ncd(d1);
    //let c = ncd(d2);
    la * o
}

pub fn put_delta(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    //let d2 = d1 - rd;
    //let v = npd(d1);
    let la = (-dividend_yield * years_to_expiry).exp();
    //let ia = (-risk_free_rate * years_to_expiry).exp();
    //let g = strike * ia;
    // Call specific
    let o = ncd(-d1);
    //let c = ncd(-d2);
    -la * o
}

pub fn gamma(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let v = npd(d1);
    let la = (-dividend_yield * years_to_expiry).exp();
    -la * v / (spot * volatility * d)
}

pub fn vega(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let v = npd(d1);
    let la = (-dividend_yield * years_to_expiry).exp();
    spot * la * v * d
}

pub fn call_theta(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let v = npd(d1);
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    // Call specific
    let o = ncd(d1);
    let c = ncd(d2);
    -la * spot * v * volatility / (2.0 * d) - risk_free_rate * g * c
        + dividend_yield * spot * la * o
}

pub fn put_theta(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let v = npd(d1);
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    let o = ncd(-d1);
    let c = ncd(-d2);
    -la * spot * v * volatility / (2.0 * d) + risk_free_rate * g * c
        - dividend_yield * spot * la * o
}

pub fn call_rho(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    // Call specific
    let c = ncd(d2);
    g * years_to_expiry * c
}

pub fn put_rho(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    //let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    let c = ncd(-d2);
    -g * years_to_expiry * c
}

pub fn put(
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let d = years_to_expiry.sqrt();
    let rd = volatility * d;
    let vs2 = (volatility * volatility) / 2.0;
    let ssln = (spot / strike).ln();
    let il = risk_free_rate - dividend_yield;
    let d1 = 1.0 / rd * (ssln + (il + vs2) * years_to_expiry);
    let d2 = d1 - rd;
    //let v = npd(d1);
    let la = (-dividend_yield * years_to_expiry).exp();
    let ia = (-risk_free_rate * years_to_expiry).exp();
    let g = strike * ia;
    // Put specific
    let o = ncd(-d1);
    let c = ncd(-d2);
    c * g - o * spot * la
}

// fn cdf(x: f32) -> f32 {
//     let exp = (-x * x * 0.5).exp();
//     let sign = if x >= 0.0 { 1.0 } else { -1.0 };
//     let t = 1.0f32 / (1.0 + sign * P * x);
//     let factor = t * (t * (t * (t * B5 + B4) + B3) + B2) + B1;
//     let cv = C * exp * t * factor;
//     if x >= 0.0 {
//         1.0 - cv
//     } else {
//         cv
//     }
// }

/// Black Scholes single option pricing
pub fn bs_price(
    dir: OptionDir,
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    match dir {
        OptionDir::CALL => call(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
        OptionDir::PUT => put(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
    }
}

/// Single delta calculator
pub fn delta(
    option_dir: OptionDir,
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    match option_dir {
        OptionDir::CALL => call_delta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
        OptionDir::PUT => put_delta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
    }
}

/// Single theta calculator
pub fn theta(
    option_dir: OptionDir,
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    match option_dir {
        OptionDir::CALL => call_theta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
        OptionDir::PUT => put_theta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
    }
}

/// Single rho calculator
pub fn rho(
    option_dir: OptionDir,
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    match option_dir {
        OptionDir::CALL => call_rho(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
        OptionDir::PUT => put_rho(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ),
    }
}

/// Implied vol from price
pub fn implied_vol(
    option_dir: OptionDir,
    price: f32,
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
) -> f32 {
    let mut volatility = 0.2f32;
    loop {
        let option_value = bs_price(
            option_dir,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let derivative = vega(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let diff = option_value - price;
        if diff.abs() < 0.0001 {
            break;
        }
        if diff > 0.0 {
            volatility = volatility - diff / derivative
        } else {
            volatility = volatility + diff / derivative
        }
    }
    volatility
}

/// Implied interest rate from price
pub fn implied_interest_rate(
    option_dir: OptionDir,
    price: f32,
    spot: f32,
    strike: f32,
    years_to_expiry: f32,
    volatility: f32,
    dividend_yield: f32,
) -> f32 {
    let mut risk_free_rate = 0.05f32;
    loop {
        let option_value = bs_price(
            option_dir,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let derivative = rho(
            option_dir,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );

        let diff = option_value - price;
        if diff.abs() < 0.0001 {
            break;
        }
        if diff > 0.0 {
            risk_free_rate = risk_free_rate - diff / derivative
        } else {
            risk_free_rate = risk_free_rate + diff / derivative
        }
        // Extremes
        if risk_free_rate < 0.0 {
            risk_free_rate = 0.0;
            break;
        }
        if risk_free_rate > 2.0 {
            risk_free_rate = 2.0;
            break;
        }
    }
    risk_free_rate
}

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

pub fn call_strike_from_delta(
    delta: f32,
    spot: f32,
    risk_free_rate: f32,
    volatility: f32,
    years_to_expiry: f32,
) -> f32 {
    let tsq = years_to_expiry.sqrt();
    let d1 = delta * (risk_free_rate * years_to_expiry).exp();
    let t_0 = -ncd(d1) * volatility * tsq + (volatility * volatility) / 2.0 * years_to_expiry;
    spot * t_0.abs()
}

pub fn put_strike_from_delta(
    delta: f32,
    spot: f32,
    risk_free_rate: f32,
    volatility: f32,
    years_to_expiry: f32,
) -> f32 {
    let tsq = years_to_expiry.sqrt();
    let d1 = delta * (risk_free_rate * years_to_expiry).exp();
    let t_0 = ncd(d1) * volatility * tsq + (volatility * volatility) / 2.0 * years_to_expiry;
    spot * t_0.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_tests() {
        let spot = 100.0;
        let strike = 100.0;
        let years_to_expiry = 24.0 / 252.0;
        let risk_free_rate = 0.02;
        let volatility = 0.2;
        let dividend_yield = 0.00;

        // Basic call/put test
        let call_s = call(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let put_s = put(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        assert!((call_s - 2.5559196).abs() < 0.00001);
        assert!((put_s - 2.3656273).abs() < 0.00001);

        // With dividends
        let dividend_yield = 0.05;
        let call_s = call(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let put_s = put(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        assert!((call_s - 2.3140182).abs() < 0.00001);
        assert!((put_s - 2.5987892).abs() < 0.00001);

        let call_d = call_delta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );

        let put_d = put_delta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        assert!((call_d - 0.4914985).abs() < 0.00001);
        assert!((put_d - -0.50375086).abs() < 0.00001);

        let vega = vega(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        assert!((vega - 12.254598).abs() < 0.00001);

        let gamma = gamma(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        assert!((gamma - -0.064336635).abs() < 0.00001);

        let call_t = call_theta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let put_t = put_theta(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        assert!((call_t - -11.346551).abs() < 0.00001);
        assert!((put_t - -14.326603).abs() < 0.00001);

        let call_r = call_rho(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let put_r = put_rho(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );

        assert!((call_r - 4.4605556).abs() < 0.00001);
        assert!((put_r - -5.045131).abs() < 0.00001);
    }

    #[test]
    fn cdf_f32_single() {
        let now = std::time::Instant::now();
        let f = 0.2f32;
        let mut x = 0.0;
        for _ in 0..12_000_000 {
            x += ncd(f);
        }
        let duration = now.elapsed().as_millis();
        println!("Time take {}ms {}", duration, x);
        assert!(x > 1.0);
    }

    #[test]
    fn implied_vol_from_price() {
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
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let v = implied_vol(
            OptionDir::CALL,
            call_s,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            dividend_yield,
        );
        assert!((v - volatility).abs() < 0.001);
    }

    #[test]
    fn implied_rho_from_price() {
        let spot = 100.0;
        let strike = 100.0;
        let years_to_expiry = 24.0 / 252.0;
        let risk_free_rate = 0.02;
        let volatility = 0.18;
        let dividend_yield = 0.00;

        // Basic call/put test
        let price = call(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );
        let v = implied_interest_rate(
            OptionDir::CALL,
            price,
            spot,
            strike,
            years_to_expiry,
            volatility,
            dividend_yield,
        );
        assert!((v - risk_free_rate).abs() < 0.001);
    }

    #[test]
    fn perf_single() {
        let spot = 150.0;
        let strike = 156.0;
        let years_to_expiry = 24.0 / 252.0;
        let risk_free_rate = 0.02;
        let volatility = 0.2;
        let dividend_yield = 0.01;

        let now = std::time::Instant::now();

        for _ in 0..10_000_000 {
            // Basic call/put test
            let _ = put(
                spot,
                strike,
                years_to_expiry,
                risk_free_rate,
                volatility,
                dividend_yield,
            );
        }
        let duration = now.elapsed().as_millis();
        println!(
            "Time take to run 10,000,000 bs calculations was {}ms",
            duration
        );
    }

    #[test]
    fn simple_demo() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            c.push(a[i] + b[i]);
        }
        println!("Sum is {:?}", c);
    }
}
