use crate::bs::OptionDir;

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

pub(crate) fn phi(e: f32) -> f32 {
    if e < -1.0e5 {
        0.0
    } else if e > 1.0e5 {
        1.0
    } else {
        0.5 * (1.0 + erf(e / 2.0f32.sqrt()))
    }
}

pub(crate) fn pdf(x: f32, mu: f32, sigma: f32) -> f32 {
    (-1.0 * (x - mu) * (x - mu) / (2.0 * sigma * sigma)).exp() / (sigma * (2.0 * 3.14159f32).sqrt())
}

fn get_d1(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    expiry: f32,
) -> f32 {
    ((spot / strike).ln()
        + (risk_free_rate - dividend_yield + volatility * volatility / 2.0) * expiry)
        / (volatility * expiry.sqrt())
}

fn get_d2(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    expiry: f32,
) -> f32 {
    ((spot / strike).ln()
        + (risk_free_rate - dividend_yield - volatility * volatility / 2.0) * expiry)
        / (volatility * expiry.sqrt())
}

/// Calculate call price of an option with dividends
pub(crate) fn call(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd1 = phi(d1);
    let d2 = get_d2(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd2 = phi(d2);
    let result = spot * (-dividend_yield * years_to_expiry).exp() * nd1
        - strike * (-risk_free_rate * years_to_expiry).exp() * nd2;
    result
}

/// Calculate call delta of an option with dividends
pub(crate) fn call_delta(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd1 = phi(d1);
    (-dividend_yield * years_to_expiry).exp() * nd1
}

/// Calculate put delta of an option with dividends
pub(crate) fn put_delta(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd1 = phi(d1);
    (-dividend_yield * years_to_expiry).exp() * (nd1 - 1.0)
}

/// Calculate gamma of an option with dividends
pub fn gamma(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd1 = pdf(d1, 0.0, 1.0);
    (-dividend_yield * years_to_expiry).exp() * nd1 / (spot * volatility * years_to_expiry.sqrt())
}

/// Calculate vega of an option with dividends
pub fn vega(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd1 = pdf(d1, 0.0, 1.0);
    (-dividend_yield * years_to_expiry).exp() * nd1 * (spot * years_to_expiry.sqrt())
}

/// Calculate call theta of an option with dividends
pub(crate) fn call_theta(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let d2 = get_d2(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );

    let result = -(-dividend_yield * years_to_expiry).exp() * spot * pdf(d1, 0.0, 1.0) * volatility
        / (2.0 * (years_to_expiry).sqrt());
    let result_part1 = risk_free_rate * strike * (-risk_free_rate * years_to_expiry).exp();
    let result_part2 = dividend_yield * spot * (-dividend_yield * years_to_expiry).exp();
    result - (result_part1 * phi(d2)) + (result_part2 * phi(d1))
}

/// Calculate put theta of an option with dividends
pub(crate) fn put_theta(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let d2 = get_d2(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );

    let result = -(-dividend_yield * years_to_expiry).exp() * spot * pdf(d1, 0.0, 1.0) * volatility
        / (2.0 * (years_to_expiry).sqrt());
    let result_part1 = risk_free_rate * strike * (-risk_free_rate * years_to_expiry).exp();
    let result_part2 = dividend_yield * spot * (-dividend_yield * years_to_expiry).exp();
    result + (result_part1 * phi(-d2)) - (result_part2 * phi(-d1))
}

/// Calculate call rho of an option with dividends
pub(crate) fn call_rho(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d2 = get_d2(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd2 = phi(d2);
    strike * years_to_expiry * (-risk_free_rate * years_to_expiry).exp() * nd2
}

/// Calculate put rho of an option with dividends
pub(crate) fn put_rho(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d2 = get_d2(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd2 = phi(d2);
    strike * years_to_expiry * (-risk_free_rate * years_to_expiry).exp() * (nd2 - 1.0)
}

/// Calculate put price an option with dividends
pub(crate) fn put(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    let d1 = get_d1(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd1 = phi(d1);
    let d2 = get_d2(
        spot,
        strike,
        volatility,
        risk_free_rate,
        dividend_yield,
        years_to_expiry,
    );
    let nd2 = phi(d2);
    let result = strike * (-risk_free_rate * years_to_expiry).exp() * (1.0 - nd2)
        - spot * (-dividend_yield * years_to_expiry).exp() * (1.0 - nd1);
    result
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
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    match dir {
        OptionDir::CALL => call(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
        OptionDir::PUT => put(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
    }
}

/// Single delta calculator
pub fn delta(
    option_dir: OptionDir,
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
) -> f32 {
    match option_dir {
        OptionDir::CALL => call_delta(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
        OptionDir::PUT => put_delta(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
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
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        ),
        OptionDir::PUT => put_rho(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
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
    let mut count = 0;
    loop {
        let option_value = bs_price(
            option_dir,
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        dbg!(&option_value);
        let derivative = vega(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        dbg!(&derivative);

        let diff = (option_value - price);
        if diff.abs() < 0.0001 {
            break;
        }
        dbg!(&diff);

        if diff > 0.0 {
            volatility = volatility - (diff / derivative).abs()
        } else {
            volatility = volatility + (diff / derivative).abs()
        }
        dbg!(volatility);
        if count > 50 {
            break;
        } else {
            count = count + 1;
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
    let mut count = 0;
    loop {
        let option_value = bs_price(
            option_dir,
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
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
            risk_free_rate = risk_free_rate - (diff / derivative).abs()
        } else {
            risk_free_rate = risk_free_rate + (diff / derivative).abs()
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
        if count > 50 {
            break;
        } else {
            count = count + 1;
        }
    }
    risk_free_rate
}

/// Binomial put pricing
pub fn american_put(
    spot: f32,
    strike: f32,
    volatility: f32,
    risk_free_rate: f32,
    dividend_yield: f32,
    years_to_expiry: f32,
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

/// Calculate the call strike from a delta value
pub fn call_strike_from_delta(
    delta: f32,
    spot: f32,
    risk_free_rate: f32,
    volatility: f32,
    years_to_expiry: f32,
) -> f32 {
    let tsq = years_to_expiry.sqrt();
    let d1 = delta * (risk_free_rate * years_to_expiry).exp();
    let t_0 = -phi(d1) * volatility * tsq + (volatility * volatility) / 2.0 * years_to_expiry;
    spot * t_0.abs()
}

/// Calculate the put strike from a delta value
pub fn put_strike_from_delta(
    delta: f32,
    spot: f32,
    risk_free_rate: f32,
    volatility: f32,
    years_to_expiry: f32,
) -> f32 {
    let tsq = years_to_expiry.sqrt();
    let d1 = delta * (risk_free_rate * years_to_expiry).exp();
    let t_0 = phi(d1) * volatility * tsq + (volatility * volatility) / 2.0 * years_to_expiry;
    spot * t_0.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d1d2() {
        for &i in [120.0f32, 121.0, 122.0, 123.0, 124.0, 125.0].iter() {
            // dbg!(get_d1(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
            // dbg!(get_d2(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
            // dbg!(call(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
            // dbg!(put(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
            // dbg!(call_rho(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
            // dbg!(put_rho(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
            dbg!(call_theta(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
            dbg!(put_theta(110.0, i, 0.15, 0.01, 0.05, 20.0 / 252.0));
        }
    }
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
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        let put_s = put(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        assert!((call_s - 2.5559196).abs() < 0.00001);
        assert!((put_s - 2.3656273).abs() < 0.00001);

        // With dividends
        let dividend_yield = 0.05;
        let call_s = call(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        let put_s = put(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        assert!((call_s - 2.3140182).abs() < 0.00001);
        assert!((put_s - 2.5987892).abs() < 0.00001);

        let call_d = call_delta(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );

        let put_d = put_delta(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        assert!((call_d - 0.4914985).abs() < 0.00001);
        assert!((put_d - -0.50375086).abs() < 0.00001);

        let vega = vega(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        assert!((vega - 12.251685).abs() < 0.00001);

        let gamma = gamma(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );

        assert!((gamma - 0.06432134).abs() < 0.00001);

        let call_t = call_theta(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );

        let put_t = put_theta(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );

        assert!((call_t - -11.343492).abs() < 0.00001);
        assert!((put_t - -14.323544).abs() < 0.00001);

        let call_r = call_rho(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );
        let put_r = put_rho(
            spot,
            strike,
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
        );

        assert!((call_r - 4.4605556).abs() < 0.00001);
        assert!((put_r - -5.045131).abs() < 0.00001);
    }

    #[test]
    fn cdf_f32_single() {
        let now = std::time::Instant::now();
        let f = 0.2f32;
        let mut x = 0.0;
        for _ in 0..100 {
            x += phi(f);
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
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
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
            volatility,
            risk_free_rate,
            dividend_yield,
            years_to_expiry,
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

        for _ in 0..100 {
            // Basic call/put test
            let _ = put(
                spot,
                strike,
                volatility,
                risk_free_rate,
                dividend_yield,
                years_to_expiry,
            );
        }
        let duration = now.elapsed().as_millis();
        println!(
            "Time take to run 10,000,000 bs calculations was {}ms",
            duration
        );
    }

    #[test]
    fn call_single() {
        let spot = [110.0, 110.0, 110.0, 110.0, 110.0];
        let strike = [120.0, 120.0, 120.0, 120.0, 120.0];
        let y2e = 25.0 / 252.0;
        let years_to_expiry = [y2e, y2e, y2e, y2e, y2e];
        let risk_free_rate = [0.02, 0.02, 0.02, 0.02, 0.02];
        let volatility = [0.15, 0.16, 0.17, 0.18, 0.19];
        let dividend_yield = [0.05, 0.05, 0.05, 0.05, 0.05];

        for v in &volatility {
            let expected = call(
                spot[0],
                strike[0],
                *v as f32,
                risk_free_rate[0],
                dividend_yield[0],
                years_to_expiry[0],
            );
            dbg!(expected);

            let gam = gamma(
                spot[0],
                strike[0],
                *v as f32,
                risk_free_rate[0],
                dividend_yield[0],
                years_to_expiry[0],
            );
            dbg!(gam);
        }
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

    #[test]
    fn iv_test2() {
        let spot = 180.7;
        let strike = 180.7;
        let years_to_expiry = 9.0 / 252.0;
        let risk_free_rate = 0.001;
        let dividend_yield = 0.00;

        let v = implied_vol(
            OptionDir::PUT,
            5.15,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            dividend_yield,
        );
        dbg!(&v);
    }
}
