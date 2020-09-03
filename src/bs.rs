///! Public interface to serial and vectorised version of black scholes pricing and related functionality
use crate::bs_f32x8_;
use bytemuck::cast;
use wide::*;

/// A container for all the greeks
#[derive(Debug)]
pub struct Greeks {
    pub pv: Vec<f32>,
    pub delta: Vec<f32>,
    pub theta: Vec<f32>,
    pub gamma: Vec<f32>,
    pub rho: Vec<f32>,
    pub vega: Vec<f32>,
}

/// Specify whether an option is put or call
#[derive(PartialEq, Debug, Copy, Clone, PartialOrd)]
pub enum OptionDir {
    CALL = 0,
    PUT = 1,
}

/// Black Scholes call pricing. The results are at the same index as the inputs
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn bs_call(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    // Make everything f32x8
    let mut res = Vec::with_capacity(1);
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::call_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Black Scholes put pricing for arrays. The results are at the same index as the inputs
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn bs_put(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::put_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Put delta
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn put_delta(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::delta(
            OptionDir::PUT,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Call delta
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn call_delta(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());

    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::delta(
            OptionDir::CALL,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Vega - is the same if call or put
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn vega(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());

    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::vega_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Gamma - is the same if call or put
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn gamma(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());

    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::gamma_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Call Theta
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn call_theta(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());

    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::theta(
            OptionDir::CALL,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Put Theta
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
/// The calculate for the put theta seems to have a number of different implementations. Bug fixes welcome
pub fn put_theta(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::theta(
            OptionDir::PUT,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }

    res
}

/// Call rho
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn call_rho(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::call_rho_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Put rho
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
pub fn put_rho(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut res = Vec::with_capacity(spot.len());
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let price: [f32; 8] = cast(bs_f32x8_::put_rho_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        ));
        res.extend(&price);
    }
    res
}

/// Calculate all the greeks for put options in a single step
/// This is more efficient than calculating the values individually, infact, if you need more than
/// a two greeks it's faster to use this than the individual pricers
/// However be aware the memory allocation cost for the results is the bottleneck and could slow things down
/// if you do not have a large L1/L2 cache.
pub fn call_greeks(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Greeks {
    let mut delta_res = Vec::with_capacity(spot.len());
    let mut vega_res = Vec::with_capacity(spot.len());
    let mut theta_res = Vec::with_capacity(spot.len());
    let mut gamma_res = Vec::with_capacity(spot.len());
    let mut rho_res = Vec::with_capacity(spot.len());
    let mut pv_res = Vec::with_capacity(spot.len());
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let greek = bs_f32x8_::call_greeks_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );

        let pv: [f32; 8] = cast(greek.pv);
        let delta: [f32; 8] = cast(greek.delta);
        let theta: [f32; 8] = cast(greek.theta);
        let gamma: [f32; 8] = cast(greek.gamma);
        let rho: [f32; 8] = cast(greek.rho);
        let vega: [f32; 8] = cast(greek.vega);
        delta_res.extend(&delta);
        vega_res.extend(&vega);
        theta_res.extend(&theta);
        gamma_res.extend(&gamma);
        rho_res.extend(&rho);
        pv_res.extend(&pv);
    }
    Greeks {
        pv: pv_res,
        delta: delta_res,
        vega: vega_res,
        gamma: gamma_res,
        theta: theta_res,
        rho: rho_res,
    }
}

/// Calculate all the greeks for put options in a single step
/// This is more efficient than calculating the values individually, infact, if you need more than
/// a two greeks it's faster to use this than the individual pricers
/// However be aware the memory allocation cost for the results is the bottleneck and could slow things down
/// if you do not have a large L1/L2 cache.
pub fn put_greeks(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Greeks {
    let mut delta_res = Vec::with_capacity(spot.len());
    let mut vega_res = Vec::with_capacity(spot.len());
    let mut theta_res = Vec::with_capacity(spot.len());
    let mut gamma_res = Vec::with_capacity(spot.len());
    let mut rho_res = Vec::with_capacity(spot.len());
    let mut pv_res = Vec::with_capacity(spot.len());
    for i in (0..spot.len()).step_by(8) {
        let spot = f32x8::from(&spot[i..i + 8]);
        let strike = f32x8::from(&strike[i..i + 8]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..i + 8]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..i + 8]);
        let volatility = f32x8::from(&volatility[i..i + 8]);
        let dividend_yield = f32x8::from(&dividend_yield[i..i + 8]);
        let greek = bs_f32x8_::put_greeks_f32x8(
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        );

        let pv: [f32; 8] = cast(greek.pv);
        let delta: [f32; 8] = cast(greek.delta);
        let theta: [f32; 8] = cast(greek.theta);
        let gamma: [f32; 8] = cast(greek.gamma);
        let rho: [f32; 8] = cast(greek.rho);
        let vega: [f32; 8] = cast(greek.vega);
        delta_res.extend(&delta);
        vega_res.extend(&vega);
        theta_res.extend(&theta);
        gamma_res.extend(&gamma);
        rho_res.extend(&rho);
        pv_res.extend(&pv);
    }
    Greeks {
        pv: pv_res,
        delta: delta_res,
        vega: vega_res,
        theta: theta_res,
        gamma: gamma_res,
        rho: rho_res,
    }
}

/// Calculate call implied vol from an option price
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
/// Note this is an iterative calculation as there is no closed form solution. It exits when all the values in the array have
/// reached a stable number
pub fn call_implied_vol(
    price: &[f32],
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut irvol = Vec::with_capacity(price.len());
    for i in (0..spot.len()).step_by(8) {
        let price = f32x8::from(&price[i..]);
        let spot = f32x8::from(&spot[i..]);
        let strike = f32x8::from(&strike[i..]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..]);
        let dividend_yield = f32x8::from(&dividend_yield[i..]);
        let res: [f32; 8] = cast(bs_f32x8_::implied_vol_f32x8(
            OptionDir::CALL,
            price,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            dividend_yield,
        ));
        irvol.extend(&res);
    }
    irvol
}

/// Calculate put implied vol from an option price
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
/// Note this is an iterative calculation as there is no closed form solution. It exits when all the values in the array have
/// reached a stable number
pub fn put_implied_vol(
    price: &[f32],
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut irvol = Vec::with_capacity(price.len());
    for i in (0..spot.len()).step_by(8) {
        let price = f32x8::from(&price[i..]);
        let spot = f32x8::from(&spot[i..]);
        let strike = f32x8::from(&strike[i..]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..]);
        let dividend_yield = f32x8::from(&dividend_yield[i..]);
        let res: [f32; 8] = cast(bs_f32x8_::implied_vol_f32x8(
            OptionDir::PUT,
            price,
            spot,
            strike,
            years_to_expiry,
            risk_free_rate,
            dividend_yield,
        ));
        irvol.extend(&res);
    }
    irvol
}

/// Calculate implied interest rate from an call option price
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
/// Note this is an iterative calculation as there is no closed form solution. It exits when all the values in the array have
/// reached a stable number
pub fn call_implied_interest_rate(
    price: &[f32],
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut irres = Vec::with_capacity(price.len());
    for i in (0..spot.len()).step_by(8) {
        let price = f32x8::from(&price[i..]);
        let spot = f32x8::from(&spot[i..]);
        let strike = f32x8::from(&strike[i..]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..]);
        let volatility = f32x8::from(&volatility[i..]);
        let dividend_yield = f32x8::from(&dividend_yield[i..]);
        let res: [f32; 8] = cast(bs_f32x8_::implied_ir_f32x8(
            OptionDir::CALL,
            price,
            spot,
            strike,
            years_to_expiry,
            volatility,
            dividend_yield,
        ));
        irres.extend(&res);
    }
    irres
}

/// Calculate implied interest rate from an put option price
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
/// Note this is an iterative calculation as there is no closed form solution. It exits when all the values in the array have
/// reached a stable number
pub fn put_implied_interest_rate(
    price: &[f32],
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    let mut irres = Vec::with_capacity(price.len());
    for i in (0..spot.len()).step_by(8) {
        let price = f32x8::from(&price[i..]);
        let spot = f32x8::from(&spot[i..]);
        let strike = f32x8::from(&strike[i..]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..]);
        let volatility = f32x8::from(&volatility[i..]);
        let dividend_yield = f32x8::from(&dividend_yield[i..]);
        let res: [f32; 8] = cast(bs_f32x8_::implied_ir_f32x8(
            OptionDir::PUT,
            price,
            spot,
            strike,
            years_to_expiry,
            volatility,
            dividend_yield,
        ));
        irres.extend(&res);
    }
    irres
}

/*
/// American put using binomial pricing
/// There's two versions of this function; this one calculates multiple american puts in parallel
/// Years to expiry should be expressed as a f32 such as 20 days is 20/252 = 0.79
/// Risk free rate, volatility and dividend yield expressed as f32 with 1.0 = 100%. 0.2 = 20% etc
/// Note this is an iterative calculation as there is no closed form solution. It exits when all the values in the array have
/// reached a stable number
pub fn american_put(
    spot: &[f32],
    strike: &[f32],
    years_to_expiry: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    dividend_yield: &[f32],
) -> Vec<f32> {
    /*
    const N: usize = 10;
    let spot = spot * &((risk_free_rate - dividend_yield) * years_to_expiry).mapv(f32::exp);
    let delta_t = years_to_expiry / N as f32;
    let up = (volatility * &delta_t.mapv(f32::sqrt)).mapv(f32::exp);
    let mut p = Array2::from_shape_fn([spot.len(), N + 1], |(j, i)| {
        let v = strike[j] - spot[j] * up[j].powf(2.0 * i as f32 - N as f32);
        if v <= 0.0 {
            0.0
        } else {
            v
        }
    });
    const P0: f32 = 0.5;
    const P1: f32 = 1.0 - P0;
    let v = p
        .outer_iter_mut()
        .map(|mut p| {
            for j in 0..N {
                let j = N - 1 - j;
                for i in 0..=j {
                    // Remove the dependency here
                    let p0 = P0 * p[i];
                    let p1 = P1 * p[i];
                    p[i] = p0 + p1;
                }
            }
            p[0]
        })
        .collect::<Array1<f32>>();
    v * (-risk_free_rate * years_to_expiry).mapv(f32::exp)
    */
    vec![]
    //    Array1::zeros([0])
}
*/
/// Calculate the call strike from delta value given
pub fn call_strike_from_delta(
    delta: &[f32],
    spot: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    years_to_expiry: &[f32],
) -> Vec<f32> {
    let mut cs = Vec::with_capacity(delta.len());
    for i in (0..spot.len()).step_by(8) {
        let delta = f32x8::from(&delta[i..]);
        let spot = f32x8::from(&spot[i..]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..]);
        let volatility = f32x8::from(&volatility[i..]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..]);
        let res: [f32; 8] = cast(bs_f32x8_::call_strike_from_delta_f32x8(
            delta,
            spot,
            risk_free_rate,
            volatility,
            years_to_expiry,
        ));
        cs.extend(&res);
    }
    cs
}

/// Calculate the call strike from delta value given
pub fn put_strike_from_delta(
    delta: &[f32],
    spot: &[f32],
    risk_free_rate: &[f32],
    volatility: &[f32],
    years_to_expiry: &[f32],
) -> Vec<f32> {
    let mut ps = Vec::with_capacity(delta.len());
    for i in (0..spot.len()).step_by(8) {
        let delta = f32x8::from(&delta[i..]);
        let spot = f32x8::from(&spot[i..]);
        let risk_free_rate = f32x8::from(&risk_free_rate[i..]);
        let volatility = f32x8::from(&volatility[i..]);
        let years_to_expiry = f32x8::from(&years_to_expiry[i..]);
        let res: [f32; 8] = cast(bs_f32x8_::put_strike_from_delta_f32x8(
            delta,
            spot,
            risk_free_rate,
            volatility,
            years_to_expiry,
        ));
        ps.extend(&res);
    }
    ps
}
