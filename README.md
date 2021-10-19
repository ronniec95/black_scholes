# SIMD Black scholes pricer
*Probably* the fastest black scholes pricer in the world.

This tries to be correct for all black scholes calculations including accounting for dividends and american binomial options
Tested against widely known pricers for accuracy

Contains 

    * call
    * put
    * delta
    * rho
    * gamma
    * theta
    * vega
    * implied_vol from price
    * implied_rho from price
    * strike from delta
    * american put

# SIMD

As a thought experiment I used this to see what performance I could eeek out of a i5 6th gen laptop compared to the the naive calculation

For 10,000,000 calls to `bs_single::call()`

* Dev: ~7500ms
* Release: ~1900ms

with `RUST_FLAGS=-C -target_feature=+avx,+fma`
* Release: ~750ms

With the SIMD version
with `RUST_FLAGS=-C -target_feature=+avx,+fma`, without Wide math functions
* Release: ~400ms

with `RUST_FLAGS=-C -target_feature=+avx,+fma`, with Wide math functions
* Release: ~100ms

So around 18x speed up when written with careful CPU consideration

# License
GNU AGPL-3.0-or-later; see [AGPL-v3](LICENSE).
