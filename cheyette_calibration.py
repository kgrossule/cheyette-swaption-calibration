import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ==========================================
# 1. MARKET DATA SETUP & YIELD CURVE ENGINE
# ==========================================

estr_data_str = """1 WK 1.92806
2 WK    1.92995
1 MO    1.92895
2 MO    1.93125
3 MO    1.93316
4 MO    1.93304
5 MO    1.93248
6 MO    1.93201
7 MO    1.93079
8 MO    1.93173
9 MO    1.93326
10 MO   1.93637
11 MO   1.94039
1 YR    1.94484
18 MO   1.97818
2 YR    2.03094
3 YR    2.13976
4 YR    2.23633
5 YR    2.31891
6 YR    2.39369
7 YR    2.46289
8 YR    2.52978
9 YR    2.59208
10 YR   2.65008
11 YR   2.70477
12 YR   2.75685
15 YR   2.87981"""


def parse_yield_curve(data_str):
    """Parses raw €STR market quotes into maturities (in years) and zero rates."""
    times, rates = [], []
    for line in data_str.split('\n'):
        if not line.strip():
            continue
        parts = line.replace('\t', ' ').split()
        val = float(parts[0])
        unit = parts[1]
        if unit == 'WK':
            t = val * 7.0 / 365.0
        elif unit == 'MO':
            t = val * 30.0 / 365.0
        elif unit == 'YR':
            t = val
        else:
            t = val
        r = float(parts[-1]) / 100.0
        times.append(t)
        rates.append(r)
    return np.array(times), np.array(rates)


class YieldCurve:
    """Discount factor and instantaneous forward rate engine using log-linear interpolation."""
    def __init__(self, times, rates):
        self.times = times
        self.log_dfs = -rates * times
        self.log_P_interp = interp1d(times, self.log_dfs, kind='linear', fill_value="extrapolate")

    def P(self, t):
        if np.isscalar(t) and t == 0:
            return 1.0
        t = np.maximum(t, 1e-6)
        return np.exp(self.log_P_interp(t))

    def f(self, t):
        dt = 1e-4
        ln_P_plus = self.log_P_interp(t + dt)
        ln_P_minus = self.log_P_interp(t - dt)
        return -(ln_P_plus - ln_P_minus) / (2 * dt)


# =====================================================
# 2. VOLATILITY SURFACE & DUPIRE MARKET PRE-PROCESSOR
# =====================================================

class RealVolSurface:
    """
    Constructs a arbitrage-aware Bachelier Implied Volatility Surface from CME Swaption quotes (5Y Tenor).
    Strikes are structured around ATM Forward moneyness (-200 bps to +200 bps).
    """
    def __init__(self):
        self.raw_data = {
            1/12: {  # Expiry: 1 Month
                'strikes': [0.008397861, 0.018397861, 0.023397861, 0.025897861, 0.028397861, 0.030897861, 0.033397861, 0.038397861, 0.048397861],
                'vols':    [0.0125917, 0.00863844, 0.00659171, 0.00574244, 0.00527809, 0.00549579, 0.00624259, 0.00831604, 0.01263272]
            },
            3/12: {  # Expiry: 3 Months
                'strikes': [0.00853084, 0.01853084, 0.02353084, 0.02603084, 0.02853084, 0.03103084, 0.03353084, 0.03853084, 0.04853084],
                'vols':    [0.01053615, 0.00779374, 0.00641361, 0.00585677, 0.00556826, 0.00569857, 0.00617026, 0.00753216, 0.0105163]
            },
            0.5: {   # Expiry: 6 Months
                'strikes': [0.008825431, 0.018825431, 0.023825431, 0.028825431, 0.033825431, 0.038825431, 0.048825431],
                'vols':    [0.00898141, 0.0071735, 0.00635934, 0.00597381, 0.0063566, 0.00724045, 0.00938581]
            },
            1.0: {   # Expiry: 1 Year
                'strikes': [0.009168537, 0.019168537, 0.024168537, 0.029168537, 0.034168537, 0.039168537, 0.049168537],
                'vols':    [0.00783434, 0.00687713, 0.00653661, 0.00645163, 0.00669478, 0.00719846, 0.00855354]
            },
            2.0: {   # Expiry: 2 Years
                'strikes': [0.009297802, 0.019297802, 0.024297802, 0.029297802, 0.034297802, 0.039297802, 0.049297802],
                'vols':    [0.00760015, 0.00709441, 0.00695462, 0.00696908, 0.00715874, 0.0075018, 0.00846702]
            }
        }
        
        self.expiries = sorted(self.raw_data.keys())
        self.smile_interpolators = {}
        
        for T in self.expiries:
            strikes = np.array(self.raw_data[T]['strikes'])
            raw_vols = np.array(self.raw_data[T]['vols'])
            
            sort_idx = np.argsort(strikes)
            strikes = strikes[sort_idx]
            raw_vols = raw_vols[sort_idx]
            
            vols_dec = raw_vols / 10000.0 if raw_vols.mean() > 1.0 else raw_vols
                
            # Natural Cubic Spline ensures continuous second derivatives (d2C/dK2) required by Dupire's formula
            self.smile_interpolators[T] = CubicSpline(strikes, vols_dec, bc_type='natural', extrapolate=True)

    def get_vol(self, T, K):
        T_clamped = np.clip(T, self.expiries[0], self.expiries[-1])
        
        # Flat extrapolation outside observed strike boundaries to prevent cubic spline divergence in the tails
        def safe_eval(t_val, k_val):
            spline = self.smile_interpolators[t_val]
            k_clamped = np.clip(k_val, spline.x[0], spline.x[-1])
            return float(spline(k_clamped))

        idx = np.searchsorted(self.expiries, T_clamped)
        if idx == 0:
            return safe_eval(self.expiries[0], K)
        if idx >= len(self.expiries):
            return safe_eval(self.expiries[-1], K)
            
        T_prev, T_next = self.expiries[idx-1], self.expiries[idx]
        w = (T_clamped - T_prev) / (T_next - T_prev)
        
        vol = (1 - w) * safe_eval(T_prev, K) + w * safe_eval(T_next, K)
        return max(float(vol), 0.0001)


class MarketPreprocessor:
    """Computes Bachelier option prices and numerical Dupire density terms (d2C/dK2)."""
    def __init__(self, real_vol_surface, yield_curve_obj):
        self.vol_surf_obj = real_vol_surface
        self.curve = yield_curve_obj

    def vol_surf(self, T, K):
        return self.vol_surf_obj.get_vol(T, K)

    def get_dupire_terms(self, t, strikes, theta):
        dk_eps = 1e-4
        P_start = self.curve.P(t)
        P_end = self.curve.P(t + theta)
        Annuity = max(theta * P_end, 1e-6)
        F_t = (P_start - P_end) / Annuity
        d2C_dK2 = []
        df_frozen = self.curve.P(t)

        def bach_price(_K):
            vol = self.vol_surf(t, _K)
            sqrt_t = np.sqrt(max(t, 1e-4))
            d = (F_t - _K) / (vol * sqrt_t)
            return df_frozen * ((F_t - _K) * norm.cdf(d) + vol * sqrt_t * norm.pdf(d))

        for K in strikes:
            C_center = bach_price(K)
            C_up = bach_price(K + dk_eps)
            C_down = bach_price(K - dk_eps)
            dk2_val = (C_up - 2 * C_center + C_down) / (dk_eps**2)
            d2C_dK2.append(dk2_val)
        return None, np.array(d2C_dK2)


# =========================================================
# 3. CHEYETTE LOCAL VOLATILITY PARTICLE CALIBRATION ENGINE
# =========================================================

class CheyetteParticleCalibrator:
    """
    Calibrates the Cheyette Local Volatility interest rate model via the McKean-Vlasov Particle Method.
    """
    def __init__(self, lambda_mean_rev, yield_curve_obj, market_prep_obj, n_particles=5000):
        self.lam = lambda_mean_rev
        self.curve = yield_curve_obj
        self.mkt_prep = market_prep_obj
        self.N = n_particles
        self.X = np.zeros(self.N)
        self.Y = np.zeros(self.N)
        self.sigma_loc_func = {}

    def _bond_price(self, t, T, x_t, y_t):
        if t == T:
            return np.ones_like(x_t)
        P0_T = self.curve.P(T)
        P0_t = self.curve.P(t)
        B = (1.0 - np.exp(-self.lam * (T - t))) / self.lam
        exponent = -B * x_t - 0.5 * (B**2) * y_t
        return (P0_T / P0_t) * np.exp(exponent)

    def _swap_rate(self, t, theta, x_t, y_t):
        T_end = t + theta
        P_end = self._bond_price(t, T_end, x_t, y_t)
        Annuity = theta * P_end
        S_val = (1.0 - P_end) / Annuity
        return S_val, P_end

    def _sensitivity_dx_swap(self, t, theta, x_t, y_t):
        T_end = t + theta
        B = (1.0 - np.exp(-self.lam * theta)) / self.lam
        P_end = self._bond_price(t, T_end, x_t, y_t)
        Annuity = theta * P_end
        return (B * P_end) / Annuity

    def calibrate_step(self, t, dt, theta):
        current_S_vals, P_vals = self._swap_rate(t, theta, self.X, self.Y)
        avg_S = np.mean(current_S_vals)
        std_S = max(np.std(current_S_vals), 0.0050)
        
        strike_grid = np.linspace(avg_S - 4 * std_S, avg_S + 4 * std_S, 25)
        _, mkt_den = self.mkt_prep.get_dupire_terms(t, strike_grid, theta)
        mkt_vols = np.array([self.mkt_prep.vol_surf(t, k) for k in strike_grid])
        base_variance = mkt_vols**2
        
        Xi_values = []
        f_tt = self.curve.f(t) + self.X
        f_t_theta = self.curve.f(t + theta)
        
        for K in strike_grid:
            indicator = (current_S_vals > K).astype(float)
            term_drift = (f_tt - f_t_theta * P_vals - current_S_vals * (1.0 - P_vals))
            Xi_values.append(np.mean(indicator * term_drift))
        Xi_values = np.array(Xi_values)

        max_den = max(np.max(np.abs(mkt_den)), 1e-9)
        den_clean = np.maximum(mkt_den, max_den * 0.10)
        
        raw_correction = (2 * Xi_values) / den_clean
        final_correction = np.clip(raw_correction * 0.5, -0.50 * base_variance, 1.00 * base_variance)
        total_var = np.maximum(base_variance + final_correction, (0.0020)**2)
        sigma_vals = np.sqrt(total_var)
        
        self.sigma_loc_func[t] = CubicSpline(strike_grid, sigma_vals, bc_type='clamped')
        return np.mean(sigma_vals)

    def evolve_particles(self, t, dt, theta):
        S_vals, _ = self._swap_rate(t, theta, self.X, self.Y)
        if t in self.sigma_loc_func:
            sigma_swap = np.clip(self.sigma_loc_func[t](S_vals), 0.0, 0.50)
        else:
            sigma_swap = np.full(self.N, 0.01)

        ds_dx = self._sensitivity_dx_swap(t, theta, self.X, self.Y)
        sigma_x = sigma_swap / (ds_dx + 1e-9)
        
        dW = np.random.normal(0, np.sqrt(dt), self.N)
        dx = (self.Y - self.lam * self.X) * dt + sigma_x * dW
        dy = (sigma_x**2 - 2 * self.lam * self.Y) * dt
        
        self.X += dx
        self.Y += dy
        # Non-negativity constraint on auxiliary variance state variable Y_t for numerical stability
        self.Y = np.maximum(self.Y, 0.0)


# =========================================================
# 4. MAIN EXECUTION: CALIBRATION, MONTE CARLO & SMILE FIT
# =========================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    times, rates = parse_yield_curve(estr_data_str)
    yc = YieldCurve(times, rates)
    vol_surface = RealVolSurface()
    mkt = MarketPreprocessor(vol_surface, yc)
    
    calibrator = CheyetteParticleCalibrator(
        lambda_mean_rev=0.10, 
        yield_curve_obj=yc, 
        market_prep_obj=mkt, 
        n_particles=5000
    )
    
    T_max = 3.0
    dt = 1/12      # Monthly time-stepping for enhanced Euler-Maruyama accuracy
    theta = 5.0    # 5Y Swap Tenor
    t_plot = 2.0   # 2Y Swaption Expiry target
    
    print("\nStarting Cheyette Particle Calibration on CME & €STR Market Data...")
    saved_S_vals = None
    
    for t in np.arange(0, T_max, dt):
        avg_vol = calibrator.calibrate_step(t, dt, theta)
        print(f"Time {t:.2f}Y calibrated | Avg Local Vol: {avg_vol*10000:.1f} bps")
        
        if np.isclose(t, t_plot):
            S_vals_at_t, _ = calibrator._swap_rate(t, theta, calibrator.X, calibrator.Y)
            saved_S_vals = S_vals_at_t.copy()
            
        calibrator.evolve_particles(t, dt, theta)

    print("Calibration completed successfully.")

    print(f"\nComputing Model vs. Market Implied Volatilities at Expiry {t_plot}Y...")
    P_start = yc.P(t_plot)
    P_end = yc.P(t_plot + theta)
    F_0 = (P_start - P_end) / (theta * P_end)
    
    strikes = np.linspace(F_0 - 0.01, F_0 + 0.01, 20)
    model_implied_vols, market_implied_vols = [], []
    
    def bach_formula(vol, K):
        d = (F_0 - K) / (vol * np.sqrt(t_plot))
        return (F_0 - K) * norm.cdf(d) + vol * np.sqrt(t_plot) * norm.pdf(d)

    for K in strikes:
        mc_price = np.mean(np.maximum(saved_S_vals - K, 0.0))
        
        try:
            impl_v = brentq(lambda v: bach_formula(v, K) - mc_price, 0.0001, 0.1500)
            model_implied_vols.append(impl_v * 10000)
        except ValueError:
            model_implied_vols.append(np.nan)
        
        market_implied_vols.append(mkt.vol_surf(t_plot, K) * 10000)

    print("\n--- NUMERICAL RESULTS (Bachelier Implied Volatility in bps) ---")
    for k, v_mod, v_mkt in zip(strikes, model_implied_vols, market_implied_vols):
        mod_str = f"{v_mod:.1f} bps" if not np.isnan(v_mod) else "NaN"
        print(f"Strike: {k*100:.2f}% | Model (MC): {mod_str} | Market (CME): {v_mkt:.1f} bps")

    plt.figure(figsize=(10, 6))
    plt.plot(strikes * 100, model_implied_vols, marker='o', label='Model Implied Vol (Cheyette Particle MC)', linewidth=2)
    plt.plot(strikes * 100, market_implied_vols, '--', label='Market Implied Vol (CME Target)', linewidth=2, alpha=0.8)
    plt.axvline(F_0 * 100, color='r', linestyle=':', label=f'ATM Forward Rate ({F_0*100:.2f}%)')
    plt.title(f"Bachelier Implied Volatility Smile: Cheyette Model vs. Market (Expiry {t_plot:.0f}Y, Tenor {theta:.0f}Y)")
    plt.xlabel("Strike (%)")
    plt.ylabel("Normal Implied Volatility (bps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    from pathlib import Path
    output_path = Path(__file__).resolve().parent / "calibration_smile_2Y5Y.png"
    plt.savefig(output_path, dpi=300)
    plt.show()