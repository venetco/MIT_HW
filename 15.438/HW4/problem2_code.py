import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, lognorm, kstest, skew, kurtosis
import warnings
warnings.filterwarnings("ignore")


# CIR closed-form zero-coupon bond price
def cir_zcb(r, beta, lr, sigma, tau):
    """CIR closed-form zero-coupon bond price P(0, tau) given r0 = r."""
    g = np.sqrt(beta**2 + 2 * sigma**2)
    eg = np.exp(np.clip(g * tau, -50, 50))
    den = (g + beta) * (eg - 1) + 2 * g
    if den <= 0:
        return 1e-10
    B = 2 * (eg - 1) / den
    expo = 2 * beta * lr / (sigma**2)
    base = 2 * g * np.exp(np.clip((beta + g) * tau / 2, -50, 50)) / den
    if base <= 0:
        return 1e-10
    A = base ** expo
    return A * np.exp(-B * r)


def y_cont(P, tau):
    """Continuous-compound spot yield from a zero price."""
    return -np.log(P) / tau


def cont_to_beb(yc):
    """Continuous yield -> bond-equivalent (semi-annual compounding) yield."""
    return 2 * (np.exp(yc / 2) - 1)


def cir_yield_beb(r, beta, lr, sigma, tau):
    return cont_to_beb(y_cont(cir_zcb(r, beta, lr, sigma, tau), tau))


def cir2_yield_beb(r10, b1, lr1, s1, r20, b2, lr2, s2, tau):
    """Two-factor CIR: r = r1 + r2, factors independent, so P = P1 * P2."""
    P = cir_zcb(r10, b1, lr1, s1, tau) * cir_zcb(r20, b2, lr2, s2, tau)
    return cont_to_beb(y_cont(P, tau))


# Part (a): calibrate 1-factor CIR to the given target yield curve
def part_a():
    print("=" * 70)
    print("PART (a): Calibrate 1-factor CIR")
    print("=" * 70)

    # Targets quoted on bond-equivalent basis
    target_taus = np.array([0.5, 1.0, 5.0, 10.0, 20.0, 30.0])
    target_beb = np.array([0.015, 0.019, 0.025, 0.029, 0.030, 0.029])
    focus_idx = [1, 2, 3, 5]   # 1, 5, 10, 30 yr per problem statement

    # Sigma is poorly identified by curve fit alone (it enters only via a small
    # convexity term), so we fix sigma at a typical level and fit (r0, beta, lr).
    sigma_fix = 0.08

    def sse_focus(p):
        r0, beta, lr = p
        if r0 <= 0 or beta <= 0 or lr <= 0:
            return 1e6
        return sum((cir_yield_beb(r0, beta, lr, sigma_fix, target_taus[i])
                    - target_beb[i])**2 for i in focus_idx)

    # Multi-start Nelder-Mead
    best = None
    for r0g in np.linspace(0.012, 0.020, 5):
        for bg in [0.1, 0.3, 0.5, 0.7, 1.0]:
            for lrg in [0.025, 0.030, 0.035, 0.040]:
                res = minimize(sse_focus, [r0g, bg, lrg], method="Nelder-Mead",
                               options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 10000})
                if best is None or res.fun < best.fun:
                    best = res

    r0, beta, lr = best.x
    print(f"\nCalibrated parameters (sigma fixed at {sigma_fix}):")
    print(f"  r0    = {r0:.5f}")
    print(f"  beta  = {beta:.5f}")
    print(f"  lr    = {lr:.5f}")
    print(f"  sigma = {sigma_fix:.5f}")
    print(f"  Feller: 2*beta*lr = {2*beta*lr:.4f} > sigma^2 = {sigma_fix**2:.4f} -> {2*beta*lr > sigma_fix**2}")

    print(f"\n{'Tau':>5} {'Target':>10} {'Model':>10} {'Err (bp)':>10}")
    for i, t in enumerate(target_taus):
        y = cir_yield_beb(r0, beta, lr, sigma_fix, t)
        err = (y - target_beb[i]) * 1e4
        marker = " (focus)" if i in focus_idx else ""
        print(f"{t:>5.1f} {target_beb[i]*100:>9.3f}% {y*100:>9.3f}% {err:>+9.2f}{marker}")
    print(f"\nSSE (focus) = {best.fun:.3e}, MSE (focus) = {best.fun/4:.3e}")

    # Plot
    plot_taus = np.linspace(0.25, 30, 200)
    fitted = np.array([cir_yield_beb(r0, beta, lr, sigma_fix, t) for t in plot_taus])
    plt.figure(figsize=(8, 5))
    plt.plot(plot_taus, fitted * 100, "b-", lw=2, label="Fitted CIR curve")
    plt.scatter(target_taus, target_beb * 100, c="red", s=80, zorder=5, label="Target points")
    plt.scatter(target_taus[focus_idx], target_beb[focus_idx] * 100,
                edgecolors="black", facecolors="none", s=160, lw=2,
                label="Focus points (1, 5, 10, 30y)")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Spot yield (%, b.e.b.)")
    plt.title(f"CIR-fitted yield curve\n"
              f"$r_0$={r0:.4f}, $\\beta$={beta:.3f}, $\\bar r$={lr:.4f}, $\\sigma$={sigma_fix:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("2a_yield_curve.png", dpi=150)
    plt.close()
    print("Saved 2a_yield_curve.png")

    return r0, beta, lr, sigma_fix


# Parts (b) & (c): histograms of the short rate and 10y yield, 5y out
def part_bc(r0, beta, lr, sigma):
    print("\n" + "=" * 70)
    print("PARTS (b) & (c): Histograms 5 years out")
    print("=" * 70)

    # Monte Carlo: monthly time step, full-truncation Euler scheme keeps r >= 0
    dt = 1 / 12
    n_months = 60
    n_paths = 5000   # well above the 100 minimum required
    np.random.seed(42)

    r = np.full(n_paths, r0)
    for _ in range(n_months):
        rp = np.maximum(r, 0)
        z = np.random.standard_normal(n_paths)
        r = r + beta * (lr - rp) * dt + sigma * np.sqrt(rp * dt) * z
        r = np.maximum(r, 0)

    # 10y yield from each terminal short rate via closed-form CIR
    P10 = np.array([cir_zcb(ri, beta, lr, sigma, 10.0) for ri in r])
    y10_beb = cont_to_beb(-np.log(P10) / 10.0)

    print(f"\n5-year-ahead short rate (n={n_paths}):")
    print(f"  mean={r.mean()*100:.3f}%  std={r.std()*100:.3f}%  "
          f"skew={skew(r):.3f}  excess kurt={kurtosis(r):.3f}")

    print(f"\n10-year spot yield 5y out:")
    print(f"  mean={y10_beb.mean()*100:.3f}%  std={y10_beb.std()*100:.3f}%  "
          f"skew={skew(y10_beb):.3f}  excess kurt={kurtosis(y10_beb):.3f}")

    # Goodness-of-fit: KS distance vs Normal and Lognormal (matched moments)
    def lognormal_mom(x):
        m, v = x.mean(), x.std()**2
        lmu = np.log(m**2 / np.sqrt(v + m**2))
        lsi = np.sqrt(np.log(1 + v / m**2))
        return lmu, lsi

    print("\nKolmogorov-Smirnov goodness-of-fit:")
    for label, x in [("short rate", r), ("10y yield", y10_beb)]:
        ks_n = kstest(x, "norm", args=(x.mean(), x.std()))
        lmu, lsi = lognormal_mom(x)
        ks_l = kstest(x, "lognorm", args=(lsi, 0, np.exp(lmu)))
        print(f"  {label:>12}: D_normal={ks_n.statistic:.4f}, "
              f"D_lognormal={ks_l.statistic:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, x, color, title, xlabel in [
        (axes[0], r,        "steelblue",   f"5-year-ahead short rate (n={n_paths})", "Short rate (%)"),
        (axes[1], y10_beb,  "darkorange",  f"10-year spot yield 5y out (n={n_paths})", "10-year spot yield (%, b.e.b.)"),
    ]:
        ax.hist(x * 100, bins=40, density=True, alpha=0.55,
                color=color, edgecolor="white", label="Simulated")
        mu, sd = x.mean(), x.std()
        xs = np.linspace(max(x.min(), 1e-6), x.max(), 400)
        ax.plot(xs * 100, norm.pdf(xs, mu, sd) / 100, "r-", lw=2,
                label=f"Normal $\\mu$={mu*100:.2f}%, $\\sigma$={sd*100:.2f}%")
        lmu, lsi = lognormal_mom(x)
        ax.plot(xs * 100, lognorm.pdf(xs, lsi, scale=np.exp(lmu)) / 100, "g--", lw=2,
                label="Lognormal (matched moments)")
        ax.set_xlabel(xlabel); ax.set_ylabel("Density (per %)")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("2bc_histograms.png", dpi=150)
    plt.close()
    print("\nSaved 2bc_histograms.png")
    return r, y10_beb


# Part (d): European call on a 1.5y T-Note, expiry in 2 years
def part_d(r0, beta, lr, sigma):
    print("\n" + "=" * 70)
    print("PART (d): European call on 1.5y T-Note")
    print("=" * 70)

    face = 100_000.0
    coupon = 0.025 / 2 * face   # semi-annual coupon = $1,250
    strike = 100_000.0
    T_opt = 2.0

    def bond_price_at_expiry(r2):
        """At delivery (t=2), bond pays coupons at +0.5, +1.0 and principal+coupon at +1.5."""
        return (coupon * cir_zcb(r2, beta, lr, sigma, 0.5)
                + coupon * cir_zcb(r2, beta, lr, sigma, 1.0)
                + (coupon + face) * cir_zcb(r2, beta, lr, sigma, 1.5))

    # Monte Carlo with monthly steps to t=2 (24 steps)
    dt = 1 / 12
    n_steps = int(round(T_opt / dt))
    n_paths = 50_000
    np.random.seed(7)

    r = np.full(n_paths, r0)
    disc_int = np.zeros(n_paths)   # accumulator for integral of r ds (trapezoidal)

    for _ in range(n_steps):
        rp = np.maximum(r, 0)
        z = np.random.standard_normal(n_paths)
        r_new = r + beta * (lr - rp) * dt + sigma * np.sqrt(rp * dt) * z
        r_new = np.maximum(r_new, 0)
        disc_int += 0.5 * (rp + np.maximum(r_new, 0)) * dt
        r = r_new

    discount = np.exp(-disc_int)
    B2 = bond_price_at_expiry(r)
    payoff = np.maximum(B2 - strike, 0)

    call_price = (discount * payoff).mean()
    call_se = (discount * payoff).std() / np.sqrt(n_paths)

    print(f"\nNumber of paths: {n_paths}")
    print(f"Mean bond price at expiry: ${B2.mean():.2f}")
    print(f"In-the-money paths: {(payoff>0).mean()*100:.2f}%")
    print(f"\nCall option price: ${call_price:.2f}")
    print(f"Standard error:    ${call_se:.2f}")
    print(f"95% CI:            [${call_price-1.96*call_se:.2f}, ${call_price+1.96*call_se:.2f}]")

    # Sanity checks: discount factor mean and forward bond price
    P02 = cir_zcb(r0, beta, lr, sigma, 2.0)
    fwd_cf = (coupon * cir_zcb(r0, beta, lr, sigma, 2.5)
              + coupon * cir_zcb(r0, beta, lr, sigma, 3.0)
              + (coupon + face) * cir_zcb(r0, beta, lr, sigma, 3.5)) / P02
    print(f"\nSanity checks:")
    print(f"  P(0,2) closed form = {P02:.6f}, MC discount mean = {discount.mean():.6f}")
    print(f"  Forward bond closed form = ${fwd_cf:.2f}, MC = ${(discount*B2).mean()/discount.mean():.2f}")
    return call_price, call_se


# Part (e): new target curve, 1-factor and 2-factor CIR
def part_e():
    print("\n" + "=" * 70)
    print("PART (e): Inverted-then-rising target curve (extra credit)")
    print("=" * 70)

    target_taus = np.array([0.5, 1.0, 5.0, 10.0, 20.0, 30.0])
    target_beb = np.array([0.054, 0.058, 0.047, 0.046, 0.048, 0.046])

    # ---- 1-factor benchmark ----
    sigma_fix = 0.08

    def sse1(p):
        r0, beta, lr = p
        if any(x <= 0 for x in p):
            return 1e6
        return sum((cir_yield_beb(r0, beta, lr, sigma_fix, t) - target_beb[i])**2
                   for i, t in enumerate(target_taus))

    best1 = None
    for r0g in [0.040, 0.050, 0.055, 0.058]:
        for bg in [0.1, 0.3, 0.5, 1.0, 2.0]:
            for lrg in [0.040, 0.045, 0.048]:
                r = minimize(sse1, [r0g, bg, lrg], method="Nelder-Mead",
                             options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 10000})
                if best1 is None or r.fun < best1.fun:
                    best1 = r
    r0_1, beta_1, lr_1 = best1.x
    sse1_val = best1.fun
    print(f"\n1-factor CIR (sigma={sigma_fix}):")
    print(f"  r0={r0_1:.5f}  beta={beta_1:.5f}  lr={lr_1:.5f}")
    print(f"  SSE={sse1_val:.4e}, MSE={sse1_val/6:.4e}")

    # ---- 2-factor fit, parameters bounded to economically reasonable ranges ----
    def sse2(p):
        e = 0
        for i, t in enumerate(target_taus):
            y = cir2_yield_beb(*p, t)
            if not np.isfinite(y) or abs(y) > 10:
                return 1e6
            e += (y - target_beb[i])**2
        return e

    bounds = [
        (0.01, 0.06), (0.05, 3.0), (0.01, 0.06), (0.02, 0.10),  # factor 1
        (0.01, 0.06), (0.05, 3.0), (0.01, 0.06), (0.02, 0.10),  # factor 2
    ]
    best_overall = None
    for seed in [42, 256]:
        res = differential_evolution(sse2, bounds, seed=seed, maxiter=400,
                                     tol=1e-12, popsize=25, polish=True)
        if best_overall is None or res.fun < best_overall.fun:
            best_overall = res
    result = best_overall
    r10, b1, lr1, s1, r20, b2, lr2, s2 = result.x
    if b1 < b2:   # canonicalize: factor 1 = faster mean reversion
        r10, b1, lr1, s1, r20, b2, lr2, s2 = r20, b2, lr2, s2, r10, b1, lr1, s1

    print(f"\n2-factor CIR:")
    print(f"  Fast: r0={r10:.5f} beta={b1:.4f} lr={lr1:.5f} sigma={s1:.4f}")
    print(f"  Slow: r0={r20:.5f} beta={b2:.4f} lr={lr2:.5f} sigma={s2:.4f}")
    print(f"  Combined r(0)={r10+r20:.5f}, combined long-run mean={lr1+lr2:.5f}")
    print(f"  SSE={result.fun:.4e}, MSE={result.fun/6:.4e}")
    print(f"  SSE reduction vs 1-factor: {(1-result.fun/sse1_val)*100:.1f}%")

    print(f"\n{'Tau':>5} {'Target':>8} {'1-fac':>8} {'2-fac':>8} {'1f bp':>8} {'2f bp':>8}")
    for i, t in enumerate(target_taus):
        y1 = cir_yield_beb(r0_1, beta_1, lr_1, sigma_fix, t)
        y2 = cir2_yield_beb(r10, b1, lr1, s1, r20, b2, lr2, s2, t)
        print(f"{t:>5.1f} {target_beb[i]*100:>7.3f}% {y1*100:>7.3f}% {y2*100:>7.3f}% "
              f"{(y1-target_beb[i])*1e4:>+7.2f} {(y2-target_beb[i])*1e4:>+7.2f}")

    # ---- Monte Carlo verification of 2-factor curve ----
    print(f"\nMonte Carlo verification of 2-factor curve (10,000 paths):")
    dt = 1 / 12

    def mc_yield(tau, n_paths=10_000, seed=2):
        np.random.seed(seed)
        n_steps = int(round(tau / dt))
        r1 = np.full(n_paths, r10); r2 = np.full(n_paths, r20)
        disc = np.zeros(n_paths)
        for _ in range(n_steps):
            rp1 = np.maximum(r1, 0); rp2 = np.maximum(r2, 0)
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            r1n = r1 + b1 * (lr1 - rp1) * dt + s1 * np.sqrt(rp1 * dt) * z1
            r2n = r2 + b2 * (lr2 - rp2) * dt + s2 * np.sqrt(rp2 * dt) * z2
            r1n = np.maximum(r1n, 0); r2n = np.maximum(r2n, 0)
            disc += 0.5 * ((rp1 + rp2) + (r1n + r2n)) * dt
            r1, r2 = r1n, r2n
        return cont_to_beb(-np.log(np.exp(-disc).mean()) / tau)

    print(f"{'Tau':>5} {'Target':>8} {'CF':>10} {'MC':>10}")
    for t in [0.5, 1.0, 5.0, 10.0]:
        y_cf = cir2_yield_beb(r10, b1, lr1, s1, r20, b2, lr2, s2, t)
        y_mc = mc_yield(t)
        idx = list(target_taus).index(t)
        print(f"{t:>5.1f} {target_beb[idx]*100:>7.3f}% {y_cf*100:>9.3f}% {y_mc*100:>9.3f}%")

    # Plot
    plot_taus = np.linspace(0.25, 30, 200)
    y1c = np.array([cir_yield_beb(r0_1, beta_1, lr_1, sigma_fix, t) for t in plot_taus])
    y2c = np.array([cir2_yield_beb(r10, b1, lr1, s1, r20, b2, lr2, s2, t) for t in plot_taus])
    plt.figure(figsize=(9, 5.5))
    plt.plot(plot_taus, y1c * 100, "b-",  lw=2, label=f"1-factor CIR  (MSE={sse1_val/6:.2e})")
    plt.plot(plot_taus, y2c * 100, "g--", lw=2, label=f"2-factor CIR  (MSE={result.fun/6:.2e})")
    plt.scatter(target_taus, target_beb * 100, c="red", s=80, zorder=5, label="Target points")
    plt.xlabel("Maturity (years)"); plt.ylabel("Spot yield (%, b.e.b.)")
    plt.title("Part (e): inverted-then-rising target curve - CIR fits")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("2e_curve_fit.png", dpi=150)
    plt.close()
    print("\nSaved 2e_curve_fit.png")


# RUN
if __name__ == "__main__":
    r0, beta, lr, sigma = part_a()
    part_bc(r0, beta, lr, sigma)
    part_d(r0, beta, lr, sigma)
    part_e()
