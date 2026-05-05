#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


SIGNALS = [
    "inssell",
    "booktomarket",
    "momentum",
    "grossprofit",
    "earnsurprise",
    "sue",
    "fscore",
    "xfin",
    "volatility",
    "accruals",
]


def run_monthly_univariate(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    rows = []
    for date, g in df.groupby("date", sort=True):
        x = sm.add_constant(g[[signal]], has_constant="add")
        y = g["ret"]
        model = sm.OLS(y, x).fit()
        rows.append(
            {
                "date": date,
                "signal": signal,
                "coef": model.params[signal],
                "intercept": model.params["const"],
                "r2": model.rsquared,
                "nobs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def run_monthly_multivariate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, g in df.groupby("date", sort=True):
        x = sm.add_constant(g[SIGNALS], has_constant="add")
        y = g["ret"]
        model = sm.OLS(y, x).fit()
        for signal in SIGNALS:
            rows.append(
                {
                    "date": date,
                    "signal": signal,
                    "coef": model.params[signal],
                    "intercept": model.params["const"],
                    "r2": model.rsquared,
                    "nobs": int(model.nobs),
                }
            )
    return pd.DataFrame(rows)


def summarize_signal(monthly_df: pd.DataFrame) -> pd.DataFrame:
    grouped = monthly_df.groupby("signal", sort=False)
    summary = grouped.agg(
        avg_coef=("coef", "mean"),
        std_coef=("coef", "std"),
        avg_r2=("r2", "mean"),
        months_used=("coef", "count"),
        total_obs=("nobs", "sum"),
    ).reset_index()
    summary["se_fmb"] = summary["std_coef"] / np.sqrt(summary["months_used"])
    summary["t_stat"] = summary["avg_coef"] / summary["se_fmb"]
    summary["is_significant_10pct"] = summary["t_stat"].abs() > 1.645
    return summary[
        [
            "signal",
            "avg_coef",
            "se_fmb",
            "t_stat",
            "avg_r2",
            "months_used",
            "total_obs",
            "is_significant_10pct",
        ]
    ]


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_print = summary_df.copy()
    summary_print["is_significant_10pct"] = summary_print["is_significant_10pct"].map(
        {True: "Yes", False: "No"}
    )
    return summary_print


def fraction_positive(monthly_df: pd.DataFrame) -> pd.DataFrame:
    frac = (
        monthly_df.groupby("signal", sort=False)["coef"]
        .apply(lambda s: (s > 0).mean())
        .reset_index(name="frac_months_positive")
    )
    frac["months_positive"] = (
        monthly_df.groupby("signal", sort=False)["coef"].apply(lambda s: (s > 0).sum()).values
    )
    frac["months_used"] = monthly_df.groupby("signal", sort=False)["coef"].count().values
    return frac[["signal", "months_positive", "months_used", "frac_months_positive"]]


def make_5x2_plot(monthly_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
    axes = axes.flatten()

    for idx, signal in enumerate(SIGNALS):
        ax = axes[idx]
        s = monthly_df.loc[monthly_df["signal"] == signal].sort_values("date")
        ax.plot(s["date"], s["coef"], color="#1f4e79", linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_title(signal, fontsize=11, fontweight="bold")
        ax.set_ylabel("Monthly Coefficient")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[-2:]:
        ax.set_xlabel("Date")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_implied_sharpe(summary_multi_df: pd.DataFrame) -> pd.DataFrame:
    sharpe_df = summary_multi_df[["signal", "t_stat", "months_used"]].copy()
    sharpe_df["implied_sharpe"] = sharpe_df["t_stat"] / np.sqrt(sharpe_df["months_used"])
    sharpe_df["abs_implied_sharpe"] = sharpe_df["implied_sharpe"].abs()
    sharpe_df = sharpe_df.sort_values("implied_sharpe", ascending=False).reset_index(drop=True)
    return sharpe_df


def make_sharpe_bar_plot(sharpe_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_df = sharpe_df.sort_values("implied_sharpe", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ["#1f4e79" if v >= 0 else "#9c2f2f" for v in plot_df["implied_sharpe"]]
    ax.bar(plot_df["signal"], plot_df["implied_sharpe"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Step C: Implied Sharpe Ratios from Step B t-Statistics", fontsize=13, fontweight="bold")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Implied Sharpe Ratio")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    def latex_escape(value: str) -> str:
        return (
            value.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
        )

    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return latex_escape(str(v))

    col_spec = "l" + "r" * (len(df.columns) - 1)
    pretty_cols = [str(c).replace("_", " ") for c in df.columns]
    header = " & ".join(latex_escape(c) for c in pretty_cols) + " \\\\"
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append(" & ".join(fmt(v) for v in row.values) + " \\\\")

    table = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label.replace('_', '-'))}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(table)


def pm_ready_commentary(summary_df: pd.DataFrame, frac_df: pd.DataFrame) -> str:
    merged = summary_df.merge(frac_df[["signal", "frac_months_positive"]], on="signal", how="left")
    strongest = merged.reindex(merged["t_stat"].abs().sort_values(ascending=False).index).head(4)
    weak = merged.loc[merged["signal"] == "accruals"].iloc[0]
    inssell = merged.loc[merged["signal"] == "inssell"].iloc[0]
    momentum = merged.loc[merged["signal"] == "momentum"].iloc[0]

    top_text = "; ".join(
        [
            f"{r.signal} (avg coef={r.avg_coef:.4f}, t-stat={r.t_stat:.2f}, fraction positive={r.frac_months_positive:.2%})"
            for r in strongest.itertuples()
        ]
    )

    lines = []
    lines.append(
        "In the univariate tests, the strongest relationships with monthly returns are momentum, earnings surprise, "
        "SUE, and volatility based on absolute t-statistics and stable coefficient signs through time. "
        f"These signals combine both statistical strength and directional consistency: {top_text}."
    )
    lines.append(
        f"Insider selling also stands out economically because it has the largest absolute average slope "
        f"({inssell.avg_coef:.4f}) and is negative in most months "
        f"(only {inssell.frac_months_positive:.2%} of months are positive). "
        "A practical interpretation is that higher net insider selling is associated with lower subsequent returns, "
        "which supports using insider-selling intensity as a short-side ranking input rather than a standalone timing signal."
    )
    lines.append(
        f"At the weak end, accruals has a low t-statistic ({weak.t_stat:.2f}) and limited reliability, "
        "so we would treat it as low-conviction in this sample. "
        "Across the coefficient paths, signs are generally more stable than magnitudes, which is typical in noisy monthly "
        "cross-sections where signal strength varies with market regime."
    )
    lines.append(
        "Economically, a coefficient gives the expected change in monthly return for a one-standard-deviation change in a signal "
        "(because signals are standardized within month). For example, a momentum slope near "
        f"{momentum.avg_coef:.4f} means a stock that is 1 SD higher in momentum is expected to have roughly "
        f"{momentum.avg_coef:.4f} higher monthly return, holding only the univariate setup fixed."
    )
    lines.append(
        "A plausible mechanism for momentum is slow information diffusion and trend-chasing behavior, where investors continue "
        "to ride recent winners before fully updating valuations. For insider selling, one explanation is that by the time trades "
        "are disclosed, part of the informational edge is already reflected in prices; nevertheless, persistent net selling still "
        "contains incremental negative information in this dataset."
    )
    return "\n\n".join(lines)


def step_b_commentary(summary_multi_df: pd.DataFrame, summary_uni_df: pd.DataFrame) -> str:
    ordered = summary_multi_df.reindex(summary_multi_df["t_stat"].abs().sort_values(ascending=False).index).head(4)
    comp = summary_multi_df[["signal", "avg_coef", "t_stat", "avg_r2", "is_significant_10pct"]].merge(
        summary_uni_df[["signal", "avg_coef", "t_stat"]].rename(
            columns={"avg_coef": "avg_coef_uni", "t_stat": "t_stat_uni"}
        ),
        on="signal",
        how="left",
    )
    comp["delta_t"] = comp["t_stat"] - comp["t_stat_uni"]
    comp["delta_coef"] = comp["avg_coef"] - comp["avg_coef_uni"]
    biggest_t_drop = comp.loc[comp["delta_t"].idxmin()]
    avg_r2 = comp["avg_r2"].iloc[0]
    accruals_row = comp.loc[comp["signal"] == "accruals"].iloc[0]

    top_text = "; ".join(
        [f"{r.signal} (avg coef={r.avg_coef:.4f}, t-stat={r.t_stat:.2f})" for r in ordered.itertuples()]
    )

    lines = []
    lines.append(
        "In the multivariate tests, most signals remain significant at the 10% threshold after controlling for the full signal set. "
        f"The strongest multivariate predictors by absolute t-statistic are {top_text}. "
        f"Accruals remains weak and statistically insignificant (t-stat={accruals_row.t_stat:.2f})."
    )
    lines.append(
        f"The largest drop in t-statistic versus univariate appears in {biggest_t_drop.signal} "
        f"(change {biggest_t_drop.delta_t:.2f}), which suggests overlap with other signals rather than pure standalone information loss. "
        "More broadly, when coefficients shrink after controls, the interpretation is that part of the univariate relation was shared "
        "with correlated characteristics."
    )
    lines.append(
        f"The average multivariate R-squared is about {avg_r2:.3f}, which is meaningful for monthly cross-sectional stock-return "
        "variation: the model explains a nontrivial share of cross-firm return differences, but still leaves substantial residual noise, "
        "so risk controls and diversification remain necessary."
    )
    lines.append(
        "In economic terms, each multivariate coefficient is a partial effect: a one-standard-deviation increase in a signal corresponds "
        "to the coefficient change in monthly return, holding all other signals fixed. This is the right lens for portfolio construction "
        "because it reflects incremental contribution rather than raw standalone association."
    )
    return "\n\n".join(lines)


def step_c_commentary(sharpe_df: pd.DataFrame) -> str:
    top_abs = sharpe_df.sort_values("abs_implied_sharpe", ascending=False).head(4)
    top_text = "; ".join(
        [
            f"{r.signal} (implied Sharpe={r.implied_sharpe:.4f})"
            for r in top_abs.itertuples()
        ]
    )
    lines = []
    lines.append(
        "Using the assignment convention, implied Sharpe ratios are computed as t-stat divided by the square root of the number of months. "
        "The strongest four signals in absolute implied Sharpe terms are "
        f"{top_text}. This ranking highlights both high positive and high negative opportunities."
    )
    lines.append(
        "For strategy design, strongly positive implied Sharpe signals are natural long candidates, while strongly negative implied "
        "Sharpe signals can be used as short candidates or negative tilts. In practice, we would combine sign, magnitude, and reliability "
        "into position sizing rather than using equal weights."
    )
    lines.append(
        "A robust implementation would include turnover and trading-cost controls, sector/market-neutral constraints, and conservative "
        "risk budgeting to avoid overexposure to a single theme despite strong historical signal performance."
    )
    return "\n\n".join(lines)


def build_tex_document(
    out_path: Path,
    step_a_summary_table_tex: str,
    step_a_frac_table_tex: str,
    step_a_text: str,
    step_b_summary_table_tex: str,
    step_b_text: str,
    step_c_sharpe_table_tex: str,
    step_c_text: str,
    team_members: str,
    step_a_figure_relpath: str,
    step_b_figure_relpath: str,
    step_c_figure_relpath: str,
) -> None:
    step_a_text_latex = step_a_text.replace("%", "\\%").replace("_", "\\_")
    step_b_text_latex = step_b_text.replace("%", "\\%").replace("_", "\\_")
    step_c_text_latex = step_c_text.replace("%", "\\%").replace("_", "\\_")
    step_a_body = step_a_text_latex.replace("\n\n", "\n\\par\n")
    step_b_body = step_b_text_latex.replace("\n\n", "\n\\par\n")
    step_c_body = step_c_text_latex.replace("\n\n", "\n\\par\n")
    tex = f"""\\documentclass{{article}}
\\usepackage[a4paper,total={{6in,8in}}]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{setspace}}
\\usepackage{{hyperref}}
\\setstretch{{1.1}}
\\setcounter{{secnumdepth}}{{0}}
\\renewcommand{{\\contentsname}}{{Table of Contens}}
\\setlength{{\\textfloatsep}}{{8pt plus 2pt minus 2pt}}
\\setlength{{\\floatsep}}{{8pt plus 2pt minus 2pt}}
\\setlength{{\\intextsep}}{{8pt plus 2pt minus 2pt}}
\\title{{Final Project}}
\\author{{{team_members}}}
\\date{{\\today}}
\\begin{{document}}
\\begin{{titlepage}}
\\centering
\\vspace*{{1.5cm}}
\\IfFileExists{{Primary_logo_thumbnail.png.webp}}{{\\includegraphics[width=0.28\\textwidth]{{\\detokenize{{Primary_logo_thumbnail.png.webp}}}}}}{{\\fbox{{MIT Sloan Logo}}}}
\\vspace{{1.2cm}}

{{\\LARGE Final Project\\par}}
\\vspace{{0.6cm}}
{{\\large 15.465 Alphanomics\\par}}
\\vspace{{1.0cm}}
{{\\large {team_members}\\par}}
\\vfill
{{\\large \\today\\par}}
\\end{{titlepage}}

\\clearpage
\\tableofcontents
\\thispagestyle{{plain}}
\\clearpage
\\listoffigures
\\thispagestyle{{plain}}
\\listoftables
\\thispagestyle{{plain}}
\\clearpage

\\section*{{Part 1: Data Analysis}}
\\addcontentsline{{toc}}{{section}}{{Part 1: Data Analysis}}
\\subsection*{{1.1 Univariate}}
\\addcontentsline{{toc}}{{subsection}}{{1.1 Univariate}}
For each month, we run a cross-sectional univariate regression of monthly return (\\texttt{{ret}}) on one standardized signal:
\\[
\\texttt{{ret}}_{{i,t}} = \\alpha_t + \\beta_t \\cdot \\texttt{{signal}}_{{i,t}} + \\varepsilon_{{i,t}}.
\\]
We repeat this separately for each of the 10 signals and for all available months. Following the assignment instructions, we compute Fama-MacBeth standard errors as:
\\[
SE(\\bar{{\\beta}}) = \\frac{{\\mathrm{{std}}(\\beta_t)}}{{\\sqrt{{T}}}}, \\quad
t = \\frac{{\\bar{{\\beta}}}}{{SE(\\bar{{\\beta}})}}.
\\]
We run these monthly cross-sectional regressions across all available months and save the full time series of coefficients for each signal for downstream analysis.

\\begin{{figure}}[!htbp]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.58\\textheight,keepaspectratio]{{\\detokenize{{{step_a_figure_relpath}}}}}
    \\caption{{Monthly univariate slope coefficients by signal (5x2 grid).}}
\\end{{figure}}

The figure above reports the coefficient paths over time. The table below reports the time-series average coefficients, Fama-MacBeth t-statistics, average R-squared values, and the number of observations used.
{step_a_summary_table_tex}

The next table reports the share of months in which each signal's monthly coefficient is positive.
{step_a_frac_table_tex}

\\subsection*{{1.2 Multivariate}}
\\addcontentsline{{toc}}{{subsection}}{{1.2 Multivariate}}
We next run monthly cross-sectional regressions including all ten signals simultaneously, so each coefficient reflects a partial relationship with return while controlling for the other signals.

\\begin{{figure}}[!htbp]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.58\\textheight,keepaspectratio]{{\\detokenize{{{step_b_figure_relpath}}}}}
    \\caption{{Monthly multivariate slope coefficients by signal (5x2 grid).}}
\\end{{figure}}

The figure reports the multivariate coefficient paths over time. The table below reports the time-series average coefficients, Fama-MacBeth t-statistics, average R-squared values, and the number of observations used.
{step_b_summary_table_tex}

\\subsection*{{1.3 Sharpe Ratio}}
\\addcontentsline{{toc}}{{subsection}}{{1.3 Sharpe Ratio}}
Using the Step B t-statistics, we compute implied Sharpe ratios and report them in descending order.

\\begin{{figure}}[!htbp]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.45\\textheight,keepaspectratio]{{\\detokenize{{{step_c_figure_relpath}}}}}
    \\caption{{Implied Sharpe ratios by signal, sorted in descending order.}}
\\end{{figure}}

{step_c_sharpe_table_tex}

\\section*{{Part 2: Written Analysis}}
\\addcontentsline{{toc}}{{section}}{{Part 2: Written Analysis}}

\\subsection*{{1.1 Univariate}}
\\addcontentsline{{toc}}{{subsection}}{{1.1 Univariate}}
{step_a_body}

\\subsection*{{1.2 Multivariate}}
\\addcontentsline{{toc}}{{subsection}}{{1.2 Multivariate}}
{step_b_body}

\\subsection*{{1.3 Sharpe Ratio}}
\\addcontentsline{{toc}}{{subsection}}{{1.3 Sharpe Ratio}}
{step_c_body}

\\end{{document}}
"""
    out_path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1 data analysis report builder.")
    parser.add_argument("--input", default="fmreg_data_std_28.csv", help="Path to input CSV")
    parser.add_argument("--outdir", default="To Upload", help="Output directory")
    parser.add_argument("--tex-output", default="main.tex", help="Single LaTeX output file")
    parser.add_argument(
        "--team-members",
        default="Team Members: [Add Name 1], [Add Name 2], [Add Name 3]",
        help="Author line for report title page",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    monthly_results = []
    for signal in SIGNALS:
        monthly_results.append(run_monthly_univariate(df, signal))
    monthly_uni_df = pd.concat(monthly_results, ignore_index=True)
    monthly_multi_df = run_monthly_multivariate(df)

    summary_uni_df = summarize_signal(monthly_uni_df)
    frac_df = fraction_positive(monthly_uni_df)
    summary_multi_df = summarize_signal(monthly_multi_df)

    make_5x2_plot(
        monthly_uni_df,
        out_root / "univariate_coefficients_5x2.png",
        "Step A: Univariate Monthly Coefficients (Fama-MacBeth Step 1)",
    )
    make_5x2_plot(
        monthly_uni_df,
        out_root / "univariate_coefficients_5x2.pdf",
        "Step A: Univariate Monthly Coefficients (Fama-MacBeth Step 1)",
    )
    make_5x2_plot(
        monthly_multi_df,
        out_root / "stepB_multivariate_coefficients_5x2.png",
        "Step B: Multivariate Monthly Coefficients (All Signals Included)",
    )
    make_5x2_plot(
        monthly_multi_df,
        out_root / "stepB_multivariate_coefficients_5x2.pdf",
        "Step B: Multivariate Monthly Coefficients (All Signals Included)",
    )

    summary_uni_print = format_summary_table(summary_uni_df)
    summary_uni_table_tex = render_latex_table(
        summary_uni_print,
        caption="Univariate Fama-MacBeth summary statistics by signal.",
        label="tab:stepA_univariate_summary",
    )
    frac_table_tex = render_latex_table(
        frac_df,
        caption="Fraction of months with positive univariate coefficient.",
        label="tab:stepA_fraction_positive",
    )
    summary_multi_print = format_summary_table(summary_multi_df)
    summary_multi_table_tex = render_latex_table(
        summary_multi_print,
        caption="Multivariate Fama-MacBeth summary statistics by signal.",
        label="tab:stepB_multivariate_summary",
    )

    sharpe_df = compute_implied_sharpe(summary_multi_df)
    make_sharpe_bar_plot(sharpe_df, out_root / "stepC_implied_sharpe_bar.png")
    make_sharpe_bar_plot(sharpe_df, out_root / "stepC_implied_sharpe_bar.pdf")
    sharpe_table_tex = render_latex_table(
        sharpe_df[["signal", "t_stat", "months_used", "implied_sharpe"]].copy(),
        caption="Implied Sharpe ratios from Step B t-statistics.",
        label="tab:stepC_implied_sharpe",
    )

    step_a_text = pm_ready_commentary(summary_uni_df, frac_df)
    step_b_text = step_b_commentary(summary_multi_df, summary_uni_df)
    step_c_text = step_c_commentary(sharpe_df)
    build_tex_document(
        out_path=Path(args.tex_output),
        step_a_summary_table_tex=summary_uni_table_tex,
        step_a_frac_table_tex=frac_table_tex,
        step_a_text=step_a_text,
        step_b_summary_table_tex=summary_multi_table_tex,
        step_b_text=step_b_text,
        step_c_sharpe_table_tex=sharpe_table_tex,
        step_c_text=step_c_text,
        team_members=args.team_members,
        step_a_figure_relpath=str(out_root / "univariate_coefficients_5x2.pdf"),
        step_b_figure_relpath=str(out_root / "stepB_multivariate_coefficients_5x2.pdf"),
        step_c_figure_relpath=str(out_root / "stepC_implied_sharpe_bar.pdf"),
    )

    package_tex = out_root / "main.tex"
    source_tex = Path(args.tex_output)
    if source_tex.resolve() != package_tex.resolve():
        upload_text = source_tex.read_text(encoding="utf-8").replace("To Upload/", "")
        package_tex.write_text(upload_text, encoding="utf-8")

    print("Deliverables generated in:", out_root.resolve())
    print("Step A figure:", (out_root / "univariate_coefficients_5x2.pdf").resolve())
    print("Step B figure:", (out_root / "stepB_multivariate_coefficients_5x2.pdf").resolve())
    print("Step C figure:", (out_root / "stepC_implied_sharpe_bar.pdf").resolve())
    print("LaTeX report (root):", source_tex.resolve())
    print("LaTeX report (upload copy):", package_tex.resolve())


if __name__ == "__main__":
    main()
