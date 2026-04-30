import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def save_table(df_table, filename, title='', col_widths=None):
    n_rows, n_cols = df_table.shape
    fig_h = max(1.2, 0.45 * (n_rows + 2))
    fig_w = col_widths if col_widths else max(6, n_cols * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                   rowLabels=df_table.index, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0f1b3c')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == -1:
            cell.set_facecolor('#e8eaf2')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('white' if row % 2 == 0 else '#f4f6fb')
        cell.set_edgecolor('#cccccc')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# LOAD DATA
df = pd.read_csv('AnalystData_2024.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows         : {len(df):,}")
print(f"Year range   : {df['Year'].min()} – {df['Year'].max()}")
print(f"Unique months: {df['YearMonth'].nunique()}")
print()
print(df[['BuyPCT', 'HoldPCT', 'SellPCT', 'MR1', 'retlag12']].describe().round(4))


# Q1
print("\n" + "=" * 60)
print("QUESTION 1 – Analysts recommandations")
print("=" * 60)

q1 = df[['BuyPCT', 'HoldPCT', 'SellPCT']].mean()
print(q1.round(4).to_string())
print(f"\nBuy/Sell ratio: {q1['BuyPCT'] / q1['SellPCT']:.1f}x")
q1_table = pd.DataFrame({'Average (%)': q1.round(4)})
q1_table.loc['Buy/Sell ratio'] = f"{q1['BuyPCT'] / q1['SellPCT']:.1f}x"
save_table(q1_table, 'q1_summary.png',
           title='Q1 – Full-Sample Recommendation Averages')

# Q2
print("=" * 60)
print("QUESTION 2 – Annual averages")
print("=" * 60)

q2 = df.groupby('Year')[['BuyPCT', 'HoldPCT', 'SellPCT']].mean().round(2)
print(q2.to_string())
save_table(q2, 'q2_annual_table.png',
           title='Q2 – Annual Recommendation Distribution (%)', col_widths=7)

fig, ax = plt.subplots(figsize=(12, 5))
bar_w = 0.6
ax.bar(q2.index, q2['BuyPCT'],  bar_w, label='BuyPCT',  color='#18a05c')
ax.bar(q2.index, q2['HoldPCT'], bar_w, label='HoldPCT',
       color='#e8a020', bottom=q2['BuyPCT'])
ax.bar(q2.index, q2['SellPCT'], bar_w, label='SellPCT',
       color='#d63b3b', bottom=q2['BuyPCT'] + q2['HoldPCT'])
ax.axvspan(2000.5, 2003.5, alpha=0.12, color='gold',
           label='Reform window (2001–2003)')
ax.set_xlabel('Year')
ax.set_ylabel('Average %')
ax.set_title(
    'Q2 – Annual Recommendation Distribution (BuyPCT / HoldPCT / SellPCT)')
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig('q2_annual_recs.png', dpi=150)
plt.close()
print("Saved: q2_annual_recs.png")


# Q3
print("=" * 60)
print("QUESTION 3a – Monthly quintile returns (MR1)")
print("=" * 60)

# a
monthly_mr1 = (
    df.groupby(['YearMonth', 'Q_MeanRec'])['MR1']
    .mean()
    .unstack()
    .rename(columns={1.0: 'Q1', 2.0: 'Q2', 3.0: 'Q3', 4.0: 'Q4', 5.0: 'Q5'})
)

monthly_mr1['Q1_minus_Q5'] = monthly_mr1['Q1'] - monthly_mr1['Q5']

print("First 10 rows of monthly quintile returns:")
print(monthly_mr1.head(10).round(4).to_string())
print(f"\nNumber of months: {len(monthly_mr1)}")

avg_by_q = monthly_mr1[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean()
print("\nTime-series average MR1 by quintile:")
print(avg_by_q.round(4).to_string())
save_table(pd.DataFrame({'Avg MR1 (%)': avg_by_q}),
           'q3a_avg_by_quintile.png', title='Q3a – Average MR1 by Quintile')
save_table(monthly_mr1.head(10).round(4), 'q3a_monthly_preview.png',
           title='Q3a – Monthly Quintile Returns: First 10 Months', col_widths=10)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#18a05c' if v >= 0 else '#d63b3b' for v in monthly_mr1['Q1_minus_Q5']]
ax.bar(range(len(monthly_mr1)),
       monthly_mr1['Q1_minus_Q5'], color=colors, width=1.0)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(monthly_mr1['Q1_minus_Q5'].mean(), color='navy', linewidth=1.5,
           linestyle='--', label=f"Mean = {monthly_mr1['Q1_minus_Q5'].mean():.3f}%")
tick_idx = list(range(0, len(monthly_mr1), 36))
ax.set_xticks(tick_idx)
ax.set_xticklabels([str(monthly_mr1.index[i])[:6]
                   for i in tick_idx], rotation=45, fontsize=9)
ax.set_xlabel('Year-Month')
ax.set_ylabel('Return Difference (%)')
ax.set_title('Q3a – Monthly Q1−Q5 Return Spread (MR1): Buy-minus-Sell Quintile')
ax.legend()
plt.tight_layout()
plt.savefig('q3a_monthly_spread.png', dpi=150)
plt.close()
print("Saved: q3a_monthly_spread.png")


fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(monthly_mr1)), monthly_mr1['Q1_minus_Q5'].cumsum(),
        color='#d63b3b', linewidth=1.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.fill_between(range(len(monthly_mr1)),
                monthly_mr1['Q1_minus_Q5'].cumsum(), 0,
                alpha=0.15, color='#d63b3b')
ax.set_xticks(tick_idx)
ax.set_xticklabels([str(monthly_mr1.index[i])[:6]
                   for i in tick_idx], rotation=45, fontsize=9)
ax.set_xlabel('Year-Month')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Q3a – Cumulative Q1−Q5 Return Spread (MR1): Jan 2000 – Nov 2024')
plt.tight_layout()
plt.savefig('q3a_cumulative.png', dpi=150)
plt.close()
print("Saved: q3a_cumulative.png")


# b
print("=" * 60)
print("QUESTION 3b – Usefulness for portfolio allocation")
print("=" * 60)
mean_diff = monthly_mr1['Q1_minus_Q5'].mean()
print(f"Mean monthly Q1–Q5 return: {mean_diff:.4f}%")


# c
print("=" * 60)
print("QUESTION 3c – t-statistic on Q1−Q5 spread")
print("=" * 60)

diff_series = monthly_mr1['Q1_minus_Q5'].dropna()
n = len(diff_series)
mean_ret = diff_series.mean()
std_ret = diff_series.std()
se = std_ret / np.sqrt(n)
t_stat = mean_ret / se

print(f"N (months)   : {n}")
print(f"Mean return  : {mean_ret:.6f}%")
print(f"Std dev      : {std_ret:.6f}%")
print(f"Std error    : {se:.6f}%")
print(f"t-statistic  : {t_stat:.4f}")
print(f"Critical val : ±1.96 (5% two-sided)")
print(
    f"Reject H₀?   : {'YES' if abs(t_stat) > 1.96 else 'NO'} (|t| = {abs(t_stat):.4f})")
q3c_table = pd.DataFrame({'Value': [n, f"{mean_ret:.6f}%", f"{std_ret:.6f}%", f"{se:.6f}%", f"{t_stat:.4f}", "±1.96", 'YES' if abs(t_stat) > 1.96 else 'NO']},
                         index=['N (months)', 'Mean return', 'Std dev', 'Std error', 't-statistic', 'Critical value (5%)', 'Reject H₀?'])
save_table(q3c_table, 'q3c_tstat.png',
           title='Q3c – t-statistic: Q1−Q5 MR1 Spread')

# Q4
print("=" * 60)
print("QUESTION 4 – Trailing returns (retlag12) by quintile")
print("=" * 60)

q4_avg = df.groupby('Q_MeanRec')['retlag12'].mean()
q4_avg.index = ['Q1 (Buy)', 'Q2', 'Q3', 'Q4', 'Q5 (Sell)']
print("Average trailing 12-month market-adjusted return by quintile:")
print(q4_avg.round(4).to_string())

monthly_rl12 = (
    df.groupby(['YearMonth', 'Q_MeanRec'])['retlag12']
    .mean()
    .unstack()
    .rename(columns={1.0: 'Q1', 2.0: 'Q2', 3.0: 'Q3', 4.0: 'Q4', 5.0: 'Q5'})
)
monthly_rl12['Q1_minus_Q5'] = monthly_rl12['Q1'] - monthly_rl12['Q5']

diff4 = monthly_rl12['Q1_minus_Q5'].dropna()
n4 = len(diff4)
mean4 = diff4.mean()
std4 = diff4.std()
se4 = std4 / np.sqrt(n4)
t_stat4 = mean4 / se4

print(f"\nQ1−Q5 trailing return spread:")
print(f"  N          : {n4}")
print(f"  Mean       : {mean4:.4f} (i.e. {mean4*100:.2f} pp)")
print(f"  Std dev    : {std4:.4f}")
print(f"  Std error  : {se4:.4f}")
print(f"  t-statistic: {t_stat4:.4f}")
print(
    f"  Reject H₀? : {'YES' if abs(t_stat4) > 1.96 else 'NO'} (|t| = {abs(t_stat4):.4f})")
save_table(pd.DataFrame({'Avg retlag12': q4_avg.round(
    4)}), 'q4_avg_by_quintile.png', title='Q4 – Average Trailing 12M Return by Quintile')
q4_stat_table = pd.DataFrame({'Value': [n4, f"{mean4:.4f}", f"{std4:.4f}", f"{se4:.4f}", f"{t_stat4:.4f}", "±1.96", 'YES' if abs(t_stat4) > 1.96 else 'NO']},
                             index=['N (months)', 'Mean spread', 'Std dev', 'Std error', 't-statistic', 'Critical value (5%)', 'Reject H₀?'])
save_table(q4_stat_table, 'q4_tstat.png',
           title='Q4 – t-statistic: Q1−Q5 retlag12 Spread')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(monthly_rl12)), monthly_rl12['Q1'], color='#18a05c',
        linewidth=1.2, label='Q1 (Buy) – trailing return')
ax.plot(range(len(monthly_rl12)), monthly_rl12['Q5'], color='#d63b3b',
        linewidth=1.2, label='Q5 (Sell) – trailing return')
ax.plot(range(len(monthly_rl12)), monthly_rl12['Q1_minus_Q5'], color='#1a3a8f',
        linewidth=2.0, linestyle='--', label='Q1−Q5 Spread')
ax.axhline(0, color='black', linewidth=0.7)
ax.set_xticks(tick_idx)
ax.set_xticklabels([str(monthly_rl12.index[i])[:6]
                   for i in tick_idx], rotation=45, fontsize=9)
ax.set_xlabel('Year-Month')
ax.set_ylabel('Trailing 12M Return (market-adj.)')
ax.set_title('Q4 – Trailing 12-Month Returns by Analyst Quintile (retlag12)')
ax.legend()
plt.tight_layout()
plt.savefig('q4_retlag12.png', dpi=150)
plt.close()
print("Saved: q4_retlag12.png")
