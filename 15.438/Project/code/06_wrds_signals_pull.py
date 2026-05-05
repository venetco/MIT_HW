#!/usr/bin/env python3
"""Pull all 9 significant signals for a list of tickers using WRDS.

Reads credentials from env vars only (never written to disk):
  WRDS_USER     -> WRDS username
  WRDS_PASSWORD -> WRDS password

Usage:
  WRDS_USER=ghadig WRDS_PASSWORD=*** python3 wrds_signals_pull.py \
      --tickers EL,COTY,ELF \
      --as-of 2026-04-24 \
      --outdir "To Upload/wrds_signals"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None


SHARES_OUTSTANDING: dict[str, int] = {
    "EL": 361_727_043,
    "COTY": 880_006_359,
    "ELF": 59_052_239,
}


class WrdsDB:
    def __init__(self, user: str, password: str):
        self.conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user=user,
            password=password,
            sslmode="require",
            connect_timeout=30,
        )
        self.conn.set_session(readonly=True, autocommit=True)

    def raw_sql(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {})
            rows = cur.fetchall()
        return pd.DataFrame(rows)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def connect_to_wrds() -> WrdsDB:
    user = os.environ.get("WRDS_USER")
    password = os.environ.get("WRDS_PASSWORD")
    if not user or not password:
        raise SystemExit("Set WRDS_USER and WRDS_PASSWORD env vars before running.")
    return WrdsDB(user=user, password=password)


def lookup_identifiers(db: WrdsDB, tickers: list[str]) -> pd.DataFrame:
    tickers_upper = [t.upper() for t in tickers]
    sql = (
        "SELECT s.gvkey, s.tic, c.conm, c.cik "
        "FROM comp.security s "
        "JOIN comp.company c USING (gvkey) "
        "WHERE s.tic = ANY(%(tics)s)"
    )
    df = db.raw_sql(sql, params={"tics": tickers_upper})
    if df.empty:
        return df
    return df.drop_duplicates(subset=["tic"], keep="first")


def lookup_permno_map(db: WrdsDB, tickers: list[str]) -> pd.DataFrame:
    tickers_upper = [t.upper() for t in tickers]
    sql = (
        "SELECT permno, ticker AS tic, namedt, nameenddt, comnam "
        "FROM crsp.stocknames "
        "WHERE ticker = ANY(%(tics)s) "
        "ORDER BY ticker, nameenddt DESC"
    )
    df = db.raw_sql(sql, params={"tics": tickers_upper})
    if df.empty:
        return df
    df = df.sort_values(["tic", "nameenddt"], ascending=[True, False])
    return df.drop_duplicates(subset=["tic"], keep="first")


def fetch_funda(db: WrdsDB, gvkeys: list[str]) -> pd.DataFrame:
    sql = (
        "SELECT gvkey, datadate, fyear, tic, conm, "
        "       at, lt, revt, cogs, ni, oancf, "
        "       dltis, dltr, sstk, prstkc, dv "
        "FROM comp.funda "
        "WHERE gvkey = ANY(%(gvkeys)s) "
        "  AND indfmt = 'INDL' AND datafmt = 'STD' "
        "  AND popsrc = 'D' AND consol = 'C' "
        "ORDER BY gvkey, datadate"
    )
    return db.raw_sql(sql, params={"gvkeys": gvkeys})


def fetch_fundq(db: WrdsDB, gvkeys: list[str]) -> pd.DataFrame:
    sql = (
        "SELECT gvkey, datadate, fyearq, fqtr, tic, conm, "
        "       atq, ltq, revtq, cogsq, niq, "
        "       epspxq, epspiq, oancfy, rdq "
        "FROM comp.fundq "
        "WHERE gvkey = ANY(%(gvkeys)s) "
        "  AND indfmt = 'INDL' AND datafmt = 'STD' "
        "  AND popsrc = 'D' AND consol = 'C' "
        "ORDER BY gvkey, datadate"
    )
    return db.raw_sql(sql, params={"gvkeys": gvkeys})


def fetch_msf(db: WrdsDB, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    sql = (
        "SELECT permno, date, prc, ret, shrout, vwretd, sprtrn "
        "FROM crsp.msf "
        "LEFT JOIN crsp.msi USING (date) "
        "WHERE permno = ANY(%(permnos)s) AND date BETWEEN %(start)s AND %(end)s "
        "ORDER BY permno, date"
    )
    return db.raw_sql(sql, params={"permnos": permnos, "start": start, "end": end})


def fetch_dsf(db: WrdsDB, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    sql = (
        "SELECT permno, date, prc, ret "
        "FROM crsp.dsf "
        "WHERE permno = ANY(%(permnos)s) AND date BETWEEN %(start)s AND %(end)s "
        "ORDER BY permno, date"
    )
    return db.raw_sql(sql, params={"permnos": permnos, "start": start, "end": end})


def fetch_ibes_actuals(db: WrdsDB, tickers: list[str]) -> pd.DataFrame:
    sql = (
        "SELECT a.ticker AS ibes_ticker, a.cusip, a.oftic, a.cname, "
        "       a.pends AS period_end, a.pdicity, a.measure, a.value AS actual_eps, "
        "       a.anndats AS announce_date "
        "FROM ibes.actu_epsus a "
        "WHERE a.oftic = ANY(%(tics)s) "
        "  AND a.measure = 'EPS' AND a.pdicity = 'QTR' "
        "ORDER BY a.oftic, a.pends"
    )
    return db.raw_sql(sql, params={"tics": [t.upper() for t in tickers]})


def fetch_ibes_consensus(db: WrdsDB, tickers: list[str]) -> pd.DataFrame:
    sql = (
        "SELECT s.ticker AS ibes_ticker, s.oftic, s.cusip, "
        "       s.fpedats AS period_end, s.statpers, s.measure, s.fpi, "
        "       s.medest, s.meanest, s.numest "
        "FROM ibes.statsum_epsus s "
        "WHERE s.oftic = ANY(%(tics)s) "
        "  AND s.measure = 'EPS' AND s.fpi = '6' "
        "ORDER BY s.oftic, s.fpedats, s.statpers"
    )
    return db.raw_sql(sql, params={"tics": [t.upper() for t in tickers]})


def latest_funda_row(funda: pd.DataFrame, gvkey: str) -> pd.Series | None:
    g = funda[funda["gvkey"] == gvkey].sort_values("datadate")
    if g.empty:
        return None
    return g.iloc[-1]


def latest_fundq_row(fundq: pd.DataFrame, gvkey: str) -> pd.Series | None:
    g = fundq[fundq["gvkey"] == gvkey].sort_values("datadate")
    if g.empty:
        return None
    return g.iloc[-1]


def prior_year_same_quarter_eps(fundq: pd.DataFrame, gvkey: str) -> tuple[float | None, float | None]:
    g = fundq[fundq["gvkey"] == gvkey].sort_values("datadate")
    if g.empty:
        return None, None
    latest = g.iloc[-1]
    fq = latest["fqtr"]
    fy = latest["fyearq"]
    prior = g[(g["fqtr"] == fq) & (g["fyearq"] == fy - 1)]
    eps_latest = latest.get("epspiq") or latest.get("epspxq")
    if prior.empty:
        return eps_latest, None
    eps_prior = prior.iloc[-1].get("epspiq") or prior.iloc[-1].get("epspxq")
    return eps_latest, eps_prior


def fetch_yahoo_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    universe = list({*tickers, "SPY"})
    df = yf.download(universe, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return df
    adj = df["Adj Close"].copy() if "Adj Close" in df else df["Close"].copy()
    adj = adj.dropna(how="all")
    return adj


def compute_volatility_from_yahoo(prices: pd.DataFrame, ticker: str, end: pd.Timestamp) -> float | None:
    if prices.empty or ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    s = s[s.index <= end]
    if len(s) < 60:
        return None
    start = end - pd.Timedelta(days=365)
    s = s[s.index >= start]
    rets = s.pct_change().dropna()
    if len(rets) < 60:
        return None
    return float(rets.std())


def compute_momentum_from_yahoo(prices: pd.DataFrame, ticker: str, end: pd.Timestamp) -> float | None:
    if prices.empty or ticker not in prices.columns or "SPY" not in prices.columns:
        return None
    monthly = prices.resample("ME").last()
    end_month = end.to_period("M").to_timestamp("M")
    monthly = monthly[monthly.index <= end_month]
    monthly_ret = monthly.pct_change().dropna(how="all")
    if monthly_ret.shape[0] < 12:
        return None
    window = monthly_ret.tail(12).iloc[:-1]
    if ticker not in window.columns or "SPY" not in window.columns:
        return None
    if window[ticker].isna().any() or window["SPY"].isna().any():
        return None
    cum_stock = float((1 + window[ticker]).prod() - 1)
    cum_mkt = float((1 + window["SPY"]).prod() - 1)
    return cum_stock - cum_mkt


def compute_book_to_market(
    fundq_row: pd.Series | None,
    funda_row: pd.Series | None,
    market_cap: float | None,
) -> float | None:
    """Use most recent quarterly book equity if available, else annual."""
    at, lt = None, None
    if fundq_row is not None:
        at = fundq_row.get("atq")
        lt = fundq_row.get("ltq")
    if (at is None or lt is None) and funda_row is not None:
        at = funda_row.get("at")
        lt = funda_row.get("lt")
    if at is None or lt is None or market_cap is None or market_cap == 0:
        return None
    book_equity = float(at) - float(lt)
    return book_equity * 1_000_000 / market_cap if market_cap else None


def compute_grossprofit(funda_row: pd.Series) -> float | None:
    revt = funda_row.get("revt")
    cogs = funda_row.get("cogs")
    at = funda_row.get("at")
    if revt is None or cogs is None or at is None or at == 0:
        return None
    return (float(revt) - float(cogs)) / float(at)


def compute_xfin(funda_row: pd.Series) -> float | None:
    fields = ["dltis", "dltr", "sstk", "prstkc", "dv"]
    vals = {k: funda_row.get(k) for k in fields}
    if any(vals[k] is None for k in fields):
        return None
    debt_issued = float(vals["dltis"])
    debt_repaid = float(vals["dltr"])
    equity_issued = float(vals["sstk"])
    equity_repurchased = float(vals["prstkc"])
    dividends = float(vals["dv"])
    at = funda_row.get("at")
    if at is None or at == 0:
        return None
    net_external_financing = (
        debt_issued - debt_repaid + equity_issued - equity_repurchased - dividends
    )
    return net_external_financing / float(at)


def compute_accruals(fundq_row: pd.Series) -> float | None:
    """Optional, signal not significant in multivariate but kept for completeness."""
    ni = fundq_row.get("niq")
    cfo = fundq_row.get("oancfy")
    at = fundq_row.get("atq")
    if ni is None or cfo is None or at is None or at == 0:
        return None
    return (float(ni) - float(cfo)) / float(at)


def compute_fscore(funda: pd.DataFrame, gvkey: str) -> int | None:
    g = funda[funda["gvkey"] == gvkey].sort_values("datadate")
    if len(g) < 2:
        return None
    cur = g.iloc[-1]
    prev = g.iloc[-2]
    score = 0

    def safe_div(a, b):
        try:
            a, b = float(a), float(b)
            return a / b if b else None
        except Exception:
            return None

    roa_cur = safe_div(cur.get("ni"), cur.get("at"))
    roa_prev = safe_div(prev.get("ni"), prev.get("at"))
    cfo_cur = cur.get("oancf")
    accruals_cur = (
        float(cur["ni"]) - float(cur["oancf"])
        if cur.get("ni") is not None and cur.get("oancf") is not None
        else None
    )

    if roa_cur is not None and roa_cur > 0:
        score += 1
    if cfo_cur is not None and cfo_cur > 0:
        score += 1
    if roa_cur is not None and roa_prev is not None and roa_cur > roa_prev:
        score += 1
    if cfo_cur is not None and cur.get("ni") is not None and cfo_cur > float(cur["ni"]):
        score += 1

    long_debt_cur = cur.get("dltis")
    long_debt_prev = prev.get("dltis")
    if long_debt_cur is not None and long_debt_prev is not None and long_debt_cur < long_debt_prev:
        score += 1

    revt_cur = cur.get("revt")
    revt_prev = prev.get("revt")
    cogs_cur = cur.get("cogs")
    cogs_prev = prev.get("cogs")
    at_cur = cur.get("at")
    at_prev = prev.get("at")

    gross_cur = (
        (float(revt_cur) - float(cogs_cur)) / float(revt_cur) if revt_cur else None
    )
    gross_prev = (
        (float(revt_prev) - float(cogs_prev)) / float(revt_prev) if revt_prev else None
    )
    if gross_cur is not None and gross_prev is not None and gross_cur > gross_prev:
        score += 1

    asset_turn_cur = safe_div(revt_cur, at_cur)
    asset_turn_prev = safe_div(revt_prev, at_prev)
    if (
        asset_turn_cur is not None
        and asset_turn_prev is not None
        and asset_turn_cur > asset_turn_prev
    ):
        score += 1

    return score


def compute_earnsurprise_and_sue(
    fundq: pd.DataFrame,
    yahoo_prices: pd.DataFrame,
    actuals: pd.DataFrame,
    consensus: pd.DataFrame,
    permno_map: pd.DataFrame,
    tic: str,
    gvkey: str,
) -> dict:
    out = {"earnsurprise": None, "sue": None, "earnings_announce_date": None, "price_t_minus_5": None}

    g = fundq[fundq["gvkey"] == gvkey].sort_values("datadate")
    if g.empty:
        return out
    latest = g.iloc[-1]
    rdq = latest.get("rdq")
    if rdq is None or pd.isna(rdq):
        return out
    rdq = pd.to_datetime(rdq)

    eps_latest, eps_prior = prior_year_same_quarter_eps(fundq, gvkey)

    if yahoo_prices.empty or tic not in yahoo_prices.columns:
        return out
    s = yahoo_prices[tic].dropna()
    pre = s[s.index < rdq].sort_index()
    if len(pre) < 5:
        return out
    price_5d = float(pre.iloc[-5])
    price_5d_date = pre.index[-5].date().isoformat()
    out["price_t_minus_5"] = price_5d
    out["price_t_minus_5_date"] = price_5d_date
    out["earnings_announce_date"] = rdq.date().isoformat()

    actuals_t = actuals[actuals["oftic"].str.upper() == tic.upper()].copy()
    consensus_t = consensus[consensus["oftic"].str.upper() == tic.upper()].copy()

    consensus_match = pd.DataFrame()
    if not consensus_t.empty:
        consensus_t["period_end"] = pd.to_datetime(consensus_t["period_end"])
        consensus_t["statpers"] = pd.to_datetime(consensus_t["statpers"])
        consensus_t = consensus_t[consensus_t["statpers"] < rdq]
        target_period = pd.to_datetime(latest["datadate"])
        consensus_match = consensus_t[
            (consensus_t["period_end"].dt.year == target_period.year)
            & (consensus_t["period_end"].dt.month == target_period.month)
        ]

    if not consensus_match.empty and eps_latest is not None and price_5d:
        consensus_eps = float(consensus_match.iloc[-1]["meanest"])
        out["earnsurprise"] = (float(eps_latest) - consensus_eps) / price_5d

    if eps_latest is not None and eps_prior is not None and price_5d:
        out["sue"] = (float(eps_latest) - float(eps_prior)) / price_5d

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull all 9 signals via WRDS.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers")
    parser.add_argument("--as-of", required=True, help="As-of date YYYY-MM-DD")
    parser.add_argument(
        "--outdir",
        default="To Upload/wrds_signals",
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    as_of = pd.to_datetime(args.as_of)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to WRDS as user: {os.environ.get('WRDS_USER')}")
    db = connect_to_wrds()

    print(f"Looking up identifiers for: {tickers}")
    ids = lookup_identifiers(db, tickers)
    print(ids)
    permno_map = lookup_permno_map(db, tickers)
    print(permno_map)

    if ids.empty:
        raise SystemExit("No Compustat company records found.")

    gvkeys = ids["gvkey"].dropna().unique().tolist()
    permnos = permno_map["permno"].dropna().astype(int).unique().tolist()

    msf_start = (as_of - pd.DateOffset(months=24)).strftime("%Y-%m-%d")
    dsf_start = (as_of - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    end_iso = as_of.strftime("%Y-%m-%d")
    yf_start = (as_of - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    yf_end = (as_of + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

    print("Fetching FUNDA, FUNDQ, MSF, DSF, IBES, Yahoo prices...")
    funda = fetch_funda(db, gvkeys)
    fundq = fetch_fundq(db, gvkeys)
    msf = fetch_msf(db, permnos, msf_start, end_iso)
    dsf = fetch_dsf(db, permnos, dsf_start, end_iso)
    actuals = fetch_ibes_actuals(db, tickers)
    consensus = fetch_ibes_consensus(db, tickers)
    yahoo_prices = fetch_yahoo_prices(tickers, yf_start, yf_end)

    funda.to_csv(out_dir / "raw_funda.csv", index=False)
    fundq.to_csv(out_dir / "raw_fundq.csv", index=False)
    msf.to_csv(out_dir / "raw_msf.csv", index=False)
    dsf.to_csv(out_dir / "raw_dsf.csv", index=False)
    actuals.to_csv(out_dir / "raw_ibes_actuals.csv", index=False)
    consensus.to_csv(out_dir / "raw_ibes_consensus.csv", index=False)
    if not yahoo_prices.empty:
        yahoo_prices.to_csv(out_dir / "raw_yahoo_prices.csv")

    rows: list[dict] = []
    for tic in tickers:
        ids_row = ids[ids["tic"].str.upper() == tic.upper()]
        if ids_row.empty:
            print(f"WARNING: no Compustat record for {tic}")
            continue
        gvkey = ids_row.iloc[0]["gvkey"]
        permno_row = permno_map[permno_map["tic"].str.upper() == tic.upper()]
        permno = int(permno_row.iloc[0]["permno"]) if not permno_row.empty else None

        funda_row = latest_funda_row(funda, gvkey)
        fundq_row = latest_fundq_row(fundq, gvkey)

        market_cap = None
        latest_close = None
        latest_close_date = None
        if not yahoo_prices.empty and tic in yahoo_prices.columns:
            s = yahoo_prices[tic].dropna()
            s = s[s.index <= as_of]
            if not s.empty:
                latest_close = float(s.iloc[-1])
                latest_close_date = s.index[-1].date().isoformat()
                shares_out = SHARES_OUTSTANDING.get(tic)
                if shares_out and shares_out > 0:
                    market_cap = latest_close * float(shares_out)

        b2m = compute_book_to_market(fundq_row, funda_row, market_cap)
        gp = compute_grossprofit(funda_row) if funda_row is not None else None
        xfin = compute_xfin(funda_row) if funda_row is not None else None
        fscore = compute_fscore(funda, gvkey)
        accruals = compute_accruals(fundq_row) if fundq_row is not None else None
        vol = compute_volatility_from_yahoo(yahoo_prices, tic, as_of)
        mom = compute_momentum_from_yahoo(yahoo_prices, tic, as_of)

        ee = compute_earnsurprise_and_sue(
            fundq, yahoo_prices, actuals, consensus, permno_map, tic, gvkey
        )

        rows.append(
            {
                "ticker": tic,
                "gvkey": gvkey,
                "permno": permno,
                "as_of": end_iso,
                "latest_close_price": latest_close,
                "latest_close_date": latest_close_date,
                "shares_outstanding": SHARES_OUTSTANDING.get(tic),
                "market_cap_usd": market_cap,
                "book_to_market": b2m,
                "gross_profitability": gp,
                "xfin": xfin,
                "fscore": fscore,
                "accruals": accruals,
                "volatility_daily": vol,
                "momentum_market_adj": mom,
                "earnsurprise": ee["earnsurprise"],
                "sue": ee["sue"],
                "earnings_announce_date": ee["earnings_announce_date"],
                "price_t_minus_5": ee["price_t_minus_5"],
                "price_t_minus_5_date": ee.get("price_t_minus_5_date"),
                "funda_datadate": funda_row["datadate"] if funda_row is not None else None,
                "fundq_datadate": fundq_row["datadate"] if fundq_row is not None else None,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "wrds_signal_values.csv"
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
