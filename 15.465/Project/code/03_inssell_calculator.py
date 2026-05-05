#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


OPEN_MARKET_BUY_CODES = {"P"}
OPEN_MARKET_SELL_CODES = {"S"}

CEO_KEYWORDS = ("ceo", "chief executive", "president", "managing director")
CFO_KEYWORDS = (
    "cfo",
    "chief financial",
    "finance director",
    "vp of finance",
    "vp finance",
    "finance officer",
)
COO_KEYWORDS = (
    "coo",
    "chief operating",
    "operations officer",
    "evp operations",
    "vp operations",
)


@dataclass
class CompanyInputs:
    ticker: str
    csv_path: Path
    shares_outstanding: float


def parse_bool(value: str) -> bool:
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in {"1", "true", "t", "yes", "y"}


def parse_float(value: str) -> float:
    if value is None:
        return 0.0
    v = str(value).strip().replace(",", "")
    if not v:
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def role_keyword_match(title: str, keywords: tuple[str, ...]) -> bool:
    if not title:
        return False
    t = title.lower()
    return any(k in t for k in keywords)


def classify_role(row: dict) -> str:
    is_director = parse_bool(row.get("is_director", ""))
    is_officer = parse_bool(row.get("is_officer", ""))
    title = (row.get("officer_title") or "").strip()

    if is_officer and role_keyword_match(title, CEO_KEYWORDS):
        return "CEO"
    if is_officer and role_keyword_match(title, CFO_KEYWORDS):
        return "CFO"
    if is_officer and role_keyword_match(title, COO_KEYWORDS):
        return "COO"
    if is_director:
        return "Director"
    return ""


def is_relevant_transaction(row: dict) -> bool:
    code = (row.get("transaction_code") or "").strip().upper()
    if code not in OPEN_MARKET_BUY_CODES | OPEN_MARKET_SELL_CODES:
        return False
    derivative = (row.get("derivative_flag") or "").strip().upper()
    if derivative == "Y":
        return False
    return True


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def deduplicate_amendments(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (
            (r.get("owner_name") or "").strip().lower(),
            (r.get("transaction_date") or "").strip(),
            (r.get("transaction_code") or "").strip().upper(),
            (r.get("derivative_flag") or "").strip().upper(),
            (r.get("security_title") or "").strip().lower(),
            (r.get("transaction_shares") or "").strip(),
            (r.get("transaction_price") or "").strip(),
        )
        if key in seen:
            existing = seen[key]
            try:
                if r.get("filing_date", "") > existing.get("filing_date", ""):
                    seen[key] = r
            except Exception:
                continue
        else:
            seen[key] = r
    return list(seen.values())


def compute_inssell(rows: list[dict]) -> dict:
    filtered = []
    for r in rows:
        role = classify_role(r)
        if not role:
            continue
        if not is_relevant_transaction(r):
            continue
        r2 = dict(r)
        r2["role"] = role
        filtered.append(r2)

    filtered = deduplicate_amendments(filtered)

    sells_total = 0.0
    buys_total = 0.0
    for r in filtered:
        code = (r.get("transaction_code") or "").strip().upper()
        shares = parse_float(r.get("transaction_shares", "0"))
        if code in OPEN_MARKET_SELL_CODES:
            sells_total += shares
        elif code in OPEN_MARKET_BUY_CODES:
            buys_total += shares

    net_sold = sells_total - buys_total
    return {
        "filtered_rows": filtered,
        "shares_sold": sells_total,
        "shares_bought": buys_total,
        "net_shares_sold": net_sold,
    }


def write_filtered_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_for_company(company: CompanyInputs, output_dir: Path) -> dict:
    rows = read_rows(company.csv_path)
    result = compute_inssell(rows)
    inssell = (
        result["net_shares_sold"] / company.shares_outstanding
        if company.shares_outstanding > 0
        else 0.0
    )

    out_csv = output_dir / company.ticker / "inssell_filtered_transactions.csv"
    write_filtered_csv(out_csv, result["filtered_rows"])

    summary = {
        "ticker": company.ticker,
        "shares_outstanding": company.shares_outstanding,
        "shares_bought": result["shares_bought"],
        "shares_sold": result["shares_sold"],
        "net_shares_sold": result["net_shares_sold"],
        "inssell": inssell,
        "filtered_transactions": len(result["filtered_rows"]),
        "filtered_csv": str(out_csv),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute inssell signal from parsed Form 4 transactions."
    )
    parser.add_argument(
        "--root",
        default="To Upload/sec_form4",
        help="Root folder of scraper output",
    )
    parser.add_argument(
        "--company",
        action="append",
        required=True,
        help="Repeatable: ticker:shares_outstanding (e.g. EL:357000000)",
    )
    parser.add_argument(
        "--output-dir",
        default="To Upload/inssell",
        help="Output folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for entry in args.company:
        if ":" not in entry:
            raise SystemExit(f"--company must be ticker:shares_out, got: {entry}")
        ticker, shares_str = entry.split(":", 1)
        ticker = ticker.strip().upper()
        shares_out = parse_float(shares_str)
        csv_path = root / ticker / "parsed" / "form4_transactions.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping {ticker}.")
            continue
        company = CompanyInputs(
            ticker=ticker,
            csv_path=csv_path,
            shares_outstanding=shares_out,
        )
        summary = run_for_company(company, out_dir)
        summaries.append(summary)

    if not summaries:
        print("No summaries produced.")
        return

    summary_csv = out_dir / "inssell_summary.csv"
    fields = list(summaries[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print("Inssell summary:")
    for s in summaries:
        print(
            f"  {s['ticker']}: bought={s['shares_bought']:,.0f}, "
            f"sold={s['shares_sold']:,.0f}, net_sold={s['net_shares_sold']:,.0f}, "
            f"shares_out={s['shares_outstanding']:,.0f}, "
            f"inssell={s['inssell']:.6e} ({s['filtered_transactions']} txns)"
        )
    print("Wrote:", summary_csv)


if __name__ == "__main__":
    main()
