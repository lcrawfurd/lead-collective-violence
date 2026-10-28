"""
00_download_acled.py  --  Download one ACLED export for the whole paper (run by hand, not
in run_all.sh).

Downloads every ACLED event worldwide, 1997 to END_YR (config.py), through the ACLED API,
year by year, and writes:
  data/ACLED_global.csv          all events (used for Senegal, the strategic-development
                                 placebo, the map and the sub-Saharan totals)
  data/acled_kenya.csv           Kenya rows of the same export
  data/acled_ghana.csv           Ghana rows of the same export
  data/ACLED_export_info.txt     download date, years, row counts, API endpoint
Because all three files come from one download, every script uses the same ACLED version.

Credentials are read from environment variables, never from this file:
  export ACLED_EMAIL="you@example.org"
  export ACLED_PASSWORD="..."
  PATH="/usr/local/bin:$PATH" python3 code/00_download_acled.py

Run time: roughly an hour (about 2.5 million events at 5,000 per request). The script saves
each year to data/acled_raw/ and skips years already saved, so it can be stopped and resumed.
API documentation: https://acleddata.com/api-documentation/getting-started
"""
import datetime as dt
import io
import os
import sys
import time

import pandas as pd
import requests

from config import DATA, END_YR

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"
FIRST_YR = 1997
PAGE = 5000
RAW = DATA / "acled_raw"


def token():
    email, pw = os.environ.get("ACLED_EMAIL"), os.environ.get("ACLED_PASSWORD")
    if not email or not pw:
        sys.exit("Set ACLED_EMAIL and ACLED_PASSWORD in your shell first (see the header).")
    r = requests.post(TOKEN_URL, data={"username": email, "password": pw, "grant_type": "password",
                                       "client_id": "acled", "scope": "authenticated"}, timeout=60)
    if r.status_code != 200:
        # ACLED's error body says why (e.g. invalid_grant = wrong email or password); it never
        # contains the password itself.
        sys.exit(f"Login failed ({r.status_code}): {r.text[:300]}\n"
                 f"Email used: {email}; password length: {len(pw)} characters.")
    return r.json()["access_token"]


TOK = {"value": None}


def get_year(year):
    out, page = [], 1
    while True:
        for attempt in range(5):
            r = requests.get(READ_URL, params={"_format": "csv", "year": year, "limit": PAGE, "page": page},
                             headers={"Authorization": f"Bearer {TOK['value']}"}, timeout=300)
            if r.status_code == 200:
                break
            if r.status_code == 401:            # token expired: log in again and retry this page
                print("  token expired; logging in again")
                TOK["value"] = token()
                continue
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False) if r.text.strip() else pd.DataFrame()
        if df.empty:
            break
        out.append(df)
        print(f"  {year} page {page}: {len(df):,} events")
        if len(df) < PAGE:
            break
        page += 1
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    RAW.mkdir(exist_ok=True)
    TOK["value"] = token()
    for year in range(FIRST_YR, END_YR + 1):
        f = RAW / f"acled_{year}.csv"
        if f.exists():
            continue
        d = get_year(year)
        if d.empty:
            sys.exit(f"No events returned for {year}; check your ACLED access level.")
        d.to_csv(f, index=False)
    g = pd.concat([pd.read_csv(RAW / f"acled_{y}.csv", low_memory=False)
                   for y in range(FIRST_YR, END_YR + 1)], ignore_index=True)
    if g.event_id_cnty.duplicated().any():
        sys.exit("Duplicate event ids across pages; re-run after deleting data/acled_raw/.")
    g.to_csv(DATA / "ACLED_global.csv", index=False)
    for cn in ["Kenya", "Ghana"]:
        g[g.country == cn].to_csv(DATA / f"acled_{cn.lower()}.csv", index=False)
    counts = g[g.country.isin(["Kenya", "Ghana", "Senegal"])].groupby("country").size()
    (DATA / "ACLED_export_info.txt").write_text(
        f"ACLED export downloaded {dt.date.today()} via {READ_URL}\n"
        f"Years {FIRST_YR}-{END_YR}, all countries, {len(g):,} events\n"
        + "\n".join(f"{c}: {n:,} events" for c, n in counts.items()) + "\n")
    print((DATA / "ACLED_export_info.txt").read_text())


if __name__ == "__main__":
    main()
