The X dig loop and cross-run ledger (from the pixie GetXAPI port) are now
tunable via env: `LAST30DAYS_X_DIG_QUERIES` sets follow-up queries per dig
round (default 3), `LAST30DAYS_X_LEDGER_MAX_QUERIES` bounds remembered queries
(default 400), and `LAST30DAYS_X_LEDGER_MAX_IDS` bounds post ids stored per
query (default 2000); `0` lifts either ledger cap.
