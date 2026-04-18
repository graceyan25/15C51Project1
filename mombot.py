import pandas as pd

daily_returns = pd.read_csv("dow30_returns.csv", index_col="date", parse_dates=True)
# print(daily_returns.head(10))
monthly_returns = daily_returns.resample("MS").first()
# print(monthly_returns.head(10))

momentum = monthly_returns.pct_change(12).shift(1)
# print(momentum.head(20))

def get_tickers_date(date):
    return momentum.columns[monthly_returns.loc[date].notna()]

portfolio = pd.DataFrame(index=monthly_returns.index, 
                         columns=monthly_returns.columns)
for date in momentum.index[12:]:
    pf_date = momentum.loc[date]
    rel_tickers = get_tickers_date(date)
    pf_date = pf_date[pf_date.index.isin(rel_tickers)]
    top5 = pf_date.nlargest(5).index
    portfolio.loc[date] = 0
    portfolio.loc[date, top5] = 1/5

portfolio.to_csv("portfolio.csv")

# returns = monthly_returns.pct_change().shift(-1)
# strategy_returns = (portfolio * returns).sum(axis=1)