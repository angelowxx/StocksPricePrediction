import pandas as pd

class stockSymbol:
    stocks_data = {
        "Name": ["YaoMinKangDe", "HaiKangWeiShi"],
        "Symbol": ["sh603259", "sz002415"]
    }

    stocks = pd.DataFrame(stocks_data)

    market_data = {
        "Name": ["ShangZhengZhiShu", "ShenZhenChengZhi"],
        "Symbol": ["sh000001", "sz399001"]
    }

    markets = pd.DataFrame(market_data)