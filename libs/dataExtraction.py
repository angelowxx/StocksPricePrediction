import akshare as ak


class StockDataExtraction():


    def extractStockInfosWHole(self, symbol = "sh000001"):
        historical_data = ak.stock_zh_index_daily(symbol=symbol)
        #monthly_stock = ak.stock_zh_index_monthly(symbol=stockSymbol, adjust="qfq", start_date=start_date, end_date=end_date)
        return historical_data

    def extractStockInfosWise(self, stockSymbol, start_date="19900101", end_date="21000118"):
        # Example: Fetch daily data for a Chinese stock (e.g., Kweichow Moutai, code: 600519)
        daily_stock = ak.stock_zh_a_daily(symbol=stockSymbol, adjust="qfq", start_date=start_date, end_date=end_date)
        #monthly_stock = ak.stock_zh_index_monthly(symbol=stockSymbol, adjust="qfq", start_date=start_date, end_date=end_date)
        return daily_stock

    def saveStockIndosAsCsvUpToDate(self, data, stockSymbol):
        data.to_csv(f"./data/{stockSymbol}.csv", index=False)  # Save to CSV without index
