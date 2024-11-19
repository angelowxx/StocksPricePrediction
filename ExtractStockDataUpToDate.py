from libs.dataExtraction import StockDataExtraction
from libs.StockSymbol import stockSymbol

def update_markets_stocks_csv():

    s = StockDataExtraction()
    stock_infos = stockSymbol()
    stocks = stock_infos.stocks
    markets = stock_infos.markets
    for index, stock in stocks.iterrows():
        symbol = stock["Symbol"]
        data = s.extractStockInfosWise(symbol)
        s.saveStockIndosAsCsvUpToDate(data, symbol)

    for index, market in markets.iterrows():
        symbol = market["Symbol"]
        data = s.extractStockInfosWHole(symbol)
        s.saveStockIndosAsCsvUpToDate(data, symbol)

if __name__ == '__main__':
    update_markets_stocks_csv()