import pandas as pd

class smp500_tickers: 
    def __init__(self):
        self.url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.tickers = None
        self.path = ".../data/tickers-list" # Placeholder

    def get_tickers(self):
        tables = pd.read_html(self.url)
        self.tickers = tables[0]['Symbol'].tolist()

    def save_data(self):
        """
        Likely parquet after using requests
        """
        pass

    def main(self):
        pass

    
