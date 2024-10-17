import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Market.market import PutStockOption, BSParams, MarketState, price
from tqdm import tqdm
class OptionDataset(Dataset):
    def __init__(self, num_samples, params):
        """
        num_samples: Number of objects
        params_range: should be dict(
                                    T = tuple([minT,maxT]) ,
                                    K = tuple([minK,maxK]) ,
                                    sigma = tuple([minsigma,maxsigma]) ,
                                    r = tuple([minr,maxr]) ,
                                    S = tuple([mins,maxs]) ,
                                    )
        """
        self.num_samples = num_samples
        self.params = params
        self.data = self.generate_data()
    def generate_data(self):
        data = []
        for _ in tqdm(range(self.num_samples)):
            T = np.random.uniform(self.params['T'][0], self.params['T'][1])  # expiration_time
            K = np.random.uniform(self.params['K'][0], self.params['K'][1])  # strike_price
            sigma = np.random.uniform(self.params['sigma'][0], self.params['sigma'][1])  # volatility
            t = np.random.uniform(0,T)  # current time
            r = np.random.uniform(self.params['r'][0], self.params['r'][1])  # interest_rate
            S = np.random.uniform(self.params['S'][0], self.params['S'][1])  # stock_price
            
            puts = PutStockOption(strike_price=K, expiration_time=T)
            ms = MarketState(stock_price=S, interest_rate=r, time = t)
            params = BSParams(volatility=sigma)
            option_price = price(puts, ms, params)
            data.append([S, K, T, t, r, sigma, option_price])
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.tensor(sample[:-1], dtype=torch.float32)
        target = torch.tensor(sample[-1], dtype=torch.float32)
        return features, target