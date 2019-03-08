from mesa import Agent, Model
from mesa.time import RandomActivation, StagedActivation
from mesa.datacollection import DataCollector
import numpy as np
import random
from collections import OrderedDict
from operator import itemgetter

def get_highest_bid(model):
    if len(model.bidders):
        highest_bid = model.bidders.popitem()
        price = highest_bid[1]
        model.bidders[highest_bid[0]] = highest_bid[1]
        return price
    else:
        return np.nan

def get_lowest_ask(model):
    if len(model.askers):
        lowest_ask = model.askers.popitem()
        price = lowest_ask[1]
        model.askers[lowest_ask[0]] = lowest_ask[1]
        return price
    else:
        return np.nan

def compute_volume(model):
    return model.volume

class PredictionMarketModel(Model):
    """A simple model of a market where people bid/ask for prediciton shares.
    """

    def __init__(self, N=100):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"bid_price": get_highest_bid, "ask_price": get_lowest_ask, 
                             "volume": compute_volume},
            agent_reporters={"Certainty": "prob", "Bids":"bid", "Asks":"ask"}
            )
        self.bidders = OrderedDict()
        self.askers = OrderedDict()
        self.volume = 0
        # Create agents
        print('creating {} agents..'.format(self.num_agents))
        for i in range(self.num_agents):
            # Create probability distribution
            #prob = np.random.normal(mu, sigma)
            #prob = np.random.uniform(0.01, 0.99)
            if random.randint(0,1):
                prob = np.random.normal(0.83, 0.05)
            else:
                prob = np.random.normal(0.53, 0.05)
            a = PredictionAgent(prob, i, self)
            self.schedule.add(a)

        print('running model...')
        self.running = True

    def step(self):
        self.volume = 0
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()


class PredictionAgent(Agent):
    """ An agent with fixed confidence."""
    def __init__(self, p, unique_id, model):
        super().__init__(unique_id, model)
        p = round(p,2)
        if p >= 1:
            p = 0.99
        elif p <= 0:
            p = 0.01
        self.prob = p
        self.bid = 0.
        self.ask = 1.
        self.model.bidders[self.unique_id] = self.bid
        self.model.askers[self.unique_id] = self.ask

    def buy(self):
        if len(self.model.askers) > 0: #needs to be people selling first of all...
            lowest_price = get_lowest_ask(self.model)
            if lowest_price <= self.bid:
                # If there is an ask out there to fill your bid, buy it and place a new bid.
                self.model.askers.popitem()
                new_bid = self.update_bid(lowest_price, increase_bid=False)
                self.place_bid(new_bid)
                self.model.volume = self.model.volume + 1
            else:
                # If not, raise your bid
                new_bid = self.update_bid(lowest_price, increase_bid=True)
                self.place_bid(new_bid)

    def sell(self):
        if len(self.model.bidders) > 0: #need buyers in the market first...
            highest_price = get_highest_bid(self.model)
            if highest_price >= self.ask:
                # If bid is higher/equal to your asking price, sell and place new ask
                self.model.bidders.popitem()
                new_ask = self.update_ask(highest_price, decrease_ask=False)
                self.place_ask(new_ask)
                self.model.volume = self.model.volume + 1
            else:
                #if not, lower your ask
                new_ask = self.update_ask(highest_price, decrease_ask=True)
                self.place_ask(new_ask)

    def update_bid(self, lowest_ask, increase_bid):
        #lowest you can bid is 0, highest you can bid is self.prob-0.01
        min_bid = 0.00
        max_bid = self.prob-0.01
        #max_bid = self.prob-0.01 if lowest_ask >= self.prob else lowest_ask
        #if you're already at the maximum you're willing to pay, stay there
        if increase_bid:
            if self.bid+0.01 >= max_bid:
                return max_bid
            else:
                #rand_bid = random.randrange(int(self.bid*100), int((max_bid+0.01)*100))
                #return round(rand_bid/100.,2)
                return self.bid+0.01
        else:
            if self.bid-0.01 < min_bid:
                return min_bid
            else:
                #rand_bid = random.randrange(int(min_bid*100), int(self.bid*100))
                #return round(rand_bid/100.,2)
                return self.bid-0.01

    def update_ask(self, highest_bid, decrease_ask):
        #highest you can ask is 1, lowest you can ask is self.prob+0.01
        max_ask = 1.00
        min_ask = self.prob+0.01
        #min_ask = self.prob+0.01 if highest_bid <= self.prob else highest_bid
        #if you're already at the minimum you're willing to sell at, stay there
        if decrease_ask:
            if self.ask-0.01 <= min_ask:
                return min_ask
            else:
                #rand_ask = random.randrange(int(min_ask*100), int((self.ask)*100))
                #return round(rand_ask/100.,2)
                return self.ask-0.01
        else:
            if self.ask+0.01 > max_ask:
                return max_ask
            else:
                #rand_ask = random.randrange(int(self.ask*100), int((max_ask+0.01)*100))
                #return round(rand_ask/100.,2)
                return self.ask+0.01

    def place_bid(self, bid):
        self.bid = bid
        self.model.bidders[self.unique_id] = self.bid
        self.model.bidders = OrderedDict(sorted(self.model.bidders.items(), 
                                    key=itemgetter(1), reverse=False))

    def place_ask(self, ask):
        self.ask = ask
        self.model.askers[self.unique_id] = self.ask
        self.model.askers = OrderedDict(sorted(self.model.askers.items(), 
                                    key=itemgetter(1), reverse=True))

    def step(self):
        #if random.randint(0, 1):
        self.buy()
        self.sell()

