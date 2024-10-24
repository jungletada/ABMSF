from mesa import Agent
import numpy as np
from ABM_CE_RecyclerAgents import Recyclers
import operator
from scipy.stats import truncnorm


class SecondHandStores(Agent):
    """
    A refurbisher which repairs modules (and eventually discard them), improve
    its processes and act as an intermediary between other actors.

    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_Model)
        original_repairing_cost (a list for a triangular distribution) ($/fu),
            (default=[0.1, 0.45, 0.28]). From Tsanakas et al. 2019.
        init_eol_rate (dictionary with initial end-of-life (EOL) ratios),
            (default={"repair": 0.005, "sell": 0.02, "recycle": 0.1,
            "landfill": 0.4375, "hoard": 0.4375}). From Monteiro Lunardi
            et al 2018 and European Commission (2015).
        repairing_learning_shape_factor, (default=-0.31). Estimated with data
            on repairing costs at different scales from JRC 2019.
        scndhand_mkt_pric_rate (a list for a triangular distribution) (ratio),
            (default=[0.4, 1, 0.7]). From unpublished study Wang et al.
        refurbisher_margin (ratio), (default=[0.03, 0.45, 0.24]). From Duvan
            & Aykaç 2008 and www.investopedia.com (accessed 03/2020).
        max_storage (a list for a triangular distribution) (years), (default=
            [1, 8, 4]). From Wilson et al. 2017.

    """

    def __init__(self, unique_id, model, ):
        super().__init__(unique_id, model)
        self.stocks = []
        self.max_time_held = 36

    def bought_used_product_from_consumer(self, smartphone):
        smartphone.repair_product()
        smartphone.time_held = 0
        self.stocks.append(smartphone)

    def calculate_sell_price(self, smartphone):
        """
        """
        smartphone.calculate_used_market_price()
    
    def sell_used_product_to_cunsumer(self):
        """"""
        pass
    
    def step(self):
        for smartphone in self.stocks:
            smartphone.update_time_held()
            if smartphone.time_held >= self.max_time_held:
                # random pick a recycler
                smartphone.recycle_product(new_owner_id)
                self.stocks.remove(smartphone)