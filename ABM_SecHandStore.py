import random
import operator
import numpy as np
from scipy.stats import truncnorm

from mesa import Agent

from ABM_Smartphone import Smartphone


class SecondHandStore(Agent):
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

    def __init__(self, model, unique_id):
        super().__init__(model)
        self.unique_id = unique_id
        self.num_used_products = 15
        self.inventory = []

        self.initialize_inventory()

        self.avg_product_price = 0
        self.max_time_held = 36

    def initialize_inventory(self):
        """
        Initialize the store's inventory with a set number of used smartphones.
        Each smartphone is created with randomized performance and time held values.
        """
        for _ in range(self.num_used_products):
            self.inventory.append(
                Smartphone(
                    model=self.model,
                    is_new=False,
                    producer_id=self.model.product_id_price['id'],
                    user_id=self.unique_id,
                    performance=random.uniform(0.7, 1),
                    time_held=random.randint(0, 24),
                    demand_used=0.3,
                    product_price=self.model.product_id_price['price'],
                    initial_repair_cost=500,
                    decay_rate=0.1
                )
            )

    def trade_with_consumer_buy(self, smartphone):
        smartphone.repair_product()
        smartphone.time_held = 0
        self.inventory.append(smartphone)

    def calculate_sell_price(self, smartphone):
        """
        """
        smartphone.calculate_used_market_price()
    
    def trade_with_cunsumer_resell(self):
        """
        """
        pass
    
    def step(self):
        for smartphone in self.inventory:
            smartphone.update_time_held()
            if smartphone.time_held >= self.max_time_held:
                # random pick a recycler
                # smartphone.recycle_product(new_owner_id)
                self.inventory.remove(smartphone)
        # print(f"SecondHandStore {self.unique_id} doing.")