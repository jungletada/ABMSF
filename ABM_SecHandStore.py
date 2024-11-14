import random
import operator
import numpy as np
from scipy.stats import truncnorm
from mesa import Agent

from ABM_Smartphone import Smartphone
from ABM_Recycler import Recycler


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

    def __init__(
        self, model, unique_id, init_num_used_products=25):
        super().__init__(model)
        self.unique_id = unique_id
        self.num_used_products = init_num_used_products
        self.inventory = []
        self.repair_cost = 0
        self.customer = []
        self.initialize_inventory()

        self.avg_product_price = 0
        self.max_time_held = 36

    def initialize_inventory(self):
        """
        Initialize the store's inventory with a set number of used smartphones.
        Each smartphone is created with randomized performance and time held values.
        """
        producer_ids = self.model.new_product_id_price.keys()
        choose_id = random.choice(list(producer_ids))
        product_price = self.model.new_product_id_price[choose_id]
        for _ in range(self.num_used_products):
            self.inventory.append(
                Smartphone(
                    model=self.model,
                    is_new=False,
                    producer_id=choose_id,
                    user_id=self.unique_id,
                    performance=random.uniform(0.7, 1),
                    time_held=random.randint(0, 24),
                    demand_used=0.3,
                    product_price=product_price,
                    initial_repair_cost=500,
                    decay_rate=0.1
                )
            )

    def calculate_resell_price(self, smartphone):
        """
        
        """
        
        smartphone.calculate_used_market_price()
        
    def buy_from_consumer(self, smartphone, consumer_id):
        """
        Purchase a used smartphone from a consumer and add it to store inventory.
        
        Args:
            smartphone (Smartphone): The smartphone being sold to the store by the consumer.
            
        This function repairs the smartphone, transfers ownership to the store by updating 
        the smartphone's user_id, and adds it to the store's inventory. The repair is done
        automatically before adding to inventory to ensure all store stock is in good condition.
        """
        smartphone.repair_product() # repair the used smartphone for reselling
        self.repair_cost += smartphone.calculate_repair_cost() # update the repairing cost
        smartphone.user_id = self.unique_id # change the owner
        smartphone.calculate_used_market_price() # decide the used product price
        self.inventory.append(smartphone) # add to the inventory
        self.customer.append(consumer_id)
        print(f'Second Market Trade: before {consumer_id}, after { smartphone.user_id}')
    
    def trade_with_consumer_resell(self, consumer_id):
        """
        Simulate the sale of a used smartphone from the store's inventory to a consumer.
        
        Args:
            consumer_id (int): The unique ID of the consumer purchasing the smartphone.
            
        Returns:
            Smartphone: The smartphone being sold to the consumer, or None if inventory is empty.
            
        This function removes a smartphone from the store's inventory and transfers ownership 
        to the purchasing consumer by updating the smartphone's user_id. If the store's 
        inventory is empty, returns None to indicate no smartphone is available for sale.
        """
        if len(self.inventory) == 0:
            return None
        smartphone = self.inventory.pop(0) # 此处是弹出第一个手机
        smartphone.user_id = consumer_id
        return smartphone
    
    def send_to_recycler(self, smartphone):
        """
        """
        recyclers = [agent for agent in self.model.agents 
                             if isinstance(agent, Recycler)]
        trader = random.choice(recyclers)
        #######################
        ## Trade with Recycler
        #######################
        trader.recycle_from_secondhand(smartphone, self.unique_id)
        print(f'Second Market Trade: before {self.unique_id}, after { smartphone.user_id}')
        
        
    def step(self):
        for smartphone in self.inventory:
            smartphone.update_time_held()
            if smartphone.time_held >= self.max_time_held:
                # random pick a recycler
                # smartphone.recycle_product(new_owner_id)
                self.send_to_recycler(smartphone)
                self.inventory.remove(smartphone)
        # print(f"SecondHandStore {self.unique_id} doing.")