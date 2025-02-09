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

    def __init__(self, model, unique_id, init_num_used_products=25):
        super().__init__(model)
        self.unique_id = unique_id
        self.num_used_products = init_num_used_products
        self.inventory = []
        self.repair_cost = 0
        self.customers = []
        self.avg_product_price = 0
        self.max_time_held = 36
        self.num_sales = 0
        self.num_buy_from = 0
        self.initialize_inventory()

    def initialize_inventory(self):
        """
        Initialize the store's inventory with a set number of used smartphones.
        Each smartphone is created with randomized performance and time held values.
        """
        producer_ids = list(self.model.new_product_id_price.keys())
        for _ in range(self.num_used_products):
            choose_id = random.choice(producer_ids)
            product_price = self.model.new_product_id_price[choose_id]
            self.inventory.append(
                Smartphone(
                    model=self.model,
                    is_new=False,
                    producer_id=choose_id,
                    user_id=None,
                    performance=random.uniform(0.7, 1),
                    time_held=random.randint(0, 24),
                    demand_used=0.3,
                    product_price=product_price,
                    initial_repair_cost=500,
                    decay_rate=0.1
                )
            )

    def buy_from_consumer(self, smartphone:Smartphone, consumer_id:int):
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
        smartphone.is_new = False
        smartphone.calculate_sechnd_market_price() # decide the used product price
        self.inventory.append(smartphone) # add to the inventory
        self.customers.append(consumer_id)
        self.num_buy_from += 1
        # print(f'Second Market Trade: before {consumer_id}, after { smartphone.user_id}')

    def trade_with_consumer_resell(self, smartphone_id:int):
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
        self.num_sales += 1
        for product in self.inventory:
            if product.unique_id == smartphone_id:
                self.inventory.remove(product)

    def send_smartphone_to_recycler(self, smartphone:Smartphone):
        """
        Send a smartphone from the store's inventory to a recycler.
        
        Args:
            smartphone (Smartphone): The smartphone to be sent for recycling.
            
        This function randomly selects a recycler from available recyclers and transfers 
        the smartphone to them. The recycler will handle the actual recycling process and
        update the smartphone's ownership status. This is typically called when a smartphone
        has been in inventory too long (exceeds max_time_held).
        """
        recyclers = [agent for agent in self.model.agents
                             if isinstance(agent, Recycler)]
        trader = random.choice(recyclers)
        #######################
        ## Trade with Recycler
        #######################
        trader.recycle_from_secondhandstore(smartphone, self.unique_id)
        # print(f'Second Market Trade: before {self.unique_id}, after { smartphone.user_id}')

    def step(self):
        """
        Main simulation step for the second-hand store.
        
        Updates the time held for all smartphones in inventory and sends phones that have
        exceeded max_time_held to recyclers. For each smartphone in inventory:
        - Calls update_time_held() to increment time and degrade performance
        - If time_held >= max_time_held, sends the phone to a randomly chosen recycler
          and removes it from inventory
        """
        for smartphone in self.inventory:
            smartphone.update_time_held()
            if smartphone.time_held >= self.max_time_held:
                # random pick a recycler
                self.send_smartphone_to_recycler(smartphone)
                self.inventory.remove(smartphone)
        # update the every year average product price
        if len(self.inventory) != 0:
            product_price = [smartphone.secondhand_market_price
                                for smartphone in self.inventory]
            self.avg_product_price = np.mean(product_price)
        # print(f"SecondHandStore {self.unique_id} doing.")