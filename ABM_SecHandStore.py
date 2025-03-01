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
    """

    def __init__(self, model, unique_id, init_num_used_products=25):
        super().__init__(model)
        self.unique_id = unique_id
        self.init_num_used_products = init_num_used_products
        self.num_used_products = init_num_used_products
        self.inventory = []
        self.repair_cost = 0
        self.customers = []
        self.avg_product_price = 0
        self.max_time_held = 60
        self.num_sales = 0
        self.num_buy_from = 0
        self.replenish_interval = random.randint(0, 11)
        self.initialize_inventory()

    def initialize_inventory(self):
        """
        Initialize the store's inventory with a set number of used smartphones.
        Each smartphone is created with randomized performance and time held values.
        """
        producer_ids = list(self.model.new_product_id_price.keys())
        choose_ids = random.choices(producer_ids, k=self.init_num_used_products)
        for choose_id in choose_ids:
            product_price = self.model.new_product_id_price[choose_id]
            product = Smartphone(
                    model=self.model,
                    is_new=False,
                    producer_id=choose_id,
                    user_id=None,
                    performance=random.uniform(0.7, 1.0),
                    time_held=random.randint(1, 24),
                    product_price=product_price,
                    initial_repair_cost=500,)
            self.inventory.append(product)

    def replenish(self):
        """
        This function simulates the replenishment of used smartphones in the store's inventory on a yearly basis. 
        It calculates the number of smartphones to be replenished based on the initial number of used products, 
        randomly selects producer IDs from the model's available new product IDs, and creates new Smartphone instances 
        with randomized performance and time held values. The new smartphones are then added to the store's inventory.
        
        The replenishment process is designed to mimic the periodic restocking of used smartphones in a real-world scenario, 
        ensuring the store's inventory remains dynamic and reflects the natural flow of products in and out of the market.
        """
        num_replenished = int(random.uniform(0.1, 0.5) * self.init_num_used_products)
        producer_ids = list(self.model.new_product_id_price.keys())
        choose_ids = random.choices(producer_ids, k=num_replenished)
        
        for choose_id in choose_ids:
            product_price = self.model.new_product_id_price[choose_id]
            # if self.unique_id == 1026:
            #     print(f'time {self.model.steps}, {choose_id}, {product_price}')
            product = Smartphone(
                    model=self.model,
                    is_new=False,
                    producer_id=choose_id,
                    user_id=None,
                    performance=random.uniform(0.7, 0.9),
                    time_held=random.randint(1, 24),
                    product_price=product_price,
                    initial_repair_cost=500,)
            self.inventory.append(product)

    def buy_from_consumer(self, smartphone:Smartphone, consumer_id:int):
        """
        Purchase a used smartphone from a consumer and add it to store inventory.
        
        Args:
            smartphone (Smartphone): The smartphone being sold to the store by the consumer.
            
        This function repairs the smartphone, transfers ownership to the store by updating 
        the smartphone's user_id, and adds it to the store's inventory. The repair is done
        automatically before adding to inventory to ensure all store stock is in good condition.
        """
        # smartphone.repair_product() # repair the used smartphone for reselling
        # self.repair_cost += smartphone.calculate_repair_cost() # update the repairing cost
        # smartphone.user_id = self.unique_id # change the owner
        # smartphone.is_new = False
        # smartphone.calculate_sechnd_market_price() # decide the used product price
        # self.inventory.append(smartphone) # add to the inventory
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
        # for product in self.inventory:
        #     if product.unique_id == smartphone_id:
        #         self.inventory.remove(product)

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
        if (self.model.steps + self.replenish_interval) % 12 == 0:
            self.replenish()

        # print(f't={self.model.steps}, id={self.unique_id}, stocks={len(self.inventory)}, buy={self.num_buy_from}')
        for smartphone in self.inventory:
            smartphone.update_time_held(owner='sechandstore')
            # if self.unique_id == 1026:
            #     print(f'{self.model.steps}, {self.unique_id} doing update ' 
            #         f'{smartphone.producer_id}: {smartphone.purchase_price},{smartphone.secondhand_market_price}')
            if smartphone.performance <= 0.4:
                # random pick a recycler
                self.send_smartphone_to_recycler(smartphone)
                self.inventory.remove(smartphone)

        # update the every year average product price
        if len(self.inventory) != 0:
            product_price = [smartphone.secondhand_market_price for smartphone in self.inventory]
            self.avg_product_price = np.mean(product_price)
        # print(f"SecondHandStore {self.unique_id} doing {self.avg_product_price}.")