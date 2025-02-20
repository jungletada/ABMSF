import math
import random
import numpy as np
from mesa import Agent


class Smartphone(Agent):
    """Represents a smartphone in the agent-based model simulation."""
    def __init__(
            self,
            model,
            is_new,
            producer_id=None,
            user_id=None,
            performance=1.0,
            time_held=0,
            product_price=5000,
            initial_repair_cost=500.0):
        """
        Initialize a Smartphone instance.

        Attributes:
            is_new (bool): Whether the phone is new (True) or used (False).
            performance (float): Performance level of the phone (0 to 1, where 1 is best performance).
            time_held (int): The time (e.g., months or years) the user has held the phone.
            status (int): If the phone is fine (1) or broken (0).
            purchase_price (float): The price at which the user bought the new phone.
            repair_cost (float): The potential cost to repair the phone if it breaks.
            store_initial_cost (float): Initial repair cost for second-hand store or recycler.
            store_handled_phones (int): Initial number of phones handled by the store or recycler.
            decay_rate (float): The rate at which the performance degrades over time (exponential decay).
            user_id (int/None): ID of the current user who holds the phone (could be a consumer or second-hand store).
        """
        super().__init__(model)
        self.is_new = is_new            # True：new smartphone; False: used smartphone
        self.performance = performance  # Range from 0 to 1, where 1 is perfect condition
        self.time_held = time_held      # Number of time units (e.g., months or years) held by the user
        self.producer_id = producer_id  # Producer id
        self.user_id = user_id          # ID of the current user holding this phone

        self.purchase_price = product_price  # Purchase price from the new market
        self.repair_cost = 0    # Repair cost if the phone is broken
        self.resell_price = 0
        self.secondhand_market_price = 0
        self.sec_features2price = random.uniform(0.5, 0.8) # only valid if used
        
        self.initial_repair_cost = initial_repair_cost # initial repair cost for second-hand store and recycler.

        self.decay_rate = 0.0005 # Rate at which the performance degrades (lambda in the exponential decay model)
        self.noise_decay = 0.0005 # Gaussian noise for performance degradation
        self.demand_used = random.uniform(-0.2, 0.2) # demand of use for each smartphone
        self.discount_rate = random.uniform(0.9, 1.0) # demand of use for each smartphone
        self.material_value = 500 # material_value depends on the ingredients of product
        self.resell_value = self.calculate_resell_price_sechnd()  # Value if resold in the second-hand market
        self.warranty_duration = 6 if self.is_new else 0  # New phones come with 12 months warranty

        # Recycle Service
        self.perf_rec = 0.1
        self.time_rec = 0.2
        self.price_rec = 0.7
        self.discount_rec = np.random.normal(0.65, 0.02)
        self.recycle_price = self.discount_rec * self.purchase_price

        # Trade-in-Service
        self.perf_tiv = 0.1
        self.time_tiv = 0.1
        self.price_tiv = 0.8
        self.discount_tiv = np.random.normal(0.5, 0.02)
        self.trade_in_value = self.discount_tiv * self.purchase_price

    def degrade_performance(self, decay_rate, noise_decay):
        """
        Simulate the degradation of performance over time using an exponential decay model with Gaussian noise.
        """
        # Calculate the exponential decay
        decay_factor = math.exp(decay_rate * self.time_held)
        self.performance = max(0, self.performance * decay_factor)
        # Add Gaussian noise to the performance
        noise = random.gauss(0, noise_decay)  # Mean of 0, standard deviation of 0.02
        self.performance = max(0, min(1, self.performance + noise))  # Ensure performance stays between 0 and 1
        # Update the status if the phone reaches a critical performance level or by random chance
        return self.performance
    
    def update_time_held(self, owner='consumer'):
        """Increment the time held by the user."""
        self.time_held += 1
        # update performance
        if owner == 'consumer':
            self.degrade_performance(decay_rate=self.decay_rate, noise_decay=self.noise_decay)
        else:
            self.degrade_performance(decay_rate=0.5 * self.decay_rate, noise_decay=self.noise_decay)
        # update cost price for eol
        self.calculate_repair_cost()
        self.calculate_recycle_price()
        self.calculate_resell_price_sechnd()
        self.calculate_sechnd_market_price()

    def calculate_repair_cost(self):
        """Attempt to repair the phone, considering warranty status and learning effects.

        If the phone is under warranty (self.time_held < self.warranty_duration), 
        the repair cost is set to a very low value. Otherwise, the repair cost is calculated
        based on phone's performance and the learning effect of the store.
        store_initial_cost (float): Initial repair cost for second-hand store or recycler.
        store_handled_phones (int): current number of phones handled by the store or recycler.
        """
        if self.time_held < self.warranty_duration:  # Phone is under warranty
            self.repair_cost = 0  # Free repair under warranty (or set to a very low cost)
            # print(f"Phone is under warranty. Repair cost: {self.repair_cost}")
        
        else:  # Phone is out of warranty, calculate repair cost based on performance and learning effect
            max_cost = 0.1 * self.purchase_price  # Set the maximum repair cost (20% of purchase price)
            self.repair_cost = max_cost * (1 - self.performance) # Basic repair cost based on performance
        return self.repair_cost

    def repair_product(self):
        """
        Attempt to repair the phone with a random chance of success.
        """
        self.performance = min(0.95, self.performance + 0.3)  # Performance improves after repair

    def calculate_resell_price_sechnd(self):
        """
        Calculate the buying price for a used smartphone from a consumer.
        
        Parameters:
            phone_performance (float): The performance of the used smartphone (Perf_{sid}^t), a value between 0 and 1.
            repair_cost (float): The calculated repair cost (C_{repair}) for the used phone.
            
        Returns:
            float: The price the second-hand store is willing to pay for the used smartphone.
        """
        # Normalized repair cost (repair cost relative to new phone price)
        normalized_repair_cost = self.repair_cost / self.purchase_price
        # Calculate the buying price considering performance, repair cost, and demand
        buying_price = (self.purchase_price * self.performance) * (1 - normalized_repair_cost)
        # Adjust the buying price based on the demand for used phones
        self.resell_price = buying_price * (1 + self.demand_used)
        return self.resell_price

    def calculate_sechnd_market_price(self):
        """
        Calculate the selling price for a used smartphone for used product market.

        Parameters:
            phone_performance (float): The performance of the used smartphone (Perf_{sid}^t), 
                a value between 0 and 1.
            repair_cost (float): The calculated repair cost (C_{repair}) for the used phone.
            
        Returns:
            float: The price the second-hand store is willing to pay for the used smartphone.
        """
        # Normalized repair cost (repair cost relative to new phone price)
        self.calculate_repair_cost()
        # print(f'Smartphone repair {self.repair_cost}')
        normalized_repair_cost = self.repair_cost / self.purchase_price
        # Calculate the buying price considering performance, repair cost, and demand for used phones
        selling_price = self.purchase_price * self.performance * \
            (1 + normalized_repair_cost) * (1 + self.demand_used) * self.discount_rate
        self.secondhand_market_price = int(min(0.9 * self.purchase_price, selling_price))
        # print(f'new {self.purchase_price}, used {self.secondhand_market_price}')
        return self.secondhand_market_price
    
    def calculate_trade_in_value(self):
        """Model the Manufacturer Recycling for Trade-in (Old-for-New Service)"""
        self.trade_in_value = self.discount_tiv * \
            (self.perf_tiv * self.performance + \
                self.time_tiv / (1 + self.time_held) + \
                    self.price_tiv * self.purchase_price)
        return self.trade_in_value
    
    def calculate_recycle_price(self):
        """
        Calculate the recycled price for a used smartphone based 
            on recoverable material value.
            
        Returns:
            float: The recycled price for the used smartphone.
        """
        # Recycled price based on material value and scaling factor alpha
        self.recycle_price = self.discount_rec * \
            (self.perf_rec * self.performance + \
                self.time_rec / (1 + self.time_held) + \
                    self.price_rec * self.purchase_price)
        return self.recycle_price

    def recycle_product(self, new_owner_id):
        """
        Process the recycling of the smartphone and update its owner.
        """
        self.user_id = new_owner_id  # Update the user ID to the new owner

    def __repr__(self):
        return (
            f"Smartphone(is_new={self.is_new}, "
            f"performance={self.performance:.2f}, " 
            f"time_held={self.time_held}, "
            f"purchase_price={self.purchase_price}, "
            f"user_id={self.user_id}, "
            f"resell_value={self.resell_value:.2f})"
            )
