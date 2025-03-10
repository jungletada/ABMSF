import math
import random
import numpy as np
import networkx as nx
from mesa import Agent

from ABM_Smartphone import Smartphone
from ABM_Recycler import Recycler


class Manufacturer(Agent):
    """
    A manufacturer agent that produces smartphones using virgin and recycled materials.
    
    The manufacturer makes decisions about material sourcing, production costs, and pricing.
    It tracks metrics like cumulative sales, income, and production costs. The manufacturer
    can adjust recycled material percentages within demand limits and aims for stability in
    material sourcing.

    Attributes:
        material_weights (dict): Weight percentages of different materials in a smartphone
        virgin_material_price (dict): Prices of virgin materials by material type
        recycled_material_price (dict): Prices of recycled materials by material type  
        recycled_material_percentages (dict): Target percentages of recycled content by material
        material_demand_limits (dict): Maximum allowable recycled content by material
        stability_goal (float): Target stability level for material sourcing (0-1)
        product_price (float): Current selling price of smartphones
        profit_margin (float): Target profit margin percentage
        demand_elasticity (float): Price sensitivity of demand
        financial_incentive (float): Government incentive for using recycled materials
        cumulative_sales (int): Total number of smartphones sold
        income (float): Total revenue from sales
        production_cost (float): Current cost to produce one smartphone
    """
    def __init__(
            self,
            model,
            unique_id,
            init_product_price=2999,
            material_weights=None,
            virgin_material_price=None,
            recycled_material_price=None,
            material_demand_limits=None):

        super().__init__(model)
        self.unique_id = unique_id
        self.product_weight = 200 # in gram

        self.material_weights = material_weights or \
            {'metals':0.45, 'glass':0.32, 'Plastics':0.17, 'Other':0.06}
        self.virgin_material_price = virgin_material_price or \
            {'metals':0.21, 'glass':500, 'Plastics':200, 'Other':350}
        self.recycled_material_price = recycled_material_price or \
            {'metals':1000, 'glass':500, 'Plastics':200, 'Other':350}

        # Upper bound
        self.demand_limits = material_demand_limits or \
            {'metals':0.25, 'glass':0.1, 'Plastics':0.1, 'Other':0.3}

        self.recycled_percentages = {}

        # Initialized as the sustainability
        for k, v in self.demand_limits.items():
            self.recycled_percentages[k] = v

        # For pricing strategy
        self.product_price = init_product_price
        self.features2price =  random.uniform(0.5, 0.8)
        self.profit_margin = random.uniform(0.1, 0.3)
        self.demand_elasticity = random.uniform(0., 0.02)
        self.cost2price_ratio = random.uniform(0.20, 0.40)
        self.price_update_month = random.randint(0, 11)
        self.financial_incentive = 0
        self.sigma_fi = 0.05

        self.customers = []
        self.rec_customers = []
        self.cumulative_sales = 0
        self.income = 0
        self.recycle_sum = 0
        self.production_cost = 0
        self.cumulative_recycle_number = 0

    def update_product_price(self):
        """
        Adjusts the price of the smartphone annually based on demand elasticity and production cost.
        
        This method updates the product price every 12 steps (representing a year) by applying a price 
        increase based on the demand elasticity, which simulates a 2-5% annual increase. 
        Additionally, it incorporates a production cost adjustment to ensure profitability, considering 
        the profit margin. The final price is rounded up to the nearest integer.

        Args:
            step (int): The current step in the simulation, representing months.
            
        Returns:
            float: The updated price of the smartphone.
        """
        # Only update price annually (every 12 steps)
        if (self.model.steps + self.price_update_month) % 12 == 0:# Calculate price increase (2-5% annually)
            self.product_price *= (1 + self.demand_elasticity) # Update product price with increase
            self.product_price = math.ceil(self.product_price) # integer price
            # print(f'product_price: {self.product_price}')

    def trade_with_consumer(self, consumer_id):
        """
        Sell a product to a consumer.

        This method creates a new Smartphone instance with the current product price
        and assigns it to the specified consumer. It also increments the cumulative
        sales count for the manufacturer.

        Args:
            consumer_id (int): The unique identifier of the consumer purchasing the smartphone.

        Returns:
            Smartphone: A new Smartphone instance created for the consumer.
        """

        smartphone = Smartphone(
            is_new=True,
            model=self.model,
            performance=1,
            time_held=0,
            product_price=self.product_price,
            producer_id=self.unique_id,
            user_id=consumer_id)
        self.cumulative_sales += 1
        self.customers.append(consumer_id)
        return smartphone

    def recycle_from_customer(self, smartphone: Smartphone, consumer_id:int):
        """
        Recycle a smartphone directly from a consumer for trade-in service.

        Args:
            smartphone (Smartphone): The smartphone to be recycled
            consumer_id (int): The ID of the consumer recycling the phone

        Returns:
            float: The recycling price for the smartphone
        """
        self.cumulative_recycle_number += 1
        tiv_price = smartphone.calculate_trade_in_value()
        self.recycle_sum += tiv_price
        self.rec_customers.append(consumer_id)
    
    def count_income(self):
        """
        Count the income of the producer.
        """
        # Add production cost adjustment
        self.production_cost = self.cost2price_ratio * self.product_price
        self.income = (self.product_price - self.production_cost) \
            * self.cumulative_sales

        return self.income

    def step(self):
        """
        Evolution of agent at each step
        """
        self.update_product_price()
        self.count_income()
        # print(f'Producer: {self.unique_id}, {self.product_price}')
        # print(f"Manufacturer {self.unique_id}, production cost: {self.production_cost:.2f}")
        # print(f"Manufacturer {self.unique_id} doing.")
