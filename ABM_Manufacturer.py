# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent - Producer
"""

from mesa import Agent
import numpy as np
import networkx as nx
import random
from ABM_Smartphone import Smartphone


class Manufacturer(Agent):
    """
    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_Model)
        second_market_prices.
        virgin_market_prices.
    """

    def __init__(self,
                 unique_id,
                 model,
                 material_weights={'metals':0.45, 'glass':0.32, 'Plastics':0.17, 'Other':0.06},
                 virgin_material_price={'metals':1000, 'glass':500, 'Plastics':200, 'Other':350},
                 recycled_material_price={'metals':1000, 'glass':500, 'Plastics':200, 'Other':350},
                 recycled_percentages={'metals':0.5, 'glass':0.1, 'Plastics':0.1, 'Other':0.3},
                 demand_limits={'metals':0.5, 'glass':0.1, 'Plastics':0.1, 'Other':0.3},
                 ):
        """
        Creation of new producer agent
        """
        super().__init__(unique_id, model)

        self.material_weights = material_weights
        self.virgin_material_price = virgin_material_price
        self.recycled_material_price = recycled_material_price
        self.recycled_percentages = recycled_percentages
        self.demand_limits = demand_limits
        self.stability_goal = 0.4

        # For pricing strategy
        self.product_price = None
        self.profit_margin = None
        self.demand_elasticity = None
        self.financial_incentive = None

        self.cumulative_sales = 0
        self.income = 0

    def calculate_production_cost(self): # TBD
        """
        Calculate the production cost based on the use of recycled materials.

        Returns:
            float: Total production cost considering the recycled materials and constraints.
        """
        production_cost = 0
        total_recycled_weight = 0
        total_weight = sum(self.material_weights.values())
        # Calculate production cost and ensure constraints are met
        for material in self.material_prices.keys():
            # Ensure the recycled percentage does not exceed the demand limit
            recycled_percentage = min(self.recycled_percentages[material], self.demand_limits[material])
            # Calculate cost for this material
            recycled_material_cost = self.recycled_material_price * recycled_percentage * self.material_weights[material]
            virgin_material_cost = self.virgin_material_price * (1 - recycled_percentage) * self.material_weights[material]
            production_cost = recycled_material_cost + virgin_material_cost
            # Accumulate the recycled weight for stability/regulatory goals
            total_recycled_weight += recycled_percentage * self.material_weights[material]

        # Check the stability goal constraint
        if total_recycled_weight < self.stability_goal * total_weight:
            raise ValueError("Stability goal for recycled materials not met.")
        self.production_cost = production_cost
        return self.production_cost
    
    def set_product_price(self):
        """
        Calculate the price of the new smartphone based on the key factors.

        Returns:
            float: The price of the new smartphone at time t.
        """
        # Base price before considering demand elasticity and financial incentives
        base_price = self.production_cost * (1 + self.profit_margin)
        # Adjust price for demand elasticity
        adjusted_price = base_price * (1 + self.demand_elasticity)
        # Apply financial incentive for recycling (reduce the price)
        final_price = adjusted_price - self.financial_incentive
        self.product_price = max(0, final_price)
        return self.product_price

    def purchase_from_consumer(self, consumer_id):
        """
        Purchase a new smartphone from the manufacturer for a consumer.

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
            purchase_price=self.product_price,
            user_id=consumer_id)
        self.cumulative_sales += 1
        return smartphone

    def count_income(self):
        """
        Count the income of the producer.
        """
        self.income = self.product_price * self.cumulative_sales
        return self.income

    def step(self):
        """
        Evolution of agent at each step
        """
        self.calculate_production_cost()
        self.set_product_price()
        self.count_income()
