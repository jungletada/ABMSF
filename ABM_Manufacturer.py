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
            material_weights=None,
            virgin_material_price=None,
            recycled_material_price=None,
            recycled_material_percentages=None,
            material_demand_limits=None):
        
        super().__init__(model)
        self.unique_id = unique_id
        self.material_weights = material_weights or {'metals':0.45, 'glass':0.32, 'Plastics':0.17, 'Other':0.06}
        self.virgin_material_price = virgin_material_price or {'metals':1000, 'glass':500, 'Plastics':200, 'Other':350}
        self.recycled_material_price = recycled_material_price or {'metals':1000, 'glass':500, 'Plastics':200, 'Other':350}
        self.recycled_percentages = recycled_material_percentages or {'metals':0.5, 'glass':0.1, 'Plastics':0.1, 'Other':0.3}
        self.demand_limits = material_demand_limits or {'metals':0.5, 'glass':0.1, 'Plastics':0.1, 'Other':0.3}
        self.stability_goal = 0.4

        # For pricing strategy
        self.product_price = 0
        self.profit_margin = 0.3
        self.demand_elasticity = 0.2
        self.financial_incentive = 0
        self.sigma_fi = 0.05
        
        self.cumulative_sales = 0
        self.income = 0
        self.production_cost = 0

    def calculate_production_cost(self):
        """
        Calculate the production cost based on the use of materials.
        制造商如何决定recycled_percentages?
        Returns:
            float: Total production cost considering the recycled materials and constraints.
        """
        production_cost = 0
        recycled_weight = 0
        # Calculate production cost and ensure constraints are met
        for material in self.material_weights.keys():
            # Calculate cost for this material
            recycled_material_cost = self.recycled_material_price[material] \
                * self.recycled_percentages[material] * self.material_weights[material]
            virgin_material_cost = self.virgin_material_price[material] \
                * (1 - self.recycled_percentages[material]) * self.material_weights[material]
            production_cost = recycled_material_cost + virgin_material_cost
            recycled_weight += self.recycled_percentages[material] * self.material_weights[material]
        
        self.financial_incentive = self.sigma_fi * recycled_weight
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
        self.product_price = adjusted_price - self.financial_incentive

        return self.product_price

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
            purchase_price=self.product_price,
            user_id=consumer_id)
        self.cumulative_sales += 1
        return smartphone

    def trade_with_recycler(self, recycler_agent):
        """
        """
        self.partner = recycler_agent

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
        # print(f"Manufacturer {self.unique_id}, production cost: {self.production_cost:.2f}")
        # self.set_product_price()
        # self.count_income()
        # print(f"Manufacturer {self.unique_id} doing.")
