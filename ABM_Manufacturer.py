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


class Manufacturer(Agent):
    """
    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_Model)
        second_market_prices.
        virgin_market_prices.
    """

    def __init__(self, unique_id, model, recycled_market_prices, virgin_market_prices,
                 social_influencability_boundaries, self_confidence_boundaries):
        """
        Creation of new producer agent
        """
        super().__init__(unique_id, model)
        self.material_percentage = {
            'Aluminum':0.06, 'Copper':0.06, 'Iron':0.148, 'Glass':0.32, 'Plastics':0.17, 'Others':0.06
        }
        self.trust_history = np.copy(self.model.trust_prod)
        self.social_influencability = np.random.uniform(
            social_influencability_boundaries[0], social_influencability_boundaries[1])
        self.agent_i = self.unique_id - self.model.num_consumers
        self.knowledge = np.random.random()
        self.social_interactions = np.random.random()
        self.knowledge_learning = np.random.random()
        self.knowledge_t = self.knowledge
        
        self.acceptance = 0
        self.symbiosis = False
        self.self_confidence = np.random.uniform(
            self_confidence_boundaries[0], self_confidence_boundaries[1])
        self.material_produced = self.producer_type()
        
        self.recycled_material_volume = 0
        self.yearly_recycled_material_volume = 0
        self.recycling_volume = 0
        self.recycled_mat_price = np.random.triangular(
            recycled_market_prices[self.material_produced][0], recycled_market_prices[
                self.material_produced][2], recycled_market_prices[
                self.material_produced][1])
        self.virgin_market_prices = np.random.triangular(
            virgin_market_prices[self.material_produced][0], virgin_market_prices[
                self.material_produced][2], virgin_market_prices[
                self.material_produced][1])
        self.all_virgin_mat_prices = virgin_market_prices
        self.recycled_material_value = 0
        
        self.producer_costs = 0
        self.avoided_costs_virgin_materials = 0
    
    def calculate_production_cost(self):
        """
        Calculate production cost of virgin materials and recycled materials for one product.
        """
        pass
    
    def calculate_usage_materials(self):
        """
        Use constraints for calculate the usage of virgin materials and recycled materials in percentage.
        """
        usuage = {
            
        }
        return usuage
    
    def set_product_price(self):
        """
        Set the price of the product.
        """
        pass
    
    def count_income(self):
        """
        Count the income of the producer.
        """
        pass

    def step(self):
        """
        Evolution of agent at each step
        """
        pass
