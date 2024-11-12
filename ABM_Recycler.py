from mesa import Agent
import numpy as np


class Recyclers(Agent):
    """
    A recycler which sells recycled materials and improve its processes.

    Attributes:
        unique_id: agent #, also relate to the node # in the network
        model (see ABM_Model)
        original_recycling_cost (a list for a triangular distribution) ($/fu) (
            default=[0.106, 0.128, 0.117]). From EPRI 2018.
        init_eol_rate (dictionary with initial end-of-life (EOL) ratios),
            (default={"repair": 0.005, "sell": 0.02, "recycle": 0.1,
            "landfill": 0.4375, "hoard": 0.4375}). From Monteiro Lunardi
            et al 2018 and European Commission (2015).
        recycling_learning_shape_factor, (default=-0.39). From Qiu & Suh 2019.
        social_influencability_boundaries (from Ghali et al. 2017)
    """

    def __init__(
            self,
            unique_id,
            model,
            material_weights={'metals':0.45, 'glass':0.32, 'Plastics':0.17, 'Other':0.06},
            virgin_material_price={'metals':1000, 'glass':500, 'Plastics':200, 'Other':350},
            quality_factor={'metals':0.7, 'glass':0.5, 'Plastics':0.5, 'Other':0.5},
            recycle_waste_rate={'metals':0.7, 'glass':0.5, 'Plastics':0.5, 'Other':0.5},):
        """
        Creation of new recycler agent
        """
        super().__init__(unique_id, model)
        self.recyclering_cost = 0
        self.recycle_number_now = 0
        self.cumulative_recycle_number = 0
        self.recycle_waste = 0

        self.material_weights = material_weights
        self.virgin_material_price = virgin_material_price
        self.quality_factor = quality_factor
        self.recycle_waste_rate = recycle_waste_rate
        
        self.recycled_material_price={}
        for material in self.virgin_material_price.keys():
            self.recycled_material_price[material] = \
                self.quality_factor[material] * self.virgin_material_price[material]

    def recycle_product_from_consumer(self, smartphone):
        """

        """
        recycle_price = smartphone.calculate_recycle_price()
        self.recycle_number_now += 1
        return recycle_price
   
    def update_recycle_waste(self, mce):
        """
        
        """
        for material in self.virgin_material_price.keys():
            self.recycle_waste += self.recycle_waste_rate[material] \
                * mce * self.material_weights[material]

    def step(self):
        """
        Evolution of agent at each step
        """
        pass
