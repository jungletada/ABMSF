import numpy as np
from mesa import Agent
from ABM_Smartphone import Smartphone


class Recycler(Agent):
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
            model,
            unique_id,
            material_weights={'metals':0.45, 'glass':0.32, 'Plastics':0.17, 'Other':0.06},
            virgin_material_price={'metals':1000, 'glass':500, 'Plastics':200, 'Other':350},
            quality_factor={'metals':0.7, 'glass':0.5, 'Plastics':0.5, 'Other':0.5},
            recycle_waste_rate={'metals':0.7, 'glass':0.5, 'Plastics':0.5, 'Other':0.5},):
        """
        Creation of new recycler agent
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.recyclering_cost = 0
        self.recycle_number_now = 0
        self.cumulative_recycle_number = 0
        self.recycle_waste = 0
        self.sechdstore_partner = []
        self.customers = []
        self.material_weights = material_weights
        self.virgin_material_price = virgin_material_price
        self.quality_factor = quality_factor
        self.recycle_waste_rate = recycle_waste_rate
        
        self.recycled_material_price={}
        for material in self.virgin_material_price.keys():
            self.recycled_material_price[material] = \
                self.quality_factor[material] * self.virgin_material_price[material]

    def trade_with_consumer_recycle(self, smartphone: Smartphone, consumer_id:int):
        """

        """
        recycle_price = smartphone.calculate_recycle_price()
        self.recycle_number_now += 1
        self.customers.append(consumer_id)
        return recycle_price
   
    def recycle_from_secondhand(self, smartphone: Smartphone, sechdstore_id:int):
        """
        Recycle a smartphone from the second-hand market.
        """
        # Recycle the smartphone and update the recycle number
        self.recycle_number_now += 1
        # Update the recycle cost
        self.recyclering_cost += smartphone.calculate_recycle_price()
        self.sechdstore_partner.append(sechdstore_id)
        # Update the recycle waste
        self.update_recycle_waste(mce=0.5)
   
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
        self.update_recycle_waste(mce=0.5)
        # print(f"Recycler {self.unique_id} recycle waste: {self.recycle_waste}")
        # print(f"Recycler {self.unique_id} doing.")
