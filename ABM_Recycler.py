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
            material_weights=None,
            virgin_material_price=None,
            quality_factor=None,
            recycle_waste_rate=None):
        """
        Creation of new recycler agent
        """
        super().__init__(model)
        self.unique_id = unique_id
        self.recyclering_cost = 0
        self.cumulative_recycle_number = 0
        self.recycle_waste = 0
        self.recycled_stocks = []
        self.sechdstore_partner = []
        self.customers = []

        # Set default values if None
        self.material_weights = material_weights if material_weights is not None else \
            {'metals':0.45, 'glass':0.32, 'Plastics':0.17, 'Other':0.06}
        self.virgin_material_price = virgin_material_price if virgin_material_price is not None else \
            {'metals':1000, 'glass':500, 'Plastics':200, 'Other':350}
        self.quality_factor = quality_factor if quality_factor is not None else \
            {'metals':0.7, 'glass':0.5, 'Plastics':0.5, 'Other':0.5}
        self.recycle_waste_rate = recycle_waste_rate if recycle_waste_rate is not None else \
            {'metals':0.7, 'glass':0.5, 'Plastics':0.5, 'Other':0.5}
        
        self.recycled_material_price={}
        for material in self.virgin_material_price.keys():
            self.recycled_material_price[material] = \
                self.quality_factor[material] * self.virgin_material_price[material]

    def recycle_from_customer(self, smartphone: Smartphone, consumer_id:int):
        """
        Facilitates the recycling of a smartphone directly from a consumer, 
        updating the recycler's records and inventory.

        Args:
            smartphone (Smartphone): The smartphone being recycled, which is removed from the consumer's possession.
            consumer_id (int): The unique identifier of the consumer who is recycling the smartphone.

        Returns:
            float: The price paid to the consumer for recycling their smartphone.
        """
        recycle_price = smartphone.calculate_recycle_price()
        self.recyclering_cost += recycle_price
        self.customers.append(consumer_id)
        self.cumulative_recycle_number += 1
        self.recycled_stocks.append(smartphone)
        return recycle_price

    def recycle_from_secondhandstore(self, smartphone: Smartphone, sechdstore_id:int):
        """
        Facilitates the recycling of a smartphone from the second-hand market, 
        updating the recycler's records and inventory.

        Args:
            smartphone (Smartphone): The smartphone being recycled, 
            which is removed from the second-hand market.
            sechdstore_id (int): The unique identifier of the second-hand store 
            from which the smartphone is being recycled.

        This method updates the recycler's records by adding the smartphone to the 
        recycled stocks, incrementing the cumulative recycle number, and updating 
        the recycling cost. It also associates the second-hand store ID with the recycling process.
        """
        self.recyclering_cost += smartphone.calculate_recycle_price()
        self.sechdstore_partner.append(sechdstore_id)
        self.cumulative_recycle_number += 1
        self.recycled_stocks.append(smartphone)

    def update_recycle_waste(self):
        """
        Update the total waste generated from recycling processes.
        """
        for product in self.recycled_stocks:
            for material in self.material_weights.keys():
                self.recycle_waste += (1 - product.performance) \
                * self.recycle_waste_rate[material] \
                * self.material_weights[material]

        for product in self.recycled_stocks:
            product.remove()

    def step(self):
        """
        Evolution of agent at each step
        """
        # Update the recycle waste
        self.update_recycle_waste()
        # Clean up the recycled products every year
        self.recycled_stocks = []
        # print(f"Recycler {self.unique_id} recycle waste: {self.recycle_waste}")
        # print(f"Recycler {self.unique_id} doing.")
