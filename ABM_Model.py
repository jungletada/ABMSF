import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector


from ABM_Consumer import Consumer
from ABM_Recycler import Recycler
from ABM_SecHandStore import SecondHandStore
from ABM_Manufacturer import Manufacturer


class AgentBasedModel(Model):
    def __init__(
            self,
            seed=None,
            calibration_n_sensitivity=1,
            calibration_n_sensitivity_2=1,
            consumers_node_degree=10,
            consumers_network_type="small-world",
            rewiring_prob=0.1,
            num_consumers=1000,
            num_recyclers=15,
            num_producers=8,
            num_sechdstores=20,
            prod_n_recyc_node_degree=5,
            prod_n_recyc_network_type="small-world",
            product_growth=[0.166, 0.045],
            growth_threshold=10,
            att_distrib_param_eol=[0.544, 0.1],
            att_distrib_param_reuse=[0.223, 0.262],
            repairability=0.55,
        ):
        # Set up variables
        super().__init__(seed=seed)
        self.seed = seed
        att_distrib_param_eol[0] = calibration_n_sensitivity
        att_distrib_param_reuse[0] = calibration_n_sensitivity_2
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Number of Agents
        self.num_consumers = num_consumers
        self.num_recyclers = num_recyclers
        self.num_producers = num_producers
        self.num_prod_n_recyc = num_recyclers + num_producers
        self.num_sechdstores = num_sechdstores
        
        # Network for consumers
        self.consumers_node_degree = consumers_node_degree
        self.consumers_network_type = consumers_network_type
        self.prod_n_recyc_node_degree = prod_n_recyc_node_degree
        self.prod_n_recyc_network_type = prod_n_recyc_network_type
        
        # Manufacturer
        self.init_product_prices = np.linspace(1999, 6999, num_producers)
        self.avg_new_product_price = 0
        self.avg_used_product_price = 0
        # Second hand store
        self.init_num_used_stocks_range = range(20, 50)
        # System
        self.running = True
        self.color_map = []
        
        self.all_comsumer_income = []
        self.avg_comsumer_income = 0.
        self.repairability = repairability
        
        self.total_waste = 0
        self.total_yearly_new_products = 0
        self.sold_repaired_waste = 0
        self.past_sold_repaired_waste = 0
        self.repairable_volume_recyclers = 0
        self.consumer_used_product = 0
        self.recycler_repairable_waste = 0
        self.yearly_repaired_waste = 0

        self.list_consumer_id = list(range(num_consumers))
        random.shuffle(self.list_consumer_id)
        
        self.new_product_id_price = {}
        self.product_growth = product_growth
        self.growth_threshold = growth_threshold
        
        # ============ Building graphs ============ #
        # Consumer's network
        self.consumer_network = self.init_network(
            network=self.consumers_network_type,
            nodes=self.num_consumers,
            node_degree=self.consumers_node_degree,
            rewiring_prob=rewiring_prob)

        # Manufacturer and Recycler's network
        self.recycler_network = self.init_network(
            network=self.prod_n_recyc_network_type,
            nodes=self.num_prod_n_recyc,
            node_degree=self.prod_n_recyc_node_degree,
            rewiring_prob=rewiring_prob)

        # Second-hand store's network
        self.sechdstore_network = self.init_network(
            network="complete graph",
            nodes=self.num_sechdstores,
            node_degree="NaN",
            rewiring_prob=rewiring_prob)

        #self.network = nx.disjoint_union(self.consumer_network, self.recycler_network)
        self.network = nx.disjoint_union(
            nx.disjoint_union(self.consumer_network, self.recycler_network),
            self.sechdstore_network)
        self.grid = NetworkGrid(self.network)
        ####################################################
        # Create agents, G nodes labels are equal to agents' unique_ID
        for node in self.network.nodes():
            #===================== Consumers =====================#
            if node < self.num_consumers:
                producer_ids = range(
                    self.num_recyclers + self.num_consumers,
                    self.num_prod_n_recyc + self.num_consumers)
                consumer = Consumer(
                    model=self,     # ABM_Model
                    unique_id=node, # agent ID ~ node ID in the network
                    preference_id=random.choice(producer_ids)
                    )
                self.grid.place_agent(consumer, node)  # Add the agent to the node
                
            elif node < self.num_recyclers + self.num_consumers:
                #===================== Recyclers =====================#
                recycler = Recycler(
                    model=self,
                    unique_id=node,
                    )
                self.grid.place_agent(recycler, node)

            elif node < self.num_prod_n_recyc + self.num_consumers:
                #===================== Producers =====================#
                producer_index = node - self.num_recyclers - self.num_consumers
                product_id_price = int(self.init_product_prices[producer_index])
                manufacturer = Manufacturer(
                    model=self,
                    unique_id=node,
                    init_product_price=product_id_price
                    )
                self.grid.place_agent(manufacturer, node)
                self.new_product_id_price[node] = product_id_price
            else:
                #================== Second-hand Store ==================#
                sechdstore = SecondHandStore(
                    model=self,
                    unique_id=node,
                    init_num_used_products=random.choice(self.init_num_used_stocks_range),
                    )
                self.grid.place_agent(sechdstore, node)

        # Defines reporters and setup data collector
        model_reporters = {
            "avg_consumer_income": lambda c:self.avg_comsumer_income,
            "consumer_proffer": lambda c: self.count_consumers_pathway("proffer"),
            "consumer_reselling": lambda c: self.count_consumers_pathway("reselling"),
            "consumer_recycling": lambda c: self.count_consumers_pathway("recycling"),
            "consumer_landfilling": lambda c: self.count_consumers_pathway("landfilling"),
            "consumer_storing": lambda c: self.count_consumers_pathway("hoarding"),
            "consumer_buying_new": lambda c: self.count_consumers_pathway("buy_new"),
            "consumer_buying_used": lambda c: self.count_consumers_pathway("buy_used"),
            "avg_new_product_price": lambda c: self.avg_new_product_price,
            "avg_used_product_price": lambda c: self.avg_used_product_price,
            'new_price_to_income': lambda c: self.report_output('new_price_to_income'),
            'used_price_to_income':lambda c: self.report_output('used_price_to_income'),
            }

        agent_reporters = {
            # "Number_product_repaired":
            #     lambda a: getattr(a, "number_product_repaired", None),
            # "Number_product_sold":
            #     lambda a: getattr(a, "number_product_sold", None),
            # "Number_product_recycled":
            #     lambda a: getattr(a, "number_product_recycled", None),
            # "Number_product_landfilled":
            #     lambda a: getattr(a, "number_product_landfilled", None),
            # "Number_product_hoarded":
            #     lambda a: getattr(a, "number_product_hoarded", None),
            # "Recycling":
            #     lambda a: getattr(a, "EoL_pathway", None),
            # "Landfilling costs":
            #     lambda a: getattr(a, "landfill_cost", None),
            # "Storing costs":
            #     lambda a: getattr(a, "hoarding_cost", None),
            # "Recycling costs":
            #     lambda a: getattr(a, "recycling_cost", None),
            # "Repairing costs":
            #     lambda a: getattr(a, "repairing_cost", None),
            # "Selling costs":
            #     lambda a: getattr(a, "scd_hand_price", None),
            # "Material produced":
            #     lambda a: getattr(a, "material_produced", None),
            # "Recycled volume":
            #     lambda a: getattr(a, "recycled_material_volume", None),
            # "Recycled value":
            #     lambda a: getattr(a, "recycled_material_value", None),
            # "Producer costs":
            #     lambda a: getattr(a, "producer_costs", None),
            # "Consumer costs":
            #     lambda a: getattr(a, "consumer_costs", None),
            # "Recycler costs":
            #     lambda a: getattr(a, "recycler_costs", None),
            # "Refurbisher costs":
            #     lambda a: getattr(a, "refurbisher_costs", None)
            }

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters)
        
    def shortest_paths(self, target_states, distances_to_target):
        """
        Compute shortest paths between chosen origin states and targets with
        the Dijkstra algorithm.
        """
        for i in self.states_graph.nodes:
            shortest_paths = []
            for j in target_states:
                shortest_paths.append(
                    nx.shortest_path_length(
                        self.states_graph, 
                        source=i,
                        target=j, 
                        weight='weight',
                        method='dijkstra'))
            shortest_paths_closest_target = min(shortest_paths)
            if shortest_paths_closest_target == 0:
                shortest_paths_closest_target = self.mean_distance_within_state
            distances_to_target.append(shortest_paths_closest_target)
        return distances_to_target

    def init_network(self, network, nodes, node_degree, rewiring_prob):
        """
        Set up model's industrial symbiosis (IS) and consumers networks.
        """
        if network == "small-world":
            return nx.watts_strogatz_graph(
                nodes, node_degree, rewiring_prob, seed=random.seed(self.seed))
        elif network == "complete graph":
            return nx.complete_graph(nodes)
        if network == "random":
            return nx.watts_strogatz_graph(nodes, node_degree, 1)
        elif network == "cycle graph":
            return nx.cycle_graph(nodes)
        elif network == "scale-free graph":
            return nx.powerlaw_cluster_graph(nodes, node_degree, 0.1)
        else:
            return nx.watts_strogatz_graph(nodes, node_degree, rewiring_prob)

    # def waste_generation(self, avg_lifetime, failure_rate, num_product):
    #     """
    #     Generate waste, called by consumers and recyclers/refurbishers
    #     (to get original recycling/repairing amounts).
    #     """
    #     correction_year = len(self.total_number_product) - 1
    #     return [j * (1 - math.e ** (-(((self.step + (correction_year - z)) /
    #                            avg_lifetime[z]) ** failure_rate))).real
    #             for (z, j) in enumerate(num_product)]

    def count_average_product_price(self):
        # used products
        num_stocks = 0
        total_used_price = 0

        # for sechndstore in self.agents_by_type[SecondHandStore]:
        #     if len(sechndstore.inventory) != 0:
        #         for smartphone in sechndstore.inventory:
        #             num_stocks += 1
        #             total_used_price += smartphone.secondhand_market_price
        # if num_stocks != 0:
        #     avg_used_product_price = int(total_used_price / num_stocks)
        # else:
        #     avg_used_product_price = None

        for agent in self.agents:
            if isinstance(agent, SecondHandStore):
                if len(agent.inventory) != 0:
                    for smartphone in agent.inventory:
                        num_stocks += 1
                        total_used_price += smartphone.calculate_sechnd_market_price()
        if num_stocks != 0:
            avg_used_product_price = int(total_used_price / num_stocks)
        else:
            avg_used_product_price = None
        self.avg_used_product_price = avg_used_product_price

        # new products
        total_new_price = 0
        for agent in self.agents:
            if isinstance(agent, Manufacturer):
                total_new_price += agent.product_price
        avg_new_product_price = total_new_price / self.num_producers
        self.avg_new_product_price = avg_new_product_price

    def count_consumers_pathway(self, condition):
        """
        Count adoption in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        for agent in self.agents_by_type[Consumer]:
            if condition == "proffer":
                count += agent.to_proffer
            elif condition == "reselling":
                count += agent.to_resell
            elif condition == "recycling":
                count += agent.to_recycle
            elif condition == "landfilling":
                count += agent.to_landfill
            elif condition == "hoarding":
                count += agent.to_store
            elif condition == "buy_new":
                count += agent.to_buy_new
            elif condition == "buy_used":
                count += agent.to_buy_used
            else:
                continue
        return count

    def update_statistics(self):
        """
        Update model statistics including consumer incomes and manufacturer prices.
        Collects current income data from consumers and updates product prices from manufacturers.
        """
        self.all_comsumer_income = [agent.income for agent in self.agents_by_type[Consumer]]
        self.avg_comsumer_income = float(np.mean(self.all_comsumer_income))
        self.count_average_product_price()
        manufacturers = list(self.agents_by_type[Manufacturer])
        for m in manufacturers:
            self.new_product_id_price[m.unique_id] = m.product_price

    def report_output(self, condition):
        """
        Count waste streams in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        count2 = 0
        industrial_waste_landfill = 0
        industrial_waste_recycled = 0
        industrial_waste_landfill_mass = 0
        industrial_waste_recycled_mass = 0

        if condition == 'new_price_to_income':
            count = self.avg_new_product_price / self.avg_comsumer_income
        elif condition == 'used_price_to_income':
            count = self.avg_used_product_price / self.avg_comsumer_income
        # for agent in self.agents:
        #     if self.num_consumers + self.num_recyclers <= agent.unique_id < \
        #             self.num_consumers + self.num_prod_n_recyc:
        #         if self.epr_business_model:
        #             industrial_waste_recycled += \
        #                 agent.industrial_waste_generated / self.num_consumers
        #             industrial_waste_recycled_mass += \
        #                 self.yearly_product_wght * \
        #                 agent.industrial_waste_generated / self.num_consumers
        #         else:
        #             industrial_waste_landfill += \
        #                 agent.industrial_waste_generated / self.num_consumers
        #             industrial_waste_landfill_mass += \
        #                 self.yearly_product_wght * \
        #                 agent.industrial_waste_generated / self.num_consumers
        #     elif condition == "product_stock" and agent.unique_id < \
        #             self.num_consumers:
        #         count += sum(agent.number_product_hard_copy)
        #     elif condition == "product_stock_new" and agent.unique_id < \
        #             self.num_consumers:
        #         count += sum(agent.new_products_hard_copy)
        #     if condition == "product_stock_used" and agent.unique_id < \
        #             self.num_consumers:
        #         count += sum(agent.used_products_hard_copy)
        #     elif condition == "prod_stock_new_mass" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.new_products_mass
        #     if condition == "prod_stock_used_mass" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.used_products_mass
        #     elif condition == "product_repaired" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_product_repaired
        #     elif condition == "product_sold" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_product_sold
        #         count2 += agent.number_product_sold
        #         count2 += agent.number_product_repaired
        #     elif condition == "product_recycled" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_product_recycled
        #         count += industrial_waste_recycled
        #     elif condition == "product_landfilled" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_product_landfilled
        #         count += industrial_waste_landfill
        #     elif condition == "product_hoarded" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_product_hoarded
        #     elif condition == "product_new_repaired" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_new_prod_repaired
        #     elif condition == "product_new_sold" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_new_prod_sold
        #     elif condition == "product_new_recycled" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_new_prod_recycled
        #         count += industrial_waste_recycled_mass
        #     elif condition == "product_new_landfilled" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_new_prod_landfilled
        #         count += industrial_waste_landfill_mass
        #     elif condition == "product_new_hoarded" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_new_prod_hoarded
        #     elif condition == "product_used_repaired" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_used_prod_repaired
        #     elif condition == "product_used_sold" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_used_prod_sold
        #     elif condition == "product_used_recycled" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_used_prod_recycled
        #     elif condition == "product_used_landfilled" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_used_prod_landfilled
        #     elif condition == "product_used_hoarded" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.number_used_prod_hoarded
        #     elif condition == "consumer_costs" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.consumer_costs
        #     elif condition == "average_landfill_cost" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.landfill_cost / self.num_consumers
        #     elif condition == "average_hoarding_cost" and agent.unique_id < \
        #             self.num_consumers:
        #         count += agent.hoarding_cost / self.num_consumers
        #     elif condition == "average_recycling_cost" and self.num_consumers\
        #             <= agent.unique_id < self.num_consumers + \
        #             self.num_recyclers:
        #         count += agent.recycling_cost / self.num_recyclers
        #     elif condition == "average_repairing_cost" and self.num_consumers\
        #             + self.num_prod_n_recyc <= agent.unique_id:
        #         count += agent.repairing_cost / self.num_sechdstores
        #     elif condition == "average_second_hand_price" and \
        #             self.num_consumers + self.num_prod_n_recyc <= agent.unique_id:
        #         count += (-1 * agent.scd_hand_price) / self.num_sechdstores
        #     elif condition == "weight":
        #         count = self.dynamic_product_average_wght
        #     elif condition == "recycled_mat_volume" and self.num_consumers + \
        #             self.num_recyclers <= agent.unique_id < \
        #             self.num_consumers + self.num_prod_n_recyc:
        #         if not np.isnan(agent.recycled_material_volume):
        #             count += agent.recycled_material_volume
        #     elif condition == "recycled_mat_value" and self.num_consumers + \
        #             self.num_recyclers <= agent.unique_id < \
        #             self.num_consumers + self.num_prod_n_recyc:
        #         if not np.isnan(agent.recycled_material_value):
        #             count += agent.recycled_material_value
        #     elif condition == "producer_costs" and self.num_consumers + \
        #             self.num_recyclers <= agent.unique_id < \
        #             self.num_consumers + self.num_prod_n_recyc:
        #         count += agent.producer_costs
        #     elif condition == "recycler_costs" and self.num_consumers <= \
        #             agent.unique_id < self.num_consumers + \
        #             self.num_recyclers:
        #         count += agent.recycler_costs
        #     elif condition == "refurbisher_costs" and self.num_consumers + \
        #             self.num_prod_n_recyc <= agent.unique_id:
        #         count += agent.refurbisher_costs
        #     elif condition == "refurbisher_costs_w_margins" and self.num_consumers + \
        #             self.num_prod_n_recyc <= agent.unique_id:
        #         count += agent.refurbisher_costs_w_margins
        
        # if condition == "product_sold":
        #     self.sold_repaired_waste += count2 - self.past_sold_repaired_waste
        #     self.past_sold_repaired_waste = count2
        return count

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        self.update_statistics()
        self.agents_by_type[Manufacturer].shuffle_do('step')
        self.agents_by_type[SecondHandStore].shuffle_do('step')
        self.agents_by_type[Consumer].shuffle_do('step')
        self.agents_by_type[Recycler].shuffle_do('step')
        self.datacollector.collect(self)
