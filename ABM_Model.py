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


class SmartphoneModel(Model):
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
            num_sechdstores=40,
            prod_n_recyc_node_degree=5,
            prod_n_recyc_network_type="small-world",
            init_consumer_purchase_dist={'used':0.2, 'new':0.8},
            consumers_distribution={"residential": 1, "commercial": 0., "utility": 0.},
            init_eol_rate={"repair": 0.005, "sell": 0.01, "recycle": 0.1, "landfill": 0.4425, "hoard": 0.4425},
            init_purchase_rate={"new": 0.9995, "used": 0.0005},
            product_growth=[0.166, 0.045],
            growth_threshold=10,
            hoarding_cost=[0, 0.001, 0.0005],
            landfill_cost=[],
            all_eol_pathways={"repair": True, "sell": True,
                            "recycle": True, "landfill": True,
                            "hoard": True},
            att_distrib_param_eol=[0.544, 0.1],
            att_distrib_param_reuse=[0.223, 0.262],
            repairability=0.55,
        ):
        """
        Initiate model
        """
        # Set up variables
        super().__init__(seed=seed)
        self.seed = seed
        att_distrib_param_eol[0] = calibration_n_sensitivity
        att_distrib_param_reuse[0] = calibration_n_sensitivity_2
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.num_consumers = num_consumers
        self.consumers_node_degree = consumers_node_degree
        self.consumers_network_type = consumers_network_type
        
        self.num_recyclers = num_recyclers # 回收商
        self.num_producers = num_producers # 制造商
        self.num_prod_n_recyc = num_recyclers + num_producers # 回收商+制造商
        self.num_sechdstores = num_sechdstores # 二手商
        
        self.prod_n_recyc_node_degree = prod_n_recyc_node_degree
        self.prod_n_recyc_network_type = prod_n_recyc_network_type

        self.init_eol_rate = init_eol_rate # dictionary with initial end-of-life (EOL) ratios
        self.init_purchase_choice = init_purchase_rate # dictionary with initial purchase ratios
        
        self.init_product_prices = np.linspace(799, 7999, num_producers)
        self.clock = 0
        self.iteration = 0
        self.running = True
        self.color_map = []
        self.all_eol_pathways = all_eol_pathways
        
        self.avg_new_product_price = 5000   # TBD
        self.avg_used_product_price = 3000   # TBD
        self.all_comsumer_income = []
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
                    init_purchase_dist=0.1,
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
                manufacturer = Manufacturer(
                    model=self,
                    unique_id=node,
                    init_product_price=int(self.init_product_prices[producer_index])
                    )
                self.grid.place_agent(manufacturer, node)
                self.new_product_id_price[node] = manufacturer.product_price
            else:
                #================== Second-hand Store ==================#
                sechdstore = SecondHandStore(
                    model=self,
                    unique_id=node,
                    )
                self.grid.place_agent(sechdstore, node)

        # Defines reporters and setup data collector
        model_reporters = {
            "schedule": lambda c: self.report_output("schedule"),
            "avg_consumer_income": lambda c: self.report_output("income"),
            # "Agents repairing": lambda c: self.count_eol_products("repairing"),
            # "Agents selling": lambda c: self.count_eol_products("selling"),
            # "Agents recycling": lambda c: self.count_eol_products("recycling"),
            # "Agents landfilling": lambda c: self.count_eol_products("landfilling"),
            # "Agents storing": lambda c: self.count_eol_products("hoarding"),
            # "Agents buying new": lambda c: self.count_eol_products("buy_new"),
            # "Agents buying used": lambda c: self.count_eol_products("buy_used"),
            # "Total product": lambda c:self.report_output("product_stock"),
            # "New product": lambda c:self.report_output("product_stock_new"),
            # "Used product": lambda c:self.report_output("product_stock_used"),
            # "New product_mass": lambda c:self.report_output("prod_stock_new_mass"),
            # "Used product_mass": lambda c:self.report_output("prod_stock_used_mass"),
            # "EoL-repaired": lambda c:self.report_output("product_repaired"),
            # "EoL-sold": lambda c: self.report_output("product_sold"),
            # "EoL-recycled": lambda c:self.report_output("product_recycled"),
            # "EoL-landfilled": lambda c:self.report_output("product_landfilled"),
            # "EoL-stored": lambda c:self.report_output("product_hoarded"),
            # "EoL-new repaired weight": lambda c:self.report_output("product_new_repaired"),
            # "EoL-new sold weight": lambda c:self.report_output("product_new_sold"),
            # "EoL-new recycled weight": lambda c:self.report_output("product_new_recycled"),
            # "EoL-new landfilled weight": lambda c:self.report_output("product_new_landfilled"),
            # "EoL-new stored weight": lambda c:self.report_output("product_new_hoarded"),
            # "EoL-used repaired weight": lambda c:self.report_output("product_used_repaired"),
            # "EoL-used sold weight": lambda c:self.report_output("product_used_sold"),
            # "EoL-used recycled weight": lambda c:self.report_output("product_used_recycled"),
            # "EoL-used landfilled weight": lambda c:self.report_output("product_used_landfilled"),
            # "EoL-used stored weight": lambda c:self.report_output("product_used_hoarded"),
            # "Average landfilling cost": lambda c:self.report_output("average_landfill_cost"),
            # "Average storing cost": lambda c:self.report_output("average_hoarding_cost"),
            # "Average recycling cost": lambda c:self.report_output("average_recycling_cost"),
            # "Average repairing cost": lambda c:self.report_output("average_repairing_cost"),
            # "Average selling cost": lambda c:self.report_output("average_second_hand_price"),
            # "Recycled material volume": lambda c:self.report_output("recycled_mat_volume"),
            # "Recycled material value": lambda c:self.report_output("recycled_mat_value"),
            # "Producer costs": lambda c:self.report_output("producer_costs"),
            # "Consumer costs": lambda c:self.report_output("consumer_costs"),
            # "Recycler costs": lambda c:self.report_output("recycler_costs"),
            # "Refurbisher costs": lambda c:self.report_output("refurbisher_costs"),
            # "Refurbisher costs w margins": lambda c:self.report_output("refurbisher_costs_w_margins")
            }

        agent_reporters = {
            "schedule": 
                lambda c: self.report_output("schedule"),
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

    def waste_generation(self, avg_lifetime, failure_rate, num_product):
        """
        Generate waste, called by consumers and recyclers/refurbishers
        (to get original recycling/repairing amounts).
        """
        correction_year = len(self.total_number_product) - 1
        return [j * (1 - math.e ** (-(((self.clock + (correction_year - z)) /
                               avg_lifetime[z]) ** failure_rate))).real
                for (z, j) in enumerate(num_product)]

    def count_eol_products(self, condition):
        """
        Count adoption in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        for agent in self.agents:
            if agent.unique_id < self.num_consumers:
                if condition == "repairing" and agent.EoL_pathway == "repair":
                    count += 1
                elif condition == "selling" and agent.EoL_pathway == "sell":
                    count += 1
                elif condition == "recycling" and agent.EoL_pathway == "recycle":
                    count += 1
                elif condition == "landfilling" and agent.EoL_pathway == "landfill":
                    count += 1
                elif condition == "hoarding" and agent.EoL_pathway == "hoard":
                    count += 1
                elif condition == "buy_new" and agent.purchase_choice == "new":
                    count += 1
                elif condition == "buy_used" and agent.purchase_choice == "used":
                    count += 1
                    self.consumer_used_product += 1
                elif condition == "buy_certified" and \
                        agent.purchase_choice == "certified":
                    count += 1
                else:
                    continue
            else:
                continue
        return count

    def count_statistics(self):
        self.all_comsumer_income = [agent.income for agent in self.agents_by_type[Consumer]]

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
        
        for agent in self.agents:
            if condition == "income" and isinstance(agent, Consumer):
                count = np.mean(self.all_comsumer_income)
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
        #     elif condition == "schedule":
        #         count = self.steps
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
        self.count_statistics()
        self.agents_by_type[Manufacturer].shuffle_do('step')
        self.agents_by_type[SecondHandStore].shuffle_do('step')
        self.agents_by_type[Consumer].shuffle_do('step')
        self.agents_by_type[Recycler].shuffle_do('step')

        # Collect data
        self.datacollector.collect(self)
