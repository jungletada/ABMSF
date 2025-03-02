import os
import logging
import mesa
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter

from ABM_Model import AgentBasedModel
from ABM_Smartphone import Smartphone
from ABM_Manufacturer import Manufacturer

from custom_logger import setup_logger
from utils.plot_utils import *


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    csv_file = 'results/output_batch.csv'
    # results = mesa.batch_run(
    #     AgentBasedModel,
    #     parameters={},
    #     iterations=1,
    #     max_steps=120,
    #     number_processes=1,
    #     data_collection_period=1,
    #     display_progress=True,
    # )
    # results_df = pd.DataFrame(results)
    # results_df.to_csv(csv_file)

    results_df = pd.read_csv('results/_output_batch_1.csv')
    setup_logger(filename='logs/ABM.log')
    logging.info(results_df.keys())

    plot_buying_action(results_df)
    plot_eol_action(results_df)
    plot_product_price_income(results_df)
    plot_price2income(results_df)
    plot_eol_pie(results_df)
    plot_buying_pie(results_df)
    #############################################################
    total_new = results_df["consumer_buying_new"].sum()
    total_used = results_df["consumer_buying_used"].sum()
    buying_array = np.array([total_new, total_used])
    all_num_buying = buying_array.sum()
    ratio_buying = buying_array / all_num_buying * 100.
    logging.info(
        f'Consumers buying action={buying_array}\n'
        f' new,   used \n'
        f'{[f"{x:.2f}" for x in ratio_buying]}')
    
    num_proffer = results_df["consumer_proffer"].sum()
    num_selling = results_df["consumer_reselling"].sum()
    num_recycling = results_df["consumer_recycling"].sum()
    num_landfilling = results_df["consumer_landfilling"].sum()
    num_storing = results_df["consumer_storing"].sum()
    eol_array = np.array([num_proffer, num_selling, num_recycling, num_landfilling, num_storing])
    all_num_eol = eol_array.sum()
    ratio_eol = eol_array / all_num_eol * 100.
    logging.info(
        f'Consumers EoL actions={eol_array}\n'
        f'proffer, resell, recycle, landfill, store\n'
        f'{[f"{x:.2f}" for x in ratio_eol]}')
    