import os
import logging
import mesa
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from ABM_Model import SmartphoneModel
from ABM_Smartphone import Smartphone
from ABM_Manufacturer import Manufacturer

from custom_logger import setup_logger
from utils.plot_utils import *


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    csv_file = 'results/output_batch.csv'
    results = mesa.batch_run(
        SmartphoneModel,
        parameters={},
        iterations=1,
        max_steps=120,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_file)

    setup_logger(filename='logs/ABM.log')
    # logging.info(results_df.keys())

    plot_consumer_income(results_df)
    plot_buying_action(results_df)
    plot_eol_action(results_df)
    plot_product_price(results_df)
    plot_price2income(results_df)