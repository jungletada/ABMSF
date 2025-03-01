import os
import matplotlib.pyplot as plt


FIG_DPI = 500
SAVE_DIR = 'results'

def plot_consumer_income(results_df):    
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["avg_consumer_income"],
        label="buying new")
    plt.title("Average Consumer Income Over Time")
    plt.xlabel("Months")
    plt.ylabel("Income")
    plt.xlim((1, 121))
    plt.savefig(os.path.join(SAVE_DIR, 'avg_consumer_income.png'), dpi=FIG_DPI)


def plot_buying_action(results_df):
    plt.figure(plt.figure(figsize=(20, 6)))
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.plot(
        results_df["Step"],
        results_df["consumer_buying_new"])
    ax1.set_title("Consumer Buying New")
    
    ax2.plot(
        results_df["Step"],
        results_df["consumer_buying_used"])
    ax2.set_title("Consumer Buying Used")

    # Set x-axis limits and ticks
    ax1.set_xlim(0, 120)  # Set x-axis range
    ax2.set_xlim(0, 120)
    
    ax1.set_xticks(range(0, 121, 20))  # Set x-axis ticks at intervals of 20
    ax2.set_xticks(range(0, 121, 20))
    plt.savefig(os.path.join(SAVE_DIR, 'agents_buying_actions.png'), dpi=FIG_DPI)


def plot_eol_action(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["consumer_repairing"],
        label="repairing")
    plt.plot(
        results_df["Step"],
        results_df["consumer_selling"],
        label="selling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_recycling"],
        label="recycling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_landfilling"],
        label="landfilling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_storing"],
        label="storing")
    
    plt.legend()
    plt.xlim((1, 120))
    plt.xlabel("Step")
    plt.ylabel("Agent Count")
    plt.title("Consumer Actions")
    plt.savefig(os.path.join(SAVE_DIR, 'agents_eol_actions.png'), dpi=FIG_DPI)


def plot_product_price(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["avg_new_product_price"],
        label="new")
    plt.plot(
        results_df["Step"],
        results_df["avg_used_product_price"],
        label="used")
    plt.title("Average Product Price Over Time")
    plt.xlabel("Months")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'avg_product_price.png'), dpi=FIG_DPI)


def plot_price2income(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df['new_price_to_income'],
        label="new")
    plt.plot(
        results_df["Step"],
        results_df['used_price_to_income'],
        label="used")
    
    plt.title("Average Product Price to Income Ratio Over Time")
    plt.xlabel("Months")
    plt.ylabel("Price-to-Income")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'price_to_income.png'), dpi=FIG_DPI)
