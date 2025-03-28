import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIG_DPI = 500
SAVE_DIR = 'results'

income_color = '#FFB6C1'
new_color = '#6A5ACD'
used_color = '#20B2AA'


def plot_buying_action(results_df):
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

    ax1.plot(
        results_df["Step"],
        results_df["consumer_buying_new"],
        color=new_color)
    ax1.set_xlabel("Month")
    ax1.set_title("Consumer Buying New")
    
    ax2.plot(
        results_df["Step"],
        results_df["consumer_buying_used"],
        color=used_color)
    ax2.set_title("Consumer Buying Used")

    # Set x-axis limits and ticks
    ax1.set_xlim(0, 120)  # Set x-axis range
    ax2.set_xlim(0, 120)
    ax2.set_xlabel("Month")
    ax1.set_xticks(range(0, 121, 24))  # Set axis ticks at intervals of 12
    ax2.set_xticks(range(0, 121, 24))
    plt.savefig(os.path.join(SAVE_DIR, 'agents_buying_actions.png'), dpi=FIG_DPI)
    plt.savefig(os.path.join(SAVE_DIR, 'agents_buying_actions.eps'), dpi=FIG_DPI)
     # 将 total_new 和 total_used 保存到 Excel 文件中
    # df_totals = pd.DataFrame({
    #     "Type": ["step, ""new", "used"],
    #     "Total": [results_df["Step"], 
    #               results_df["consumer_buying_new"], 
    #               results_df["consumer_buying_used"]]
    # })
    # df_totals.to_excel(os.path.join(SAVE_DIR, 'agents_buying_totals.xlsx'), index=False)


def plot_eol_action(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(
        results_df["Step"],
        results_df["consumer_proffer"],
        color='#6495ED', 
        label="proffer")
    plt.plot(
        results_df["Step"],
        results_df["consumer_reselling"],
        color='#8B008B', 
        label="selling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_recycling"],
        color='#008080',
        label="recycling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_landfilling"],
        color='#FF7F50', 
        label="landfilling")
    plt.plot(
        results_df["Step"],
        results_df["consumer_storing"],
        color='#FFDAB9',
        label="storing")
    
    plt.legend()
    plt.xticks(range(0, 121, 12))
    plt.xlabel("Month")
    plt.ylabel("Agent Count")
    plt.title("Consumer Actions")
    plt.savefig(os.path.join(SAVE_DIR, 'agents_eol_actions.png'), dpi=FIG_DPI)


def plot_product_price_income(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df["avg_consumer_income"],
        color=income_color,
        label="income")
    plt.plot(
        results_df["Step"],
        results_df["avg_new_product_price"],
        color=new_color,
        label="new")
    plt.plot(
        results_df["Step"],
        results_df["avg_used_product_price"],
        color=used_color,
        label="used")
    plt.title("Average Consumer Income and Product Price Over Time")
    plt.xlabel("Months")
    plt.ylabel("CNY")
    plt.legend()
    plt.xticks(range(0, 121, 12))
    plt.savefig(os.path.join(SAVE_DIR, 'avg_price_income.png'), dpi=FIG_DPI)


def plot_price2income(results_df):
    plt.figure()
    plt.plot(
        results_df["Step"],
        results_df['new_price_to_income'],
        color=new_color,
        label="new")
    plt.plot(
        results_df["Step"],
        results_df['used_price_to_income'],
        color=used_color,
        label="used")
    
    plt.title("Average Product Price to Income Ratio Over Time")
    plt.xlabel("Months")
    plt.ylabel("Price-to-Income")
    plt.legend()
    plt.xticks(range(0, 121, 12))
    plt.savefig(os.path.join(SAVE_DIR, 'price_to_income.png'), dpi=FIG_DPI)


def plot_eol_pie(results_df):
    num_proffer = results_df["consumer_proffer"].sum()
    num_selling = results_df["consumer_reselling"].sum()
    num_recycling = results_df["consumer_recycling"].sum()
    num_landfilling = results_df["consumer_landfilling"].sum()
    num_storing = results_df["consumer_storing"].sum()
    
    num_rec_mnf = results_df["consumer_recycle_manufactor"].sum()
    num_rec_rcl = results_df["consumer_recycle_recycler"].sum()
    
    eol_array = np.array([num_proffer, num_selling, num_recycling, num_landfilling, num_storing])
    eol_labels = ['proffer', 'reselling', 'recycling', 'landfilling', 'storing']
    eol_colors = [
        '#6495ED',
        '#8B008B', 
        '#008080',
        '#FF7F50',
        '#FFDAB9',
    ]
    rec_array = np.array([num_rec_mnf, num_rec_rcl])
    rec_labels = ['to manufacturer', 'to recycler']
    rec_colors = ['#9BBBE1', '#F09BA0'] 
    
    # 建立画布 + 两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # -- 子图1：EOL Actions 总览 --
    ax1.pie(
        eol_array,
        labels=eol_labels,
        autopct='%1.2f%%',
        startangle=140,
        colors=eol_colors
    )
    ax1.set_title('EOL Actions Distribution')
    
    # -- 子图2：Recycling 内部分布 --
    ax2.pie(
        rec_array,
        labels=rec_labels,
        autopct='%1.2f%%',
        startangle=140,
        colors=rec_colors
    )
    ax2.set_title('Recycling Distribution')
    
    # 自动调整布局，防止重叠
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'agents_eol_pie.png'), dpi=FIG_DPI)
    
    
def plot_buying_pie(results_df):
    total_new = results_df["consumer_buying_new"].sum()
    total_used = results_df["consumer_buying_used"].sum()
    buying_array = np.array([total_new, total_used])
    labels = ['new', 'used']
    colors = [
        new_color,
        used_color, 
    ]
    plt.figure()
    plt.pie(
        buying_array,
        labels=labels,
        autopct='%1.2f%%',
        startangle=140,
        colors=colors
    )
    plt.title('Buying Actions Distribution')
    plt.savefig(os.path.join(SAVE_DIR, 'agents_buying_pie.png'), dpi=FIG_DPI)