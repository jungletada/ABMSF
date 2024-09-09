# Agent-based Modeling of Social and Financial Impacts on Smartphone Reuse and Recycling

[ABMSF: Project Page](https://github.com/jungletada/ABMSF)   
[References: ABSiCE](https://github.com/NREL/ABSiCE)  

## Libraries:
- [Mesa: Agent-based modeling in Python](https://mesa.readthedocs.io/en/stable/index.html)

## Diffuculties:
1. Environment：美国50个州 $\rightarrow$ 日本各都道府县？
2. 计算PV材料的Efficiency growth[S1]: ${MP}_t=be^{-at},$
其中 $b$ and $a$ are the intercept and regression coefficient. PV通过function unit (fu)，计算得到PV材料的数量 $\rightarrow$ 计算手机数量(个)？ *PV质量为连续值，手机数量应为离散值.*   
代码对应于 `ABM_CE_ConsumerAgents.py`, 函数`mass_per_function_model(self, product_as_function):`
    ```python
    # product_average_wght (kg/fu), (b=0.1kg/Wp). fu=Wp
    # mass_to_function_reg_coeff, (a=0.03year).
    mass_conversion_coeffs = [self.model.product_average_wght * \
                math.e ** (-self.model.mass_to_function_reg_coeff * x) 
                    for x in range(len(product_as_function))] 
    # mass_conversion_coeffs: kg/Wp, 通过function unit计算得到质量(kg)
    # product_as_function: Number of EOL smartphones in fu
    product_as_mass = [product_as_function[i] * mass_conversion_coeffs[i]
                        for i in range(len(product_as_function))]
    ```
3. 对于二手店，是否需要考虑加上回收商在 $t$ 时刻流出的手机数量 $V_k^t$.
    $$
    \begin{aligned}
    V_j^t&=\frac{1}{\sum_j}\left(RR\times\left(\sum_i V_i^t +\sum_k V_k^t\right)\right)\\
    \rightarrow V_j^t&=\frac{1}{\sum_j}\left(RR\times\sum_i V_i^t\right)
    \end{aligned}
    $$ 

4. 生产商产生的废物 [S9] $\rightarrow$ 回收商产生的废物
    $${IW}_m^t=\sum_i {PA}_i^t\times{MCE}_t\times{MF}_m\times{IWR}_m$$
    ${IW}_m^t$: 生产商在 $t$ 时刻产生的材料 $m$ 的生产废物量（克）.  
    ${IWR}_m$: 材料 $m$ 的生产废料率.

## ToDo
1. Agents decision tree
