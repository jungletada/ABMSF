from ABM_Model import Smartphone_MODEL
from ABM_Smartphone import Smartphone

if __name__ == "__main__":
    model = Smartphone_MODEL()
    for t in range(36):
        model.step()
    # a = Smartphone(model=model, 
    #                is_new=False, producer_id=0, user_id=0, performance=0.5, time_held=0, demand_used=0.3, 
    #                product_price=5000, initial_repair_cost=500, decay_rate=0.1)
    # b = a.copy()
    # a.user_id = 15151
    # print(b.user_id)