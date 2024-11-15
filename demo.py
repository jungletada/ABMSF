from ABM_Model import Smartphone_MODEL
from ABM_Smartphone import Smartphone
from ABM_Manufacturer import Manufacturer


model = Smartphone_MODEL()

if __name__ == "__main__":
    for t in range(36):
        model.step()