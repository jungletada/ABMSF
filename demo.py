from ABM_Model import SmartphoneModel
from ABM_Smartphone import Smartphone
from ABM_Manufacturer import Manufacturer


model = SmartphoneModel()


if __name__ == "__main__":
    for t in range(80):
        model.step()
        