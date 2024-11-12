from ABM_Model import Smartphone_MODEL

if __name__ == "__main__":
    model = Smartphone_MODEL()
    for t in range(1):
        model.step()
        
        