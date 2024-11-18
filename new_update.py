import numpy as np


class ConsumerChoiceModel:
    def __init__(self, income, beta1, beta2, beta3):
        """
        Initialize the consumer choice model.

        Parameters:
            income (float): Consumer's income.
            beta1 (float): Weight for income effect.
            beta2 (float): Weight for price effect.
            beta3 (float): Weight for features effect.
        """
        self.income = income
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def calculate_utility(self, price, features):
        """
        Calculate the utility for a given smartphone.

        Parameters:
            price (float): Price of the smartphone.
            features (float): Features/quality score of the smartphone.

        Returns:
            float: Utility value.
        """
        return self.beta1 * np.log(self.income) - self.beta2 * price + self.beta3 * features

    def choose_smartphone(self, smartphones):
        """
        Choose the smartphone with the highest utility.

        Parameters:
            smartphones (list of dict): List of smartphones, each represented by a dictionary
                                        with keys "price" and "features".

        Returns:
            int: Index of the chosen smartphone.
        """
        utilities = [self.calculate_utility(phone["price"], phone["features"]) for phone in smartphones]
        probabilities = np.exp(utilities) / np.sum(np.exp(utilities))
        chosen_index = np.argmax(probabilities)  # Smartphone with the highest probability
        return chosen_index, probabilities

# Example usage
consumer_income = 50000
beta1 = 1.0  # Importance of income
beta2 = 0.01  # Importance of price
beta3 = 5.0  # Importance of features

smartphones = [
    {"price": 1000, "features": 0.9},  # Smartphone 1
    {"price": 800, "features": 0.8},   # Smartphone 2
    {"price": 1200, "features": 0.95}  # Smartphone 3
]

consumer_model = ConsumerChoiceModel(consumer_income, beta1, beta2, beta3)
chosen_index, probabilities = consumer_model.choose_smartphone(smartphones)

print(f"Chosen smartphone index: {chosen_index}")
print(f"Choice probabilities: {probabilities}")
