# import churn_modeller from "./src/ann.py"
from src import churn_modeller
model = churn_modeller.model()
model.train("./src/dataset/bank_churn_data.csv", 32, 10, 'bs32_ep100')
print(model.predict([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]], 'bs32_ep100'))
