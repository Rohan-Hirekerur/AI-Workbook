# import churn_modeller from "./src/ann.py"
from src import churn_modeller
model = churn_modeller.model('bs32_ep100')
model.train(data_source="bank_churn_data.csv",
            batch_size=32, epochs=100)
# print(model.predict(
#     [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]], 'bs32_ep100'))
print(model.predict([['France', 600, "Male", 40, 3,
                      60000, 2, "Yes", "Yes", 50000]]))
