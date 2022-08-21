from src import churn_modeller
import os

print("\n-----------------------------------------------")
print("Bank Churn Modeller")
print("-----------------------------------------------")

print("\nWhat do you want to do?")
print("1. Train a new model")
print("2. Determine churn for a customer")

user_choice = input()
while (user_choice not in ["1", "2"]):
    user_choice = input("Please enter a valid option...\n")

if(user_choice == "1"):
    datasets = os.listdir("src/dataset")

    print("Which dataset do you want to use?")
    for idx, file_name in enumerate(datasets):
        print("{}. {}".format(idx+1, file_name))

    selected_dataset_number = int(input())
    while (selected_dataset_number > len(datasets)):
        selected_dataset_number = int(
            input("Please enter a valid option...\n"))

    data_source = datasets[selected_dataset_number - 1]
    model_name = input("Give a name for your model : ")
    batch_size = input("Specify batch size : ")
    epochs = input("Specify number of epochs : ")

    print("\nTraining your model...\n")
    model = churn_modeller.model(model_name)
    model.train(data_source=data_source,
                batch_size=batch_size, epochs=epochs)

else:
    models = os.listdir("src/models/ann")

    print("Which dataset do you want to use?")
    for idx, file_name in enumerate(models):
        print("{}. {}".format(idx+1, file_name))

    selected_model_no = int(input())
    while (selected_model_no > len(models)):
        selected_model_no = int(
            input("Please enter a valid option...\n"))

    model_name = models[selected_model_no - 1]

    print("Enter customer details :")
    cust_id = input("customer Id : ")
    name, surname = input(
        "Name of customer (format: Name Surname) : ").split(" ")
    credit_score = input("Credit score : ")
    geography = input("Geography : ")
    gender = input("Gender : (Male/Female) : ")
    age = input("Age : ")
    tenure = input("Tenure : ")
    balance = input("Balance : ")
    no_of_products = input("Number of products : ")
    has_credit_card = input("Has credit card ? (Enter 1 for Yes, 0 for No) : ")
    is_active_member = input(
        "Is an active member ? (Enter 1 for Yes, 0 for No) : ")
    estimated_salary = input("Estimated salary : ")

    model = churn_modeller.model(model_name)
    will_churn = model.predict([[cust_id, name, surname, credit_score, geography, gender, age, tenure,
                                 balance, no_of_products, has_credit_card, is_active_member, estimated_salary]])[0][0]
    print("\nAnswer to will {} {} leave the bank is : {}".format(
        name, surname, will_churn))
