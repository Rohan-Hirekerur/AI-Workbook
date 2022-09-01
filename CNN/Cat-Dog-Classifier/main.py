from src import cat_dog_classifier

classifier = cat_dog_classifier.model("ep_25")
classifier.train(25)
print(classifier.predict())