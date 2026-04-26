import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

print(model.classes_)
