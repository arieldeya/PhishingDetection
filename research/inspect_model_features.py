import joblib

MODEL_FILE = "final_random_forest_phishing_model.pkl"

print("=" * 100)
print("INSPECTING TRAINED PHISHING MODEL")
print("=" * 100)

model = joblib.load(MODEL_FILE)

print("\nModel type:")
print(type(model))

print("\nModel:")
print(model)

print("\nModel classes:")
print(model.classes_)

print("\nNumber of features:")
print(getattr(model, "n_features_in_", "Not available"))

print("\nFeature names:")
feature_names = getattr(model, "feature_names_in_", None)

if feature_names is not None:

    for i, feature in enumerate(feature_names):

        print(f"{i:3d}. {feature}")

else:

    print("Model does not contain feature_names_in_")

print("\n" + "=" * 100)
print("MODEL INSPECTION COMPLETE")
print("=" * 100)