import numpy as np
import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define feature names
feature_names = [
    "floor_area",
    "plot_size",
    "building_year",
    "object_type",
    "num_annexes",
    "neighborhood_code",
    "quality_rating",
    "maintenance_rating",
    "amenities_rating",
    "location_rating",
]

# Create a more realistic dataset for demonstration
np.random.seed(42)
n_samples = 10000

# Generate features
floor_area = np.random.uniform(30, 500, n_samples)  # 30 to 500 m²
plot_size = np.random.uniform(0, 1000, n_samples)  # 0 to 1000 m²
building_year = np.random.randint(1900, 2024, n_samples)
object_type = np.random.randint(1, 5, n_samples)
num_annexes = np.random.randint(0, 3, n_samples)
neighborhood_code = np.random.randint(1000, 9999, n_samples)
quality_rating = np.random.uniform(1, 5, n_samples)
maintenance_rating = np.random.uniform(1, 5, n_samples)
amenities_rating = np.random.uniform(1, 5, n_samples)
location_rating = np.random.uniform(1, 5, n_samples)

X = np.column_stack(
    (
        floor_area,
        plot_size,
        building_year,
        object_type,
        num_annexes,
        neighborhood_code,
        quality_rating,
        maintenance_rating,
        amenities_rating,
        location_rating,
    )
)

# Generate target values with more emphasis on floor area and plot size
base_value = 200000 + (floor_area * 2000) + (plot_size * 500)
year_factor = np.clip((building_year - 1900) / 124, 0, 1) * 100000
quality_factor = (quality_rating + maintenance_rating + amenities_rating + location_rating) / 4 * 100000
random_factor = np.random.normal(0, 50000, n_samples)

y = np.clip(base_value + year_factor + quality_factor + random_factor, 200000, 1500000)

# Create a pipeline with scaling and Random Forest regression
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), [0, 1, 2, 6, 7, 8, 9]),  # Numerical features
        ("cat", "passthrough", [3, 4, 5]),  # Categorical features
    ]
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ]
)

# Fit the model
model.fit(X, y)

# Define the ONNX model input type (single input tensor)
input_type = [("input", FloatTensorType([None, 10]))]

# Convert the scikit-learn pipeline to ONNX
onnx_model = convert_sklearn(model, "woz_model", input_type, target_opset=9)

# Save the ONNX model
with open("woz/woz_simplified_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model created and saved as 'woz_simplified_model.onnx'")

# Create a sample input (single numpy array)
sample_input = np.array(
    [
        [
            150.0,  # floor_area
            300.0,  # plot_size
            1980,  # building_year
            1,  # object_type
            1,  # num_annexes
            1234,  # neighborhood_code
            4.0,  # quality_rating
            3.5,  # maintenance_rating
            4.5,  # amenities_rating
            4.0,  # location_rating
        ]
    ],
    dtype=np.float32,
)

# Create an ONNX Runtime session
sess = rt.InferenceSession("woz/woz_simplified_model.onnx")

# Run the model
output = sess.run(None, {"input": sample_input})

woz_value = output[0][0][0]  # Access the scalar value

print(f"Estimated WOZ value: {woz_value:.2f}")

# Test with multiple random inputs
test_inputs = np.random.rand(10, 10).astype(np.float32)
test_inputs[:, 0] *= 470  # Scale floor_area to 0-470 m²
test_inputs[:, 1] *= 1000  # Scale plot_size to 0-1000 m²
test_outputs = sess.run(None, {"input": test_inputs})

print("\nTest with multiple random inputs:")
for i, (inp, out) in enumerate(zip(test_inputs, test_outputs[0])):
    print(f"Sample {i+1}:")
    print(f"  Floor Area: {inp[0]:.2f} m², Plot Size: {inp[1]:.2f} m²")
    print(f"  Estimated WOZ value: {out[0]:.2f}")

# Print feature importances
importances = model.named_steps["regressor"].feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Print ONNX and ONNX Runtime versions
print(f"\nONNX version: {onnx.__version__}")
print(f"ONNX Runtime version: {rt.__version__}")
