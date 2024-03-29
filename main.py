import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
import torch.optim as optim
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = LogisticRegression()
clr.fit(X_train, y_train)

initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onx = convert_sklearn(clr, initial_types=initial_type, target_opset=12, options={type(clr): {"zipmap": False}})
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("logreg_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (you can replace this with any other classifier you prefer)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Convert the classifier to ONNX format
initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onx = convert_sklearn(clf, initial_types=initial_type, target_opset=12, options={type(clf): {"zipmap": False}})
with open("random_forest_breast_cancer.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Load the ONNX model and perform inference
sess = rt.InferenceSession("random_forest_breast_cancer.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)

# Load Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a more complex neural network model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)  # 13 input features, 64 hidden units
        self.fc2 = nn.Linear(64, 32)  # 64 hidden units, 32 hidden units
        self.fc3 = nn.Linear(32, 3)  # 32 hidden units, 3 output classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model
model = Model()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Accuracy on test set: {accuracy:.2f}")

# Export the model to ONNX format
dummy_input = torch.randn(1, 13)  # Example input tensor
output_path = "wine_pytorch.onnx"
torch.onnx.export(model, dummy_input, output_path, verbose=True, input_names=["input"], output_names=["output"])
print(f"Model exported to {output_path}")
