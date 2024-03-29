---
model: logreg_iris.onnx
output:
  probabilities: probabilities
  class: label
input: float_input
features:
- sepal_length: 1
- sepal_width: 2
- petal_length: 3
- petal_width: 4
---

# Iris dataset classifier

This is the default iris dataset classifier
