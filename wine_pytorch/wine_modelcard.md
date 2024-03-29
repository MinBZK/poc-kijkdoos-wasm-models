---
model: wine_pytorch.onnx
output:
  probabilities: output
input: input
features:
  - alcohol: 13.0
  - malic_acid: 2.0
  - ash: 2.5
  - alcalinity_of_ash: 15.0
  - magnesium: 100.0
  - total_phenols: 2.0
  - flavanoids: 2.0
  - nonflavanoid_phenols: 0.5
  - proanthocyanins: 1.0
  - color_intensity: 5.0
  - hue: 0.5
  - od280/od315_of_diluted_wines: 2.0
  - proline: 500.0

---

# Wine model in pytorch
