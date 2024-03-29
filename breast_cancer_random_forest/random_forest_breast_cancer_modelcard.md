---
model: random_forest_breast_cancer.onnx
output:
  probabilities: probabilities
  class: label
input: float_input
features:
  - mean_radius: 15.0
  - mean_texture: 18.0
  - mean_perimeter: 120.0
  - mean_area: 1000.0
  - mean_smoothness: 0.1
  - mean_compactness: 0.2
  - mean_concavity: 0.15
  - mean_concave_points: 0.1
  - mean_symmetry: 0.2
  - mean_fractal_dimension: 0.05
  - radius_error: 0.5
  - texture_error: 0.3
  - perimeter_error: 3.0
  - area_error: 50.0
  - smoothness_error: 0.01
  - compactness_error: 0.05
  - concavity_error: 0.03
  - concave_points_error: 0.02
  - symmetry_error: 0.08
  - fractal_dimension_error: 0.02
  - worst_radius: 25.0
  - worst_texture: 30.0
  - worst_perimeter: 160.0
  - worst_area: 2000.0
  - worst_smoothness: 0.2
  - worst_compactness: 0.3
  - worst_concavity: 0.25
  - worst_concave_points: 0.15
  - worst_symmetry: 0.3
  - worst_fractal_dimension: 0.08
---

# Breast Cancer Wisconsin (Diagnostic) RandomForestClassifier

This is a model card for the RandomForestClassifier trained on the Breast Cancer Wisconsin (Diagnostic).
