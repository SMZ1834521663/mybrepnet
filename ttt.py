import numpy as np
import torch

#IOU
num_classes=8
num_labels_per_face=8
predicted_classes=[0,2,0,1]
labels=[2,2,0,2]
correct=(predicted_classes & labels)
per_class_intersections = [0.0] * num_classes
per_class_unions = [0.0] * num_classes
for i in range(num_labels_per_face):
    selected = (predicted_classes == i)
    selected_correct = (selected & correct)
    labelled = (labels == i)
    union = selected | labelled
    per_class_intersections[i] += selected_correct.sum().item()
    per_class_unions[i] += union.sum().item()
