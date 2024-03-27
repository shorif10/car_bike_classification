from predict import predict_image
import torch
import tkinter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
result = predict_image(model="pretrained_models/model.pth", image_path="data/test/bike/Bike (260).jpeg", device=device)