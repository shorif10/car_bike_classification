
def predict_image(model, image_path, device="cpu"/'cuda'):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    class_names = ['bike', 'car']
    prediction = class_names[predicted.item()]
    # displaying the title
    # plt.title(prediction,
    #           fontsize='20',
    #           backgroundcolor='red',
    #           color='white')
    # plt.imshow(image)
    return prediction