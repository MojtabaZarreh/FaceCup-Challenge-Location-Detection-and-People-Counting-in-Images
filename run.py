from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import pytorch

yolo_model = YOLO("/content/drive/MyDrive/facecup/gender_human_datection.pt")

class SimpleEfficientNet(nn.Module):
    def __init__(self, num_classes=len(class_names), pretrained=True):
        super(SimpleEfficientNet, self).__init__()

        self.efficientnet = models.efficientnet_b6(pretrained=pretrained)

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

class_names = [str(i) for i in range(488)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = SimpleEfficientNet()
classification_model.load_state_dict(torch.load("/content/drive/MyDrive/efficientnet_b6.pth", map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_folder(folder_path, output_csv):
    data = []

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if not os.path.isfile(image_path):
            continue

        male_count = 0
        female_count = 0

        # YOLO inference
        yolo_results = yolo_model(image_path)
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls)  # Class index
                if cls == 0:  # Assuming 0 represents 'male'
                    male_count += 1
                elif cls == 1:  # Assuming 1 represents 'female'
                    female_count += 1

        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = classification_model(img)
            probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()

        row = {
            'path': image_name,
            'males': male_count,
            'females': female_count
        }

        for idx, prob in enumerate(probabilities):
            row[idx] = float(f"{prob}")

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

process_folder("/content/test", "/content/drive/MyDrive/Submission-efficientnet_b6.csv")
