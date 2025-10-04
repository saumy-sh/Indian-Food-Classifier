from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from sklearn.preprocessing import LabelEncoder
import torchvision.models as models
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

label_encoder = LabelEncoder()
labels = [
 'lassi', 'dal_tadka', 'malapua', 'poornalu', 'chana_masala',
 'sohan_papdi', 'shankarpali', 'shrikhand', 'aloo_gobi', 'phirni',
 'rabri', 'poha', 'bhatura', 'double_ka_meetha', 'dal_makhani',
 'lyangcha', 'kalakand', 'pootharekulu', 'sandesh', 'aloo_tikki',
 'unni_appam', 'kachori', 'chicken_tikka_masala', 'sutar_feni',
 'paneer_butter_masala', 'biryani', 'doodhpak', 'mysore_pak', 'chikki',
 'gavvalu', 'dharwad_pedha', 'jalebi', 'misi_roti', 'karela_bharta',
 'bandar_laddu', 'daal_baati_churma', 'dum_aloo', 'chak_hao_kheer',
 'sheer_korma', 'bhindi_masala', 'ghevar', 'misti_doi', 'aloo_matar',
 'ras_malai', 'butter_chicken', 'imarti', 'chicken_tikka',
 'makki_di_roti_sarson_da_saag', 'kadai_paneer', 'qubani_ka_meetha',
 'gajar_ka_halwa', 'maach_jhol', 'sohan_halwa', 'navrattan_korma',
 'gulab_jamun', 'litti_chokha', 'basundi', 'boondi', 'chicken_razala',
 'palak_paneer', 'naan', 'rasgulla', 'kofta', 'sheera', 'modak',
 'kakinada_khaja', 'ariselu', 'daal_puri', 'kadhi_pakoda', 'chapati',
 'aloo_shimla_mirch', 'aloo_methi'
]

label_encoder.fit(labels)

class TransferNN(nn.Module):
    def __init__(self,n_features=None):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # backbone.load_state_dict(torch.load('pre-trained-weights.pth'))
        in_features = backbone.fc.in_features
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) # * is the unpacking operator
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features,256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,n_features)
        )
        # freezing resnet's weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x



model = TransferNN(n_features=72).to(device)
model.load_state_dict(torch.load('fine-tuned-weights.pth'),strict=False)


test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])



# testing
dir_path = "indian_food_test"
images = os.listdir(dir_path)
img_paths = []
predictions = []
for img_path in images:
    img_path = os.path.join(dir_path,img_path)
    image = Image.open(img_path).convert('RGB')      
    input_tensor = test_transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)       
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_class_idx = torch.argmax(output, dim=1).item()
        print(img_path)
        pred = label_encoder.inverse_transform([pred_class_idx])[0]
        print(pred)
        img_paths.append(img_path)
        predictions.append(pred)


def predict(img):
    model.to(device)
    input_tensor = test_transform(img)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_class_idx = torch.argmax(output, dim=1).item()
        print(img_path)
        predicted_class = label_encoder.inverse_transform([pred_class_idx])[0]
    return predicted_class