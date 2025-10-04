import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)





train_dir = 'images/train'
val_dir = 'images/test'
data_dir = 'archive/food_images_dataset'


# Creating data splits
# X_train,X_val,label_train,label_val = [],[],[],[]
# image_dirs = os.listdir(train_dir)
# for image_label in image_dirs:
#     dir_path = os.path.join(train_dir,image_label)
#     images = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith(('.jpg'))]
#     img_label = [image_label]*len(images)
#     X_train += images
#     label_train += img_label
# image_dirs = os.listdir(val_dir)
# for image_label in image_dirs:
#     dir_path = os.path.join(val_dir,image_label)
#     images = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith(('.jpg'))]
#     img_label = [image_label]*len(images)
#     X_val += images
#     label_val += img_label
# train_idx = list(range(len(X_train)))
# val_idx = list(range(len(X_val)))
# random.shuffle(train_idx)
# random.shuffle(val_idx)
# X_train = np.array(X_train)
# X_val = np.array(X_val)
# label_train = np.array(label_train)
# label_val = np.array(label_val)
# X_train = X_train[train_idx]
# X_val = X_val[val_idx]
# label_train = label_train[train_idx]
# label_val = label_val[val_idx]
# train_data = {
#     'image_path':X_train,
#     'labels':label_train
# }
# val_data = {
#     'image_path':X_val,
#     'labels':label_val
# }

# train_df = pd.DataFrame(train_data)
# val_df = pd.DataFrame(val_data)
# labels = train_df['labels'].unique()
# print(labels)




X_train_paths,X_val_paths,label_train,label_val = [],[],[],[]
image_dirs = os.listdir(data_dir)
for image_label in image_dirs:
    dir_path = os.path.join(data_dir,image_label)
    images = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith(('.jpg'))]
    img_label = [image_label]*len(images)
    X_train,X_val,y_train,y_val = train_test_split(images,img_label,test_size=0.1,random_state=42,shuffle=True)
    # print(len(X_train),len(X_val),len(y_train),len(y_val))
    X_train_paths += X_train
    X_val_paths  += X_val
    label_train += y_train
    label_val  += y_val
train_idx = list(range(len(X_train_paths)))
val_idx = list(range(len(X_val_paths)))
random.shuffle(train_idx)
random.shuffle(val_idx)
X_train_paths = np.array(X_train_paths)
X_val_paths = np.array(X_val_paths)
label_train = np.array(label_train)
label_val = np.array(label_val)
X_train_paths = X_train_paths[train_idx]
X_val_paths = X_val_paths[val_idx]
label_train = label_train[train_idx]
label_val = label_val[val_idx]
train_data = {
    'image_path':X_train_paths,
    'labels':label_train
}
val_data = {
    'image_path':X_val_paths,
    'labels':label_val
}

train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
labels = train_df['labels'].unique()
# print(labels)
# train_df.to_csv('train.csv',index=False)
# val_df.to_csv('val.csv',index=False)



# Load and transform data
label_encoder = LabelEncoder()
label_encoder.fit(labels)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=45),
    # transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

class CustomImageDataset(Dataset):
    def __init__(self,dataframe,transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe["labels"])).to(device)
    def __len__(self):
        return self.dataframe.shape[0]
    def __getitem__(self,idx):
        img_path = self.dataframe.iloc[idx,0]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image).to(device)
        return image,label
    



train_dataset = CustomImageDataset(train_df,transform)
val_dataset = CustomImageDataset(val_df,transform)



# Training

# parameters
LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 30
N_FEATURES = len(labels)
#################################################


train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)


# Transfer Learning
class TransferNN(nn.Module):
    def __init__(self,n_features=N_FEATURES):
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
    





model = TransferNN().to(device)
print(*list(model.children()))
# load everything except the final linear layer
state_dict = torch.load("pre-trained-weights.pth")
# remove the classifier's last layer weights/bias so load_state_dict ignores them
del state_dict['classifier.4.weight']
del state_dict['classifier.4.bias']
model.load_state_dict(state_dict, strict=False)

summary(model,input_size=(3,224,224))



criterion = nn.CrossEntropyLoss()
optimiser = Adam(model.parameters(),lr=LR)



total_loss_train_plot = []
total_loss_val_plot = []
total_acc_train_plot = []
total_acc_val_plot = []
for epoch in range(EPOCHS):
    total_loss_train = 0
    total_loss_val = 0
    total_acc_train = 0
    total_acc_val = 0
    total_train,total_val = 0,0
    for data in train_loader:
        image,labels = data
        optimiser.zero_grad()
        predictions = model.forward(image)
        batch_loss = criterion(predictions,labels)
        total_loss_train += batch_loss.item()
        batch_loss.backward()
        acc = (torch.argmax(predictions,axis=1) == labels).sum().item()
        total_acc_train += acc
        optimiser.step()
        total_train += len(labels)
    with torch.no_grad():
        for images,labels in val_loader:
            predictions = model(images)
            val_loss = criterion(predictions,labels)
            total_loss_val += val_loss.item()
            val_acc = (torch.argmax(predictions,axis=1) == labels).sum().item()
            total_acc_val += val_acc
            total_val += len(labels)
    print("Training data:",total_train)
    print("Validation data:",total_val)
    total_loss_train_plot.append(round(total_loss_train/total_train,4))
    total_loss_val_plot.append(round(total_loss_val/total_val,4))
    total_acc_train_plot.append(round((total_acc_train/total_train)*100,4))
    total_acc_val_plot.append(round((total_acc_val/total_val)*100,4))
    print(f"Epoch: {epoch+1}")
    print(f"Training Loss:{total_loss_train_plot[epoch]} | Training Accuracy:{total_acc_train_plot[epoch]} | Validation Loss:{total_loss_val_plot[epoch]} | Validation Accuracy:{total_acc_val_plot[epoch]}")
    



# Plotting

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
ax[0].plot(total_loss_train_plot,label="Training Loss")
ax[0].plot(total_loss_val_plot,label="Validation Loss")
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_ylim([0,0.5])
ax[0].set_title("Loss vs Epoch")
ax[0].legend()

ax[1].plot(total_acc_train_plot,label="Training Accuracy")
ax[1].plot(total_acc_val_plot,label="Validation Accuracy")
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_ylim([0,100])
ax[1].set_title("Accuracy vs Epoch")
ax[1].legend()
plt.savefig('loss-accuracy.png')


#  Inference

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])



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

# writing to txt file
with open('test.txt','w',encoding='utf-8') as f:
    for i in range(len(img_paths)):
        f.write(img_paths[i])
        f.write('\n')
        f.write(predictions[i])
        f.write('################')


torch.save(model.state_dict(),'fine-tuned-weights.pth')









