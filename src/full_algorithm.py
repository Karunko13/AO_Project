import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import datasets
import os
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageFile
import torch.optim as optim

ImageFile.LOAD_TRUNCATED_IMAGES = True


# data augmentation, losowe transformacje aby poszerzyc baze
trans_for_learn = transforms.Compose([transforms.RandomRotation(10),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])


trans_for_testing = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

# załadowanie zdjec z odpowiedniego folderu i  podział na train,test,valid
data_dir = r"" # <- sciezka do folderu z baza danych
train_directory = os.path.join(data_dir, 'train/')
valid_directory = os.path.join(data_dir, 'valid/')
test_directory = os.path.join(data_dir, 'test/')

train_data = datasets.ImageFolder(train_directory, transform=trans_for_learn)
valid_data = datasets.ImageFolder(valid_directory, transform=trans_for_testing)
test_data = datasets.ImageFolder(test_directory, transform=trans_for_testing)

# parametry potrzebne do funkcji DataLoader
num_workers = 0
batch_size = 20

load_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         num_workers=num_workers, shuffle=True)
load_valid = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                         num_workers=num_workers, shuffle=True)
load_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                        num_workers=num_workers, shuffle=True)

image_dirs = {'train': load_train,
              'valid': load_valid, 'test': load_test}

breeds_number = 133

# model bez transfer learningu


class ModelFromScratch(nn.Module):
    def __init__(self):
        super(ModelFromScratch, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.25)

        self.fc2 = nn.Linear(1024, breeds_number)


first_model = ModelFromScratch()
number_of_epochs = 15


def train_model(number_epochs, dir_to_load, chosen_model, optimizer, loss_fnc,  save_path):
    print("training")
    for epoch in range(1, number_epochs + 1):

        loss_during_train = 0.0
        loss_during_val = 0.0
        print("training")

        chosen_model.train()
        for batch_number, (data, target) in enumerate(dir_to_load['train']):

            optimizer.zero_grad()
            output = chosen_model(data)
            loss = loss_fnc(output, target)
            loss.backward()
            optimizer.step()

            loss_during_train = loss_during_train + \
                ((1 / (batch_number + 1)) * (loss.data - loss_during_train))

        chosen_model.eval()
        for batch_number, (data, target) in enumerate(dir_to_load['valid']):

            output = chosen_model(data)

            loss = loss_fnc(output, target)

            loss_during_val = loss_during_val + \
                ((1 / (batch_number + 1)) * (loss.data - loss_during_val))

        torch.save(chosen_model.state_dict(), save_path)

    return chosen_model


# zakomentowane aby ten model się nie uruchamiał

# n_epochs = 15
# optimizer_scratch = optim.Adagrad(first_model.parameters(), lr=0.01)
# loss_function = nn.CrossEntropyLoss()
# first_model = train_model(n_epochs, image_dirs, first_model, optimizer_scratch,loss_function,  'learned_model_from_transfer.pt')
#
#
# first_model.load_state_dict(torch.load('learned_model_from_transfer.pt'))


def test_function(loaders, chosen_model, loss_function):

    loss_during_testing = 0.
    correct_predictions = 0.
    total_predictions = 0.

    chosen_model.eval()
    for batch_number, (data, target) in enumerate(loaders['test']):

        output = chosen_model(data)

        loss = loss_function(output, target)

        loss_during_testing = loss_during_testing + ((1 / (batch_number + 1))
                                                     * (loss.data - loss_during_testing))
        pred = output.data.max(1, keepdim=True)[1]

        correct_predictions += np.sum(np.squeeze(
            pred.eq(target.data.view_as(pred))).cpu().numpy())
        total_predictions += data.size(0)

    print('Test Loss: {:.6f}\n'.format(loss_during_testing))


#test_function(load_dir, model_scratch, loss_function)

loaders_transfer = image_dirs
model_2_transfer = models.densenet161(pretrained=True)

# "zamrozenie" czesci modelu i trenowanie reszty
for temp in model_2_transfer.features.parameters():
    temp.requires_grad = False

number_of_inputs = model_2_transfer.classifier.in_features
# wyodrebnienie ostatniej warstwy
last_layer = nn.Linear(number_of_inputs, breeds_number)
model_2_transfer.classifier = last_layer

loss_function = nn.CrossEntropyLoss()
optimizer_function = optim.Adagrad(
    model_2_transfer.classifier.parameters(), lr=0.01)

number_of_epochs = 15
model_2_transfer = train_model(number_of_epochs, loaders_transfer, model_2_transfer,
                               optimizer_function, loss_function, 'learned_model_from_transfer.pt')


model_2_transfer.load_state_dict(torch.load('learned_model_from_transfer.pt'))

class_names = [item[4:].replace("_", " ")
               for item in image_dirs['train'].classes]


def predict_function(img_path):

    model_2_transfer.eval()
    # tranformacja("unormowanie") wczytanego do programu obrazka
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform(img).float()
    img = img.unsqueeze(0)

    # predykcja
    predictions = model_2_transfer(img)
    _, index = predictions.max(1)

    return class_names[index.item()]


image_path = r""  # <- ścieżka do zdjęcia na którym testujemy
print(predict_function(image_path))
