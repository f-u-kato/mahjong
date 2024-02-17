import torch 
import torchvision.models as models
import os
import torch.nn as nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
import random
import shutil
import pandas as pd

def resnet_eval(img):
    # model=models.resnet50()
    # model.fc = torch.nn.Linear(model.fc.in_features, 2)
    # model.load_state_dict(torch.load('./weights/resnet/model_weight_20.pth'))
    model=torch.load('./weights/resnet18/model_10.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    image_size = 224
    mean = (0., 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    input = transform(img).unsqueeze(dim=0).to(device)
    outputs = model(input)
    # 確率の最も高いクラスを予測ラベルとする。
    class_id = int(outputs.argmax(dim=1)[0])
    return class_id

def plot_history(history):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    # 損失の推移
    ax1.set_title("Loss")
    ax1.plot(history["epoch"], history["train_loss"], label="train")
    ax1.plot(history["epoch"], history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # 精度の推移
    ax2.set_title("Accuracy")
    ax2.plot(history["epoch"], history["train_accuracy"], label="train")
    ax2.plot(history["epoch"], history["val_accuracy"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.show()


def resnet_train(name='riichi',class_num=2):
    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    save_path=f'./weights/{name}_resnet18'
    os.makedirs(save_path,exist_ok=True)
    image_size = (224,224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    os.makedirs(save_path,exist_ok=True)
    train_image_dir = f'./{name}/train'
    val_image_dir = f'./{name}/val'
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    train_dataset = torchvision.datasets.ImageFolder(root=train_image_dir, transform=data_transform['train'])
    val_dataset = torchvision.datasets.ImageFolder(root=val_image_dir, transform=data_transform['val'])
    batch_size = 32
    train_dataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataLoader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    dataloaders={'train':train_dataLoader,'val':val_dataLoader}

    # 出力層の出力数を ImageNet の 1000 からこのデータセットのクラス数である 2 に置き換える。
    model.fc = nn.Linear(model.fc.in_features, class_num)    # モデルを計算するデバイスに転送する。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device==torch.device("cpu"):
        print("CPUが使用されている")
        return
    model_ft = model.to(device)

    # 損失関数を作成する。
    criterion = nn.CrossEntropyLoss()

    # 最適化手法を選択する。
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    n_epochs = 10  # エポック数
    history= train(
        model_ft, criterion, optimizer, scheduler, dataloaders, device, n_epochs
    )
    torch.save(model.state_dict(),os.path.join(save_path,f'model_weight_{n_epochs}.pth'))
    torch.save(model, os.path.join(save_path,f'model_{n_epochs}.pth'))

    plot_history(history)

def train_on_epoch(model, criterion, optimizer, scheduler, dataloaders, device):
    info = {}
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # モデルを学習モードに設定
        else:
            model.eval()  # モデルを推論モードに設定

        total_loss = 0
        total_correct = 0
        for inputs, labels in dataloaders[phase]:
            # データ及びラベルを計算を実行するデバイスに転送する。
            inputs, labels = inputs.to(device), labels.to(device)
            

            # 学習時は勾配を計算するため、set_grad_enabled(True) で中間層の出力を記録するように設定する。
            with torch.set_grad_enabled(phase == "train"):
                # 順伝搬
                outputs = model(inputs)
                # 確率の最も高いクラスを予測ラベルとする
                preds = outputs.argmax(dim=1)

                # 損失関数の値を計算
                loss = criterion(outputs, labels)

                if phase == "train":
                    # 逆伝搬
                    optimizer.zero_grad()
                    loss.backward()

                    # パラメータを更新
                    optimizer.step()

            # この反復の損失及び正答数を加算
            total_loss += float(loss)
            total_correct += int((preds == labels).sum())

        if phase == "train":
            # 学習率を調整
            scheduler.step()

        # 損失関数の値の平均及び精度を計算
        info[f"{phase}_loss"] = total_loss / len(dataloaders[phase].dataset)
        info[f"{phase}_accuracy"] = total_correct / len(dataloaders[phase].dataset)

    return info

def train(model, criterion, optimizer, scheduler, dataloaders, device, n_epochs):
    history = []
    for epoch in range(n_epochs):
        info = train_on_epoch(
            model, criterion, optimizer, scheduler, dataloaders, device
        )
        info["epoch"] = epoch + 1
        history.append(info)

        print(
            f"epoch {info['epoch']:<2} "
            f"[train] loss: {info['train_loss']:.6f}, accuracy: {info['train_accuracy']:.0%} "
            f"[test] loss: {info['val_loss']:.6f}, accuracy: {info['val_accuracy']:.0%}"
        )
    history = pd.DataFrame(history)

    return history


def trigger_test():
    result=[]
    for i in range(2):
        for j in glob.glob(f'./trigger/test/{i}/*.png'):
            im=cv2.imread(j)
            c,_,_=eval.win_eval(im)
            
            if len(c)==1 and c[0]==37:
                c=0
            else:
                c=1
            new=eval.trigger_eval(im)
            result.append([i,c,new])
    df=pd.DataFrame(result)
    df.to_csv('./result_trigger.csv')

def train_val_div(source_folder,train_ratio=0.8):
    class_folders = os.listdir(source_folder)
    if 'train' in class_folders:
        print('Already divided')
        return
    for folder in ['train', 'val']:
        for class_folder in class_folders:
            os.makedirs(f'./{source_folder}/{folder}/{class_folder}', exist_ok=True)
    
    for class_folder in class_folders:
        image_files = glob.glob(f'{source_folder}/{class_folder}/*.png')
        random.shuffle(image_files)
        split_index = int(len(image_files) * train_ratio)
        train_files = image_files[:split_index]
        test_files = image_files[split_index:]

        for file in train_files:
            shutil.copy(file, f'./{source_folder}/train/{class_folder}')
        for file in test_files:
            shutil.copy(file, f'./{source_folder}/val/{class_folder}')

if __name__ == '__main__':
    train_val_div('./save_riichi',0.8)
    resnet_train('save_riichi',class_num=2)
    # resnet_train('trigger')
    # trigger_test()
    # img_path=r"F:\PBL\yolact\trriger\val\0\970.png"
    # img_path=r"F:\PBL\yolact\trriger\val\1\3600.png"
    # result=resnet_eval(Image.open(img_path))
    # print(result)