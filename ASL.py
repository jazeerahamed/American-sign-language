import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image
from torchvision.transforms.v2 import Compose, ToDtype, CenterCrop, RandomEqualize
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm
from pathlib import Path
from os import listdir, scandir

plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)

def show_image(image, title=None):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis(False)
    plt.imshow(image.permute(1, 2, 0))

def plot_history(history, net_name):
    x_ticks = range(1, len(history['train']['loss']) + 1)
    
    for item in history['train'].keys():
        plt.figure(figsize=(12, 4))
        
        for prefix, color in zip(['train', 'val'], ['r', 'b']):
            plt.plot(x_ticks, history[prefix][item], c=color, alpha=0.75, linestyle='--',
                     label=prefix)
        
        plt.title('{} {}'.format(net_name, item))
        plt.xlabel('Epoch')
        plt.xticks(x_ticks)
        plt.ylabel(item)
        plt.grid()
        plt.legend()
        plt.show()

def plot_class_distribution(df):
    indices = df['sign'].unique().argsort()
    signs = df['sign'].unique()[indices]
    counts = df['sign'].value_counts().values[indices]
    
    plt.figure(figsize=(12, 4))
    plt.bar(signs, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.xticks(range(len(signs)), labels=signs)
    plt.xlim((-1, len(signs)))
    plt.ylabel('Sign')
    plt.show()

def plot_metric_values(metric_values, df, net_name):
    for item in metric_values:
        fig, ax = plt.subplots(figsize=(12, 4))
        indices = df['sign'].unique().argsort()
        signs = df['sign'].unique()[indices]
        x = range(len(signs))
        y = metric_values[item].cpu().detach().numpy()[indices]
        
        ax.axhline(y.mean(), color='r', linestyle='--', alpha=0.5)
        
        bars = ax.bar(x, y)
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    s='{:.2}'.format(bar.get_height()), ha='center', va='bottom')
        
        ax.set_xlabel('Class')
        ax.set_xticks(x, labels=signs)
        ax.set_xlim((-1, len(signs)))
        ax.set_ylim((0.6, 1))
        ax.set_title('{} {} on test'.format(net_name, item))
        plt.show()
      
workdir_path = Path('/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train')
image_height, image_width, n_channels = 196, 196, 3


data = []

with scandir(workdir_path) as entries:
    for entry in entries:
        if entry.is_dir():
            path_to_dir = workdir_path / entry.name
            
            for filename in listdir(path_to_dir):
                path_to_image = path_to_dir / filename
                
                data.append((path_to_image, entry.name.upper()))

df = pd.DataFrame(data, columns=['path_to_image', 'sign'])
df['sign'] = df['sign'].replace({'SPACE': '_', 'DEL': '-', 'NOTHING': '!'})

sign_to_class = {sign: i for i, sign in enumerate(df['sign'].unique())}
df['class'] = df['sign'].map(sign_to_class)

df


n_classes = df['class'].max() + 1

print('There are {} classes in total'.format(n_classes))

plot_class_distribution(df)

signs = np.sort(df['sign'].unique())

fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(12, 12))
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9)
for i, ax in enumerate(axs.flatten()):
    if i < len(signs):
        sign = signs[i]
        path_to_image = df[df['sign'] == signs[i]].iloc[0]['path_to_image']
        image = decode_image(path_to_image).permute(1, 2, 0)
        
        ax.imshow(image)
        ax.set_title(sign)
    ax.axis(False)


transform = Compose([
    ToDtype(dtype=torch.float32, scale=True),
    CenterCrop(size=(image_height, image_width)),
    RandomEqualize(p=1)
])


class Dataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    
    def __getitem__(self, i):
        path_to_image, _, class_label = self.df.iloc[i]
        image = decode_image(path_to_image)
        
        transformed_image = self.transform(image)
        gamma_corrected_image = transformed_image ** (1 / 2.2)
        high_contrast_image = gamma_corrected_image * gamma_corrected_image
        
        return high_contrast_image, class_label
    
    def __len__(self):
        return self.df.shape[0]

ds = Dataset(df, transform)

sample_image, sample_class = ds[0]
show_image(sample_image, 'Sample Image')
print('sample_image.shape: {}, sample_class: {}'.format(sample_image.shape, sample_class))

batch_size = 128

train_ds, val_ds, test_ds = random_split(ds, [0.8, 0.1, 0.1])

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size)
test_dl = DataLoader(test_ds, batch_size)


class Jazeerahamed(torch.nn.Sequential):
    def __init__(self, n_channels, n_classes):
        super().__init__(
            torch.nn.Conv2d(in_channels=n_channels, out_channels=96, kernel_size=11,
                            stride=4),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=6400, out_features=4096),
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.Linear(in_features=4096, out_features=n_classes)
        )


net = Jazeerahamed(n_channels=n_channels, n_classes=n_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.to(device)

summary(net, (n_channels, image_height, image_width))



def eval(net, eval_dl, prefix, criterion, metrics, device):
    eval_loss = torch.tensor(0.0).to(device)
    eval_metric_values = {
        metric: torch.zeros(metrics[metric].num_classes if metrics[metric].average is None
                            else 1).to(device) for metric in metrics
    }
    
    net.to(device)
    net.eval()
    with torch.no_grad():
        for X, y in eval_dl:
            X = X.to(device)
            y = y.to(device)
            
            preds = net(X)
            loss = criterion(preds, y)
            
            eval_loss += loss.detach().cpu() * eval_dl.batch_size
            for metric in metrics:
                eval_metric_values[metric] += metrics[metric](preds, y) * \
                    eval_dl.batch_size
    
    eval_loss /= len(eval_dl.dataset)
    for metric in metrics:
        eval_metric_values[metric] /= len(eval_dl.dataset)
    
    print('{}_loss: {:.3f}'.format(prefix, eval_loss), end='')
    for metric in metrics:
        print(', {}_{}: {:.3f}'.format(prefix, metric, eval_metric_values[metric].mean()),
              end='')
    
    return eval_loss, eval_metric_values


def train(net, train_dl, val_dl, n_epochs, criterion, metrics, device, lr):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    history = {
        'train': {'loss': []} | {metric: [] for metric in metrics},
        'val': {'loss': []} | {metric: [] for metric in metrics}
    }
    
    net.to(device)
    for epoch in range(n_epochs):
        train_loss = torch.tensor(0.0).to(device)
        train_metric_values = {
            metric: torch.tensor(0.0).to(device) for metric in metrics
        }
        
        net.train()
        for X, y in tqdm(train_dl, desc='Epoch {}/{}'.format(epoch + 1, n_epochs),
                         total=len(train_dl)):
            X = X.to(device)
            y = y.to(device)
            
            preds = net(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_loss += loss.detach().cpu() * train_dl.batch_size
                for metric in metrics:
                    train_metric_values[metric] += train_dl.batch_size * \
                        metrics[metric](preds, y)
        
        train_loss /= len(train_dl.dataset)
        for metric in metrics:
            train_metric_values[metric] /= len(train_dl.dataset)
        
        print('train_loss: {:.3f}'.format(train_loss), end=', ')
        for metric in metrics:
            print('train_{}: {:.3f}'.format(metric, train_metric_values[metric]),
                  end=', ')
        
        history['train']['loss'].append(train_loss.cpu().detach().numpy().item())
        for metric in metrics:
            history['train'][metric].append(train_metric_values[metric].cpu().detach() \
                                            .numpy().item())
            
        val_loss, val_metric_values = eval(net, val_dl, prefix='val', criterion=criterion,
                                           metrics=metrics, device=device)
        
        scheduler.step(val_loss)
        
        history['val']['loss'].append(val_loss.cpu().detach().numpy().item())
        for metric in metrics:
            history['val'][metric].append(val_metric_values[metric].cpu().detach() \
                                          .numpy().item())
    return history

criterion = torch.nn.CrossEntropyLoss()

train_metrics = {
    'accuracy': Accuracy(task='multiclass', num_classes=n_classes,
                         average='macro').to(device),
    'precision': Precision(task='multiclass', num_classes=n_classes,
                           average='macro').to(device),
    'recall': Recall(task='multiclass', num_classes=n_classes,
                     average='macro').to(device),
    'f1-score': F1Score(task='multiclass', num_classes=n_classes,
                        average='macro').to(device)
}

history = train(net, train_dl, val_dl, n_epochs=5, criterion=criterion,
                metrics=train_metrics, device=device, lr=0.0001)



plot_history(history, net_name='Jazeerahamed')


test_metrics = {
    'accuracy': Accuracy(task='multiclass', num_classes=n_classes,
                         average=None).to(device),
    'precision': Precision(task='multiclass', num_classes=n_classes,
                           average=None).to(device),
    'recall': Recall(task='multiclass', num_classes=n_classes,
                     average=None).to(device),
    'f1-score': F1Score(task='multiclass', num_classes=n_classes,
                        average=None).to(device)
}

test_metric_values = eval(net, test_dl, prefix='test', criterion=criterion,
                             metrics=test_metrics, device=device)


plot_metric_values(test_metric_values, df, net_name='Jazeerahamed')
