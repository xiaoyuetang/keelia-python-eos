import numpy as npn
import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from torchvision import transforms, utils

from KeypointsDataset import KeypointsDataset
from KeypointsDataset import Rescale

from model import Net

from tqdm import tqdm

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    training_loss = 0
    correct = 0
    dataset_size = len(train_loader)
    for batch in tqdm(train_loader):
        data, target = batch['image'], batch['ground_truth']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)

        loss = criterion(output, target)
        training_loss += loss.item()

        correct += output.eq(target.view_as(output)).sum().item()

        loss.backward()
        optimizer.step()

    accuracy = correct / dataset_size
    training_loss /= dataset_size

    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(training_loss, correct, data_size,
        100. * accuracy))

    return accuracy, training_loss

def evaluate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    data_size = len(val_loader)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            data, target = batch['image'], batch['ground_truth']
            data, target = data.to(device), target.to(device)
            output = model(data).to(device)
            val_loss += criterion(output, target).item()  # sum up batch loss

            correct += output.eq(target.view_as(output)).sum().item()
        accuracy = correct / data_size

    val_loss /= data_size
    accuracy = correct / dataset_size

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(val_loss, correct, data_size,
        100. * accuracy))

    return accuracy, val_loss

def train_model(train_loader, val_loader, test_loader, model, optimizer, criterion, num_epochs, device):

    best_test_accuracy = 0
    best_epoch =-1

    # array for plotting
    training_loss_array = []
    test_loss_array = []
    val_loss_array = []
    train_accuracy_array = []
    test_accuracy_array = []
    val_accuracy_array = []

    for epoch in range(1, num_epochs+1):
        print('♥' * 60)
        print('Epoch {}/{}'.format(epoch, num_epochs))

        model.train(True)
        train_accuracy, training_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch)

        model.train(False)
        val_accuracy, val_loss = evaluate(model, device, val_loader, criterion)

        model.train(False)
        test_accuracy, test_loss = evaluate(model, device, test_loader, criterion)
        print('perform_accuracy', test_accuracy)

        training_loss_array.append(training_loss)
        test_loss_array.append(test_loss)
        val_loss_array.append(val_loss)
        train_accuracy_array.append(train_accuracy)
        test_accuracy_array.append(test_accuracy)
        val_accuracy_array.append(val_accuracy)

        if test_accuracy > best_test_accuracy:
            bestweights = model.state_dict()
            best_test_accuracy = test_accuracy
            best_epoch = epoch
            print('current best', test_accuracy, 'at epoch ', best_epoch)

    # plt.plot(training_loss_array, label='training loss')
    # plt.plot(test_loss_array, label='test loss')
    # plt.plot(val_loss_array, label='validation loss')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # plt.plot(train_accuracy_array, label='train accuracy')
    # plt.plot(test_accuracy_array, label='test accuracy')
    # plt.plot(val_accuracy_array, label='validation accuracy')
    # plt.legend(loc='upper left')
    # plt.show()
    return best_epoch, best_test_accuracy, bestweights

def main():
    train_dataset = KeypointsDataset(csv_file='ytb/training_frames_keypoints.csv',
                                    root_dir='ytb/training/',
                                    transform=Rescale((64, 64)))

    test_dataset = KeypointsDataset(csv_file='ytb/test_frames_keypoints.csv',
                                    root_dir='ytb/test/',
                                    transform=Rescale((64, 64)))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    model = Net()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    best_epoch, best_perform_accuracy, bestweights = train_model(train_loader=train_loader,
                                                                 val_loader=test_loader,
                                                                 test_loader=test_loader,
                                                                 model=model,
                                                                 optimizer=optimizer,
                                                                 criterion=criterion,
                                                                 num_epochs=5,
                                                                 device=device)

if __name__ == "__main__":
    main()