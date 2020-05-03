import numpy as npn
import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from torchvision import transforms, utils

from KeypointsDataset import KeypointsDataset
from KeypointsDataset import Rescale

from model import Net

from tqdm import tqdm

from random import randint

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    training_loss = 0
    correct = 0
    dataset_size = len(train_loader)*BATCH_SIZE

    for batch in tqdm(train_loader):
        data, target = batch['image'], batch['ground_truth']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)

        loss = criterion(output, target)
        training_loss += loss.item()

        correct += output.isclose(target.view_as(output)).sum().item()

        loss.backward()
        optimizer.step()

    accuracy = correct / dataset_size
    training_loss /= dataset_size

    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(training_loss, correct, dataset_size,
        100. * accuracy))

    return accuracy, training_loss

def evaluate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    dataset_size = len(val_loader)*BATCH_SIZE

    with torch.no_grad():
        for batch in tqdm(val_loader):
            data, target = batch['image'], batch['ground_truth']
            data, target = data.to(device), target.to(device)
            output = model(data).to(device)
            val_loss += criterion(output, target).item()  # sum up batch loss

            correct += output.isclose(target.view_as(output)).sum().item()
        accuracy = correct / dataset_size

    val_loss /= dataset_size
    accuracy = correct / dataset_size

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(val_loss, correct, dataset_size,
        100. * accuracy))

    return accuracy, val_loss

def train_model(train_loader, val_loader, test_loader, model, optimizer, criterion, num_epochs, device):

    best_test_accuracy = 0
    best_test_loss= 10
    best_epoch = -1

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

        # bestweights = None

        # if test_accuracy > best_test_accuracy:
        #     bestweights = model.state_dict()
        #     best_test_accuracy = test_accuracy
        #     best_epoch = epoch
        #     print('current best', test_accuracy, 'at epoch ', best_epoch)

        if test_loss < best_test_loss:
            bestweights = model.state_dict()
            best_test_loss = test_loss
            best_epoch = epoch
            print('current best loss', test_loss, 'at epoch ', best_epoch)

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


def draw_mesh(model, dataset, idx):
    data = dataset[idx]
    img_name, img = data['image_name'], data['image']

    bestweights = torch.load("bestweights.pt")
    model.load_state_dict(bestweights)

    output = model(img)
    print(img_name, "\n", output)


def main():
    train_dataset = KeypointsDataset(csv_file='ytb/training_frames_keypoints.csv',
                                    root_dir='ytb/training/',
                                    transform=Rescale((64, 64)))

    test_dataset = KeypointsDataset(csv_file='ytb/test_frames_keypoints.csv',
                                    root_dir='ytb/test/',
                                    transform=Rescale((64, 64)))

    # torch.save(train_dataset, "train_dataset.pt")
    # torch.save(test_dataset, "test_dataset.pt")

    # train_dataset = torch.load('train_dataset.pt', map_location=lambda storage, loc: storage)
    # test_dataset = torch.load('test_dataset.pt', map_location=lambda storage, loc: storage)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Net()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    best_epoch, best_perform_accuracy, bestweights = train_model(train_loader=train_loader,
                                                                 val_loader=test_loader,
                                                                 test_loader=test_loader,
                                                                 model=model,
                                                                 optimizer=optimizer,
                                                                 criterion=criterion,
                                                                 num_epochs=5,
                                                                 device=device)
    torch.save(bestweights, "bestweights.pt")
    print("Best epoch is: ", best_epoch)

    idx = randint(0, len(train_dataset)-1)
    draw_mesh(model, train_dataset, idx)

if __name__ == "__main__":
    BATCH_SIZE = 64
    main()
