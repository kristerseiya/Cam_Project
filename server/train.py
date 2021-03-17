
import config
import torch.nn.functional as F
import numpy as np
import torch

def train(model, optimizer, max_epoch, train_loader, val_loader=None, checkpoint_dir=None, max_tolerance=10):

    best_loss = 99999.
    tolerance = 0

    log = np.zeros([max_epoch, 4], dtype=np.float)
    n_train_batch = len(train_loader)
    if val_loader is not None:
        n_val_batch = len(val_loader)

    for e in range(max_epoch):
        total_loss = 0.
        n_correct = 0.
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            output = model(images)
            output = output.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(output, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predict = torch.zeros_like(labels, requires_grad=False)
            predict[output > 0.5] = 1
            n_correct += (predict == labels).sum().item() / float(output.size(0))

        log[e, 0] = total_loss / float(n_train_batch)
        log[e, 1] = n_correct / float(n_train_batch)

        print('Batch #{:d}'.format(e+1))
        print('Train Loss: {:.3f}'.format(log[e, 0]))
        print('Train Accs: {:.3f}'.format(log[e, 1]))

        if val_loader is not None:
            with torch.no_grad():
                total_loss = 0.
                n_correct = 0.
                model.eval()
                for images, labels in val_loader:
                    images = images.to(config.DEVICE)
                    labels = labels.to(config.DEVICE)
                    output = model(images)
                    output = output.squeeze(-1)
                    total_loss += F.binary_cross_entropy_with_logits(output, labels.float()).item()
                    predict = torch.zeros_like(labels)
                    predict[output > 0.5] = 1
                    n_correct += (predict == labels).sum().item() / float(output.size(0))

            log[e, 2] = total_loss / float(n_val_batch)
            log[e, 3] = n_correct / float(n_val_batch)

            print('Val Loss: {:.3f}'.format(log[e, 2]))
            print('Val Accs: {:.3f}'.format(log[e, 3]))

            if (best_loss > log[e, 2]):
                best_loss = log[e, 2]
                torch.save(model.state_dict(), checkpoint_dir)
                print('Best Loss! Saved.')
            else:
                tolerance += 1
                if tolerance > max_tolerance:
                    return log[0:e, :]


    return log

if __name__ == '__main__':

    import argparse
    import data
    from torch.utils.data import random_split, DataLoader
    from torch.optim import Adam
    import model

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--save', type=str, required=True)
    args = parser.parse_args()

    dataset = data.HotDogNotHotDogDataset(args.data_dir+'/train', transform=config.IMAGE_TRANFORM_TRAINING)
    train_len = int(len(dataset) * config.TRAIN_VAL_SPLIT[0])
    val_len = len(dataset) - train_len
    trainset, valset = random_split(dataset, [train_len, val_len])
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=config.BATCH_SIZE, shuffle=False)

    net = model.HotDogNotHotDogClassifier().to(config.DEVICE)
    if args.weights is not None:
        net.load_state_dict(torch.load(args.weights, map_location=config.DEVICE))
    optimizer = Adam(net.parameters(), lr=1e-4)
    train(net, optimizer, args.n_epoch, trainloader, valloader, args.save, 7)

    # torch.save(net.state_dict(), args.save)
