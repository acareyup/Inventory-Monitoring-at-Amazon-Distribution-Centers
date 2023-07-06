import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os
from smdebug import modes
import smdebug.pytorch as smd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import sys

logger.addHandler(logging.StreamHandler(sys.stdout))
from tqdm import tqdm


def test(Model, test_loader, criterion, hook):
    logger.debug(f"Test the Model on test split")
    Model.eval()
    if hook is not None:
        hook.set_mode(modes.EVAL)
    loss = 0
    corrects = 0

    for i, X in tqdm(test_loader):
        output = Model(i)
        loss = criterion(output, X)
        _, pred = torch.max(output, 1)
        loss += loss.item() * i.size(0)
        corrects += torch.sum(pred == X.data)

    t_loss = loss / len(test_loader.dataset)
    t_acc = corrects.double() / len(test_loader.dataset)

    logger.info(f"Testing Loss: {t_loss}")
    logger.info(f"Testing Accuracy: {t_acc}")


def train(Model, train_loader, validation_loader, criterion, optimizer, hook):
    epochs = 5
    best_loss = 1e6
    dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in range(epochs):
        for step in ['train', 'valid']:
            logger.info(f"Epoch: {epoch}, Phase: {step}")
            if step == 'train':
                Model.train()
                if hook is not None:
                    hook.set_mode(modes.TRAIN)
            else:
                Model.eval()
                if hook is not None:
                    hook.set_mode(modes.EVAL)
            loss = 0.0
            corrects = 0.0
            samples = 0

            t_sample_in_this_step = len(dataset[step].dataset)

            for i, X in dataset[step]:
                output = Model(i)
                loss = criterion(output, X)

                if step == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, pred = torch.max(output, 1)
                loss += loss.item() * i.size(0)
                corrects += torch.sum(pred == X.data)
                samples += len(i)

                accuracy = corrects / samples
                logger.debug(
                    "Epoch {}, Phase {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        epoch,
                        step,
                        samples,
                        t_sample_in_this_step,
                        100.0 * (float(samples) / float(t_sample_in_this_step)),
                        loss.item(),
                        corrects,
                        samples,
                        100.0 * accuracy,
                    ))

            epoch_loss = loss / samples
            epoch_acc = corrects / samples

            if step == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(step,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))

        if loss_counter == 1:
            logger.info(f"Break : increasing in loss at epoch: {epoch}")
            break
    return Model


def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5))
    return model


def create_data_loaders(data, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(os.environ['SM_CHANNEL_TRAIN'], transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(os.environ['SM_CHANNEL_TEST'], transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(os.environ['SM_CHANNEL_VALID'], transform=test_transform)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    logger.info(f'HP LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Paths: {args.data}')

    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)
    Model = net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Model.fc.parameters(), lr=args.learning_rate)

    logger.info("Creating hook")
    try:
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(Model)
        hook.register_loss(criterion)
        logger.info("Hook created")
    except:
        hook = None
        logger.info("Hook not available.")

    logger.info("Starting Model Training")
    Model = train(Model, train_loader, validation_loader, criterion, optimizer, hook)

    logger.info("Testing Model")
    test(Model, test_loader, criterion, hook)

    logger.info("Saving Model")
    torch.save(Model.cpu().state_dict(), os.path.join(args.model_dir, "Model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()
    logger.debug(f'Arguments: {args}')

    main(args)
