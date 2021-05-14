from dataset import *
from model import *
from loss import *

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn


if __name__ == '__main__':
    train_dataset = Cifar10Train('/home/josh/Data/cifar-10-python/cifar-10-batches-py/')
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=1)
    print('successfully loaded {} training images and labels'.format(len(train_dataloader.dataset)))

    valid_dataset = Cifar10Valid('/home/josh/Data/cifar-10-python/cifar-10-testbatches-py')
    valid_dataloader = DataLoader(valid_dataset, batch_size=512, shuffle=True, num_workers=1)
    print('successfully loaded {} validation images and labels'.format(len(valid_dataloader.dataset)))

    model = TransformerEncoder()
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    else:
        print('cuda is not available')

    optimizer = optim.Adam(model.parameters(), lr=3.2768e-5, weight_decay=5e-4)

    criterion = Loss()
    for epoch in range(100):
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0
        
        model.train()
        for step, item in enumerate(train_dataloader):
            def train():
                optimizer.zero_grad()
                train_data, train_label = item
                train_data = train_data.float().cuda()
                train_label = train_label.cuda()

                outputs = model(train_data)

                loss = criterion(outputs, train_label)
                loss.backward()
                return loss

            step_loss = optimizer.step(train)
            train_loss = train_loss + step_loss.item()
            # print('step : {} || step_loss : {}'w.format(step, step_loss))

        model.eval()
        for step, item in enumerate(valid_dataloader):
            def valid():
                valid_accuracy = 0
                valid_data, valid_label = item
                valid_data = valid_data.float().cuda()
                valid_label = valid_label.cuda()

                outputs = model(valid_data)

                loss = criterion(outputs, valid_label)
                outputs = torch.argmax(outputs, dim=1)
                for i in range(len(valid_label)):
                    if outputs[i].item() == valid_label[i].item():
                        valid_accuracy = valid_accuracy + 1

                return valid_accuracy, loss.item()
            _a, _l = valid()
            accuracy = accuracy + _a
            valid_loss = valid_loss + _l

        print('epoch : {} || train_loss : {} || validation acc : {} validation_loss : {}'.format(epoch, train_loss / len(train_dataloader.dataset),
                                                                                                   accuracy / len(valid_dataloader.dataset),valid_loss / len(valid_dataloader.dataset)))
        if epoch % 10 == 0:
            torch.save(model.module.state_dict(), '/home/josh/Weights/state_dict/ViT_' + repr(epoch) + '.pth')