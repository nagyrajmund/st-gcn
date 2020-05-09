# TODO: this should be added to every runnable script to make imports work
# Eventually we can use python 
import sys
sys.path.append('../')

from data.datasets import KTHDataset
from network import stgcn
from torch import optim, nn, utils
from torch.utils.data import DataLoader
from pathlib import Path

def train_network(config):
    # initialise dataset and network
    # TODO tidy config
    dataset = KTHDataset(config['metadata_file'], config['dataset_dir'])
    dataloader = DataLoader(dataset, config['batch_size'],
                            sampler=config['sampler'])  # Sampler can handle randomization etc.
    model = stgcn.STGCN(config['C_in'], config['gamma'], config['nr_classes'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(config['n_epochs']):
        for batch_idx, (data, label, scores) in enumerate(dataloader):
            # TODO @amrita add augmentation every e epochs
            data = data[0]  # shape (1,1,T,V,C) tmp fix TODO @amrita remove
            label = label[0]
            data, label = data.to(config['device']), label.to(config['device'])  # Move to GPU

            optimizer.zero_grad() # pytorch accumulates gradients on every call to loss.backward() so need to 0 gradients to get correct parameter update
            output = model.forward(data.double())

            loss = criterion(output, label)
            # TODO @amrita add loss_train, loss_val
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            if epoch % 10 == 0:
                print('Epoch: ', epoch + 1, '\t loss: ', loss)

            if batch_idx == config['batch_size'] - 1:
                break

    # get prediction
    # TODO replace with test set
    # pred = torch.argmax(model(test_data), dim=1)



if __name__ == "__main__":
    # this should be read from a file or cmd line
    dataset_dir = '../../datasets/KTH_Action_Dataset'
    config = {
        'dataset_dir': Path(dataset_dir),
        'metadata_file': Path(dataset_dir) / 'metadata.csv',
        'batch_size': 1,  # 32 TODO get working with batch size > 1
        'n_epochs': 20,
        'nr_classes': 6,
        'device': None,
        'sampler': None,
        'C_in': 2,  # number of input channels
        'gamma': 9  # temporal convolution kernel size
    }

    train_network(config)





