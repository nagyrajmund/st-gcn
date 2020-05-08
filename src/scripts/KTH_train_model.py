from src.data.datasets import KTHDataset
from network import stgcn
from torch import optim, nn, utils
import torch
from torch.utils.data import DataLoader
from pathlib import Path

def train_network(config):
    # initialise dataset and network
    dataset = KTHDataset(config['metadata_file'], config['dataset_dir'])
    dataloader = DataLoader(dataset, config['batch_size'],
                            sampler=config['sampler'])  # Sampler can handle randomization etc.
    stgcn_ntwk = stgcn.STGCN(config['C_in'], config['gamma'])

    model = stgcn_ntwk.model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # train
    batch_size = 20
    n_batches = 10 # TODO change
    for epoch_idx in range(config['n_epochs']):
        model.train()  # Tell the model that we are training it (this affects the behaviour of dropout, batchnorm etc.)
        for batch_idx, (data, label, scores) in enumerate(dataloader):
        # for i in range(n_batches):
        #     start_index = i * batch_size
        #     end_index = (i+1) * batch_size
        #     data, labels, scores = dataset[start_index:end_index]
        #     print(data.dtype)
        #     print(data.shape)
        #     data = torch.from_numpy(data)
        #     labels = torch.tensor(labels)

            data = data[0]  # shape (1,1,T,V,C) tmp fix TODO remove
            data, label = data.to(config['device']), label.to(config['device'])  # Move to GPU


            optimizer.zero_grad()
            output = model(data)  # Forward pass
            loss = criterion(output, label)
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            if batch_idx == config.batch_size - 1:
                break


if __name__ == "__main__":
    # this should be read from a file or cmd line
    dataset_dir = '../../datasets/KTH_Action_Dataset'
    config = {
        'dataset_dir': Path(dataset_dir),
        'metadata_file': Path(dataset_dir) / 'metadata.csv',
        'batch_size': 1,  # 32 TODO get working with batch size > 1
        'n_epochs': 10,
        'device': None,
        'sampler': None,
        'C_in': 2,  # number of input channels
        'gamma': 9  # temporal convolution kernel size
    }

    train_network(config)





