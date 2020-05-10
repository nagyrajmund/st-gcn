# TODO: this should be added to every runnable script to make imports work
# Eventually we can use python
import sys
sys.path.append('../')

from data.datasets import KTHDataset
from network import stgcn
from torch import optim, nn, utils, autograd
from torch.utils.data import DataLoader
from pathlib import Path
from data import util

def train_network(config):
    """
    Initialise dataset and network.

    Parameters:
        config:  map containing relevant parameters
    """

    # TODO tidy config
    dataset = KTHDataset(config['metadata_file'], config['dataset_dir'], use_confidence_scores=False)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=config['sampler'],
                            collate_fn=util.loopy_pad_collate_fn)
    model = stgcn.STGCN(config['C_in'], config['gamma'], config['nr_classes'], edge_importance=config['edge_importance_weighting'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    autograd.set_detect_anomaly(True) # enable anomaly detection TODO @amrita remove for debugging purposes only

    # train
    for epoch in range(config['n_epochs']):

        for batch_idx, (data, label) in enumerate(dataloader):
            if batch_idx == len(dataset) // 2:
                break

            data, label = data.to(config['device']), label.to(config['device'])  # Move to GPU

            optimizer.zero_grad() # pytorch accumulates gradients on every call to loss.backward() so need to 0 gradients to get correct parameter update
            output = model.forward(data.double())

            loss = criterion(output, label)
            # TODO @amrita add loss_train, loss_val
            # TODO @amrita need to double check this is okay, needed for when edge importance weighting is used
            if batch_idx == 0:
                loss.backward(retain_graph = True)
            else:
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
        'batch_size': 32,
        'n_epochs': 20,
        'nr_classes': 6,
        'device': None,
        'sampler': None,  # Sampler can handle randomization etc.
        'C_in': 2,  # number of input channels
        'gamma': 9,  # temporal convolution kernel size
        'edge_importance_weighting': True  # whether to use edge importance weighting
    }

    train_network(config)





