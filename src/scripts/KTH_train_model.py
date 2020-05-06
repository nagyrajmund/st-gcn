from src.data.datasets import KTHDataset
from network import gcn
from torch import optim, nn, utils
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # this should be read from a file or cmd line
    config = 
    {
        dataset_dir   : Path(__file__).parent / '../../datasets/KTH_Action_Dataset/',
        metadata_file : dataset_dir / 'metadata.csv',
        batch_size    : 32,
        n_epochs      : 10,
        device        : None,
        sampler       : None
    }

    # Maybe pass the dictionary instead of individual params?
    dataset    = KTHDataset(config.metadata_csv_path, config.numpy_data_folder)
    dataloader = DataLoader(dataset, config.batch_size, sampler=config.sampler) # Sampler can handle randomization etc.
    model      = gcn.model() # the entire model, not the layer - not yet implemented
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters())

    for epoch_idx in range(config.n_epochs):
        model.train() # Tell the model that we are training it (this affects the behaviour of dropout, batchnorm etc.)
        
        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(config.device), label.to(config.device) # Move to GPU
            optimizer.zero_grad()
            output = model(data) # Forward pass
            loss = criterion(output, label)
            loss.backward() # Backward pass
            optimizer.step() # Update the weights

            if batch_idx == config.batch_size-1:
                break

        
            

    