from omegaconf import DictConfig

from modis.utils.config import read_config
from modis.utils.data import get_dataloaders, SemiSupervisedDataset
from modis.train import train_model

from dataset import get_datasets

@read_config(config_file='./config/intersim_2_delta.yaml')
def main(config: DictConfig):
    train_datasets = get_datasets(
        dataset_name = 'intersim_2_delta',
        pairing = 'unpaired',
        split = 'train',
        return_sample_ids = False
    )

    # Simulate the semi-supervised setting by adjust the dataset labels
    if config.training_mode == 'semisupervised':
        # Fully unsupervised dataset, config: intersim_2_delta.yaml
        # train_datasets = [SemiSupervisedDataset(dataset, labeled_ratio=0) for dataset in train_datasets]  # Adjust to desired settings

        # Fully labeled dataset
        # pass

        # Make a partially labeled-dataset
        # train_datasets = [SemiSupervisedDataset(dataset, labeled_ratio=0.05) for dataset in train_datasets]  # Adjust to desired settings
        train_datasets = [SemiSupervisedDataset(dataset, class_samples=10, remove_unlabeled=False) for dataset in train_datasets]
        # train_datasets = [SemiSupervisedDataset(dataset, class_samples=[None, None, None, 2, 2]) for dataset in train_datasets]

    train_dataloaders = get_dataloaders(train_datasets, batch_size=config.batch_size, drop_last=True, shuffle=True)

    train_model(train_dataloaders, config)

if __name__ == "__main__":
    main()
