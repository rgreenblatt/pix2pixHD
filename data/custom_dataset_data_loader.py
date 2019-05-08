import torch.utils.data
from data.base_data_loader import BaseDataLoader
import torch
from torch.utils.data.distributed import DistributedSampler


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class SequentialDistributedSampler(DistributedSampler):
    def __iter__(self):
        length = len(self.dataset) // self.num_replicas
        start = self.rank * length

        return iter(range(start, start + length))

    def __len__(self):
        return len(self.dataset) // self.num_replicas


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        if opt.distributed:
            sampler = SequentialDistributedSampler(self.dataset)
        else:
            sampler = None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            pin_memory=True,
            sampler=sampler)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
