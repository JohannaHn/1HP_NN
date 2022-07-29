"""Definition of Dataloader"""

import numpy as np
from torch import Tensor


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        def combine_batch_dicts(batch):
            """
            Combines a given batch (list of dicts) to a dict of numpy arrays
            :param batch: batch, list of dicts
                e.g. [{k1: v1, k2: v2, ...}, {k1:, v3, k2: v4, ...}, ...]
            :returns: dict of numpy arrays
                e.g. {k1: [v1, v3, ...], k2: [v2, v4, ...], ...}
            """
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
                    #print(key, value.shape)
                    #if key=="run_id":
                    #    print(value)
            return batch_dict

        def run_id_to_int(run_id):
            """
            Converts a run_id to an integer
            :param run_id: run_id
            :returns: integer
            """
            return int(run_id.split("_")[-1])

        def batch_to_numpy(batch):
            """Transform all values of the given batch dict to numpy arrays"""
            numpy_batch = {}
            for key, value in batch.items():
                numpy_batch[key] = np.array(value)
            return numpy_batch

        def batch_to_tensor(batch):
            """Transform all values of the given batch dict to tensors"""
            tensor_batch = {}
            for key, value in batch.items():
                if key=="run_id":
                    tensor_batch[key] = Tensor([run_id_to_int(id) for id in value])
                else:
                    tensor_batch[key] = Tensor(value)
                #print(value, value[0].shape)
                #print(tensor_batch[key].shape)
            return tensor_batch


        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_iterator = iter(range(len(self.dataset)))

        batch = []
        for index in index_iterator:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch_to_tensor(combine_batch_dicts(batch))
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch_to_tensor(combine_batch_dicts(batch))


    def __len__(self):
        """
        Return the number of batches in the dataset
        """
        
        length = None
        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = int(np.ceil(len(self.dataset) / self.batch_size))

        return length