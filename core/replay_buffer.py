import random
import torch


class ReplayBuffer(object):
    """ TODO: use torch.roll instead rather than maintain adding here or there.
    """

    def __init__(self,
                 max_size,
                 tensor_sizes,
                 dtypes=None,
                 device=torch.device("cpu")
                 ):
        self.device = device

        self.max_size = max_size
        self.storage = [None] * max_size
        self.size = 0
        self.add_here = 0

        self.tensor_sizes = tensor_sizes
        self.tensors = []
        if dtypes is None:
            dtypes = [torch.float32] * len(tensor_sizes)

        for tensor_idx, tensor_size in enumerate(tensor_sizes):
            self.tensors.append(torch.zeros(
                (max_size, *tensor_size), dtype=dtypes[tensor_idx],
                device=self.device))

    def add(self, data):
        assert len(data) == len(self.tensors)

        for field_id in range(len(data)):
            self.tensors[field_id][self.add_here] = data[field_id]

        self.add_here = (self.add_here + 1) % self.max_size
        self.size = min([self.size + 1, self.max_size])

    def batch_add(self, data):
        assert len(data) == len(self.tensors)
        size_batch = data[0].size()[0]
        assert all([z.size()[0] == size_batch for z in data])
        assert size_batch < self.max_size, 'Batch size larger than buffer size! %d' % (
            size_batch, )

        imm_size = min(self.max_size - self.add_here, size_batch)
        overflow_size = size_batch - imm_size

        for field_id in range(len(data)):
            self.tensors[field_id][self.add_here: self.add_here +
                                   imm_size] = data[field_id][:imm_size]
            self.tensors[field_id][0:overflow_size] = data[field_id][imm_size:]

        self.add_here = (self.add_here + size_batch) % self.max_size
        self.size = min([self.size + size_batch, self.max_size])

    def sample(self, sample_size):
        # Need to select indices randomly
        assert sample_size <= self.size, "%s %s" % (sample_size, self.size)

        I = random.sample(range(self.size), sample_size)

        return self.__getitem__(I)

    def shuffle(self):
        idx = torch.randperm(self.size)
        for field_id in range(len(self.tensors)):
            self.tensors[field_id][:self.size] = self.tensors[field_id][idx]

    def __getitem__(self, indices):
        ret = []
        for field_id in range(len(self.tensors)):
            ret.append(self.tensors[field_id][indices])

        return tuple(ret)

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0
        self.add_here = 0


class PrioritizedReplayBuffer(object):
    """ TODO: use torch.roll instead rather than maintain adding here or there.
    TODO: subclass with standard replay buffer.
    """

    def __init__(self,
                 max_size,
                 tensor_sizes,
                 dtypes=None,
                 store_device=torch.device('cpu'),
                 out_device=torch.device("cpu")
                 ):
        self.device = store_device
        self.out_device = out_device

        self.max_size = max_size
        self.storage = [None] * max_size
        self.size = 0
        self.add_here = 0

        self.tensor_sizes = tensor_sizes
        self.tensors = []
        if dtypes is None:
            dtypes = [torch.float32] * len(tensor_sizes)

        for tensor_idx, tensor_size in enumerate(tensor_sizes):
            self.tensors.append(torch.zeros(
                (max_size, *tensor_size), dtype=dtypes[tensor_idx],
                device=self.device))

        self.priorities = torch.zeros(max_size, device=self.device)

    def add(self, data):
        assert len(data) == len(self.tensors)

        for field_id in range(len(data)):
            self.tensors[field_id][self.add_here] = data[field_id]

        self.add_here = (self.add_here + 1) % self.max_size
        self.size = min([self.size + 1, self.max_size])

    def batch_add(self, data):
        assert len(data) == len(self.tensors)
        size_batch = data[0].size()[0]
        assert all([z.size()[0] == size_batch for z in data])
        assert size_batch < self.max_size, 'Batch size larger than buffer size! %d' % (
            size_batch, )

        imm_size = min(self.max_size - self.add_here, size_batch)
        overflow_size = size_batch - imm_size

        for field_id in range(len(data)):
            self.tensors[field_id][self.add_here: self.add_here +
                                   imm_size] = data[field_id][:imm_size]
            self.tensors[field_id][0:overflow_size] = data[field_id][imm_size:]

        self.priorities[self.add_here: self.add_here + imm_size] = 1.0
        self.priorities[0:overflow_size] = 1.0

        self.add_here = (self.add_here + size_batch) % self.max_size
        self.size = min([self.size + size_batch, self.max_size])

    def sample(self, sample_size, method='uniform'):
        # Need to select indices randomly
        assert sample_size <= self.size, "%s %s" % (sample_size, self.size)

        I = list(torch.utils.data.WeightedRandomSampler(
            self.priorities, sample_size, replacement=True))

        return self.__getitem__(I), I

    def update_priorities(self, sample_idx, values: torch.Tensor):
        self.priorities[sample_idx] = values

    def shuffle(self):
        idx = torch.randperm(self.size)
        for field_id in range(len(self.tensors)):
            self.tensors[field_id][:self.size] = self.tensors[field_id][idx]

    def __getitem__(self, indices):
        ret = []
        for field_id in range(len(self.tensors)):
            ret.append(self.tensors[field_id][indices].to(self.out_device))

        return tuple(ret)

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0
        self.add_here = 0


if __name__ == '__main__':
    buffer = ReplayBuffer(1000, [(2, 3), (4,)])

    add1, add2 = torch.ones((2, 3)), torch.ones(4)
    buffer.add((add1, add2))

    add1, add2 = torch.ones((2, 3))*2, torch.ones(4)*2
    buffer.add((add1, add2))
    add1, add2 = torch.ones((2, 3))*3, torch.ones(4)*3
    buffer.add((add1, add2))

    samples = buffer.sample(1)
    print(samples)
