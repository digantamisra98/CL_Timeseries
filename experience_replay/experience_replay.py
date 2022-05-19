import abc
import torch
from torch import nn
from torch.nn import functional as F
import copy
import numpy as np


# There might be some dimension errors which can be fixed based on the input. The general idea is that you compute the feature distribution mean of a past window and compare it to the present window and if they're close, keep that past window in an exemplar set to replay.


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.
    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = (
            []
        )  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True
        hidden_sizes = [12,12] # Change Accordingly
        self.net = nn.Sequential()
        sequence_length = 58 # Change Accordingly
        self.net.add_module("input", nn.Linear(sequence_length, hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            #  Each layer will divide the size of feature map by 2
            self.net.add_module(
                "linear%d" % i,
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
            )
            self.net.add_module("relu%d" % i, nn.ReLU(True))
          # settings
        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def feature_extractor(self, batch_x):

      return self.net(batch_x)
        

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        """Construct set of [n] exemplars from [dataset] using 'herding'.
        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order."""

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding:
            # compute features for each example in [dataset]
            first_entry = True

            # dataloader = define your dataloader

            # image_batch is just the batch of timeseries
            for i in range(0, len(dataset), 32): # Change Accordingly
                image_batch = torch.tensor(dataset[i:i+32]).type(torch.float32) # Change Accordingly
                # print(image_batch.shape)
                image_batch = image_batch.to(self._device())
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            exemplar_features = torch.zeros_like(features[: min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)

                exemplar_set.append(dataset[index_selected])
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 10000
        else:
            indeces_selected = np.random.choice(
                n_max, size=min(n, n_max), replace=False
            )
            for k in indeces_selected:
                exemplar_set.append(dataset[k])

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))

        # set mode of model back
        print(np.array(self.exemplar_sets).shape)
        self.train(mode=mode)
