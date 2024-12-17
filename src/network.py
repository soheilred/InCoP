import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet18_Weights, ResNet50_Weights,\
    vgg11, vgg16, alexnet, VGG11_Weights, VGG16_Weights, AlexNet_Weights,\
    GoogLeNet_Weights 

import logging
import utils
from data_loader import Data
import constants as C
from models import resnet18

# from EDGE_4_4_1 import EDGE
# from matplotlib.ticker import MaxNLocator
# plt.style.use('ggplot')
log = logging.getLogger("sampleLogger")

class Network():
    def __init__(self, device, arch, num_classes=10, pretrained="True",
                 feature_extracting=False):
        """Setup the network and adjust it to the dataset dimentions.
        Parameters
        ----------
        device: str
            "cpu" or "cuda"
        arch: str
            network architecture we want to use, "vgg16", "resnet", "alexnet"
        num_classes: int
            number of classes in the dataset
        pretrained: bool
            want the network to be pretrained or not
        feature_extracting: bool
            whether we want to only do feature extraction or network
        finetuning. This is used for adjusting the layers requires_grad
        attribute.
        """
        self.preprocess = None
        self.model = None
        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = device
        self.feature_extracting = feature_extracting
        self.input_size = 224

    def set_model(self):
        if self.arch == "vgg11":
            if self.pretrained == 1:
                self.model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.vgg11(weights=None)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        elif self.arch == "vgg16":
            if self.pretrained == 1:
                self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.vgg16(weights=None)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        elif self.arch == "resnet18":
            if self.pretrained == 1:
                self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.resnet18(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.arch == "alexnet":
            if self.pretrained == 1:
                self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.alexnet(weights=None)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        elif self.arch == "squeezenet":
            if self.pretrained == 1:
                self.model = models.squeezenet1_0(weights=Inception_V3_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.squeezenet1_0(weights=None)
            num_ftrs = 512
            self.model.classifier[1] = nn.Conv2d(num_ftrs, self.num_classes,
                                                 kernel_size=(1,1), stride=(1,1))

        elif self.arch == "densenet":
            if self.pretrained == 1:
                self.model = models.densenet121(weights=Inception_V3_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.densenet121(weights=None)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Conv2d(num_ftrs, self.num_classes)

        elif self.arch == "googlenet":
            if self.pretrained == 1:
                self.model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.inception_v3(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes)

        elif self.arch == "inception":
            if self.pretrained == 1:
                self.model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
                self.set_parameter_requires_grad()
            else:
                self.model = models.inception_v3(weights=None)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Conv2d(num_ftrs, self.num_classes)
            self.input_size = 299

        else:
            sys.exit("Wrong architecture")

        self.model = self.model.to(self.device)
        return self.model

    def set_parameter_requires_grad(self):
        if self.feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def trained_enough(self, correlation=None, grads=None, epsilon=None):
        # log.debug(f"{i} epoch extra training, accuracy: {100 * accuracy}")
        if grads:
            if epsilon is None:
                epsilon = 0.5
            opt_grad = torch.Tensor(grads[0][0])
            cur_grad = torch.Tensor(grads[-1][-1])
            log.debug(f"Grad diff {(cur_grad - opt_grad).norm().item()}")
            return (cur_grad - opt_grad).norm().item() < epsilon

        elif correlation:
            if epsilon is None:
                epsilon = 0.11
            cur_corr = torch.Tensor([elem.mean() for elem in correlation[-1]])
            opt_corr = torch.Tensor([elem.mean() for elem in correlation[0]])
            # if there are nans in correlation
            log.debug(f"Nan in correlations?: {any([torch.isnan(corr.view(-1)).any().item() for corr in correlation[-1]])}")
            if any([torch.isnan(corr.view(-1)).any().item() for corr in correlation[-1]]):
                return False

            # if the norm difference is less than a threshold
            log.debug(f"Correlation norm diff: {(cur_corr - opt_corr).norm().item()}")
            return (cur_corr - opt_corr).norm().item() < epsilon

        else:
            sys.exit("error in trained_enough")

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

        # self.fc = nn.Linear(512 * block.expansion, num_classes)


def make_grads_zero(model, mask):
    """Sets grads of fixed weights to 0.
        During training this is called to avoid storing gradients for the
        frozen weights, to prevent updating.
        This is unaffected in the shared masks since shared weights always
        have the current index unless frozen.
    """
    EPS = 1e-6
    # layer_id = 0
    # assert self.current_masks
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            param.grad.data *= mask[name[:-7]]
            # layer_id += 1

    # for module_idx, module in enumerate(self.model.shared.modules()):
    #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #         layer_mask = self.mask[module_idx]
    #         # Set grads of all weights not belonging to current dataset to 0.
    #         if module.weight.grad is not None:
    #             module.weight.grad.data[layer_mask.ne(self.task_num)] = 0
    #         if self.task_num > 0 and module.bias is not None:
    #             module.bias.grad.data.fill_(0)

    #     elif 'BatchNorm' in str(type(module)) and self.task_num > 0:
    #         # Set grads of batchnorm params to 0.
    #         module.weight.grad.data.fill_(0)
    #         module.bias.grad.data.fill_(0)

            # tensor = p.data.cpu().numpy()
            # grad_tensor = p.grad.data.cpu().numpy()
            # grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
            # p.grad.data = torch.from_numpy(grad_tensor).to(device)


def train(model, dataloader, loss_fn, optimizer, mask, epochs, device):
    model.train()
    # log.debug('Training...')
    size = len(dataloader.dataset)

    for t in range(epochs):
        # log.debug(f"Epoch {t+1}")
        correct = 0
        running_loss = 0.
        last_loss = 0.

        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            if mask is not None:
                # Set frozen param grads to 0.
                make_grads_zero(model, mask)

            optimizer.step()

            running_loss += loss.item()

            if batch % 100 == 0:
                last_loss, current = running_loss / 100, batch * len(X)
                # log.debug(f"loss: {last_loss:>3f}  [{current:>5d}/{size:>5d}]")
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= size
        log.debug(f"Epoch {t+1} training acc.: {(100*correct):>0.1f}%")
    return 100.0 * correct, loss


def test(model, dataloader, loss_fn, device):
    # log.debug('Testing')
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
    log.debug(f"Test acc.: {(100*correct):>0.1f}%, Avg loss: {test_loss:.2f}")
    model.train()
    return 100. * correct


def warm_up(model, train_dl, test_dl, loss_fn, optimizer, args, device):
    max_acc = -1

    for i in range(args.net_warmup):
        log.debug(f"warm-up epoch [{i} / {args.net_warmup}]")
        acc, _ = train(model, train_dl, loss_fn, optimizer, None,
                       args.net_train_per_epoch, device)
        max_acc = max(max_acc, test(model, test_dl, loss_fn, device))
    log.debug(f"Max acc. for warm-up: {max_acc}")
    return max_acc


def main():
    args = utils.get_args()
    logger = utils.setup_logger_dir(args)
    args = utils.get_yaml_args(args)
    device = utils.get_device(args)

    data = Data(args.batch_size, C.DATA_DIR, args.dataset)
    num_classes = data.get_num_classes()
    train_dl, test_dl = data.train_dataloader, data.test_dataloader
    network = Network(device, args.arch, num_classes, args.pretrained)
    preprocess = network.preprocess
    model = network.set_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for i in range(3):
        train_acc = train(model, train_dl, loss_fn, optimizer,
                          args.train_epochs, device) 
        test_acc = test(model, test_dl, loss_fn, device)


if __name__ == '__main__':
    main()

