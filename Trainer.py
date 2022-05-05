import torch
import torch.nn as nn
from dataset.dataLoader import DL
from network import Classifier
from sklearn.metrics import accuracy_score
from vision import plot_loss, plot_acc


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._init_data()
        self._init_model()

    def _init_model(self):
        self.net = Classifier(self.n_classes, self.args)
        self.net.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), )
        self.cri = nn.CrossEntropyLoss()

    def _init_data(self):
        data = DL()
        self.n_classes = data.n_classes
        self.traindl = data.traindl
        self.testdl = data.testdl

    def eva(self):
        with torch.no_grad():
            real = []
            pred = []
            for batch, (inputs, targets, length) in enumerate(self.testdl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = self.net(inputs, length)
                real += targets.detach().cpu().numpy().tolist()
                pred += output.argmax(dim=1).detach().cpu().numpy().tolist()

            acc = accuracy_score(real, pred)

        return acc

    def train(self):
        train_accs = []
        eva_accs = []
        all_loss = []
        patten = 'Iter: %d/%d   [=============]   loss: %.5f    train_acc: %.5f     eva_acc: %.5f'
        for epoch in range(self.args.epochs):
            real = []
            pred = []
            losses = 0
            for batch, (inputs, targets, length) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = self.net(inputs, length)
                loss = self.cri(output, targets.squeeze())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                real += targets.detach().cpu().numpy().tolist()
                pred += output.argmax(dim=1).detach().cpu().numpy().tolist()
                losses += loss.item()

            train_acc = accuracy_score(real, pred)
            eva_acc = self.eva()
            print(patten % (
                epoch,
                self.args.epochs,
                losses,
                train_acc,
                eva_acc,
            ))

            train_accs.append(train_acc)
            eva_accs.append(eva_acc)
            all_loss.append(losses)

        plot_acc(train_accs, eva_accs)
        plot_loss(all_loss)
