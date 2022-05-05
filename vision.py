from matplotlib import pyplot as plt


def plot_acc(train_acc, eva_acc):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    x = list(range(len(train_acc)))

    ax.plot(x, train_acc, label="train_acc")
    ax.plot(x, eva_acc, label='eva_acc')
    plt.legend(loc='best')

    plt.savefig('results/acc.png', dpi=400)


def plot_loss(losses):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    x = list(range(len(losses)))

    ax.plot(x, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.savefig('results/loss.png', dpi=400)
