from nn_models import CNNClassifier
from torchvision import datasets, transforms
import torch
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

mnist_train = datasets.MNIST(root="datasets/mnist_train",
                             download=True, train=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))]))

mnist_test = datasets.MNIST(root="datasets/mnist_test",
                             download=True, train=False,
                             transform=transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))]))

train_load = DataLoader(mnist_train, batch_size=16,
            shuffle=True, num_workers=round(cpu_count()/2))
test_load = DataLoader(mnist_test, batch_size=16,
            shuffle=False, num_workers=round(cpu_count()/2))


nn_sdg = CNNClassifier()
sdg = torch.optim.SGD(nn_sdg.parameters(), lr=0.01)

nn_sdg_m = CNNClassifier()
sdg_m = torch.optim.SGD(nn_sdg.parameters(), lr=0.01, momentum=0.99)

nn_sdg_nesterov = CNNClassifier()
sdg_nesterov = torch.optim.SGD(nn_sdg_nesterov.parameters(), lr=0.01, momentum=0.99, nesterov=True)

nn_adam = CNNClassifier()
adam = torch.optim.AdamW(nn_adam.parameters(), lr=0.01, weight_decay=1e-4)

models = {"SDG":nn_sdg,
            "SDG_Momentum": nn_sdg_m,
            "SDG_Nesterov": nn_sdg_nesterov,
            "Adam": nn_adam}

optimisers = {"SDG": sdg,
                "SDG_Momentum": sdg_m,
                "SDG_Nesterov": sdg_nesterov,
                "Adam": adam}

loss = {}
y_val = {}
y_hat_val={}
for model in models.keys():
    loss[model] = models[model].fit(train_load, test_load, return_loss=True,
                            optimiser=optimisers[model],
                            epochs=10, acceptable_error=0.0001, lr=0.01)

    y_val[model], y_hat_val[model] = models[model].predict(test_load, return_y=True)

fig, axs = plt.subplot(1, 4)

for ax, model in zip(axs, loss.keys()):
    ax.plot(loss[model]['training'], label="Training set loss")
    ax.plot(loss[model]['validation'], label="Validation set loss")
    ax.set(xlabel=f"Epochs\nl={loss[model]['validation'][-1]}\n{model}", ylabel="CE")
    ax.legend()
    print(model)
    print(torch.cat((y_val[model], y_hat_val[model]), dim=1)[0:10])
    print("R^2 score:", r2_score(y_hat_val[model].detach().numpy(), y_val[model].detach().numpy()))

fig.show()