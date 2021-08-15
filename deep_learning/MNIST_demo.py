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

X_val = mnist_test.test_data
y_val = mnist_test.test_labels

nn_classifier = CNNClassifier(X_val.shape[1], int(torch.max(y_val)+1))
loss = nn_classifier.fit(train_load, test_load, return_loss=True,
                            epochs=10, acceptable_error=0.0001, lr=0.01)
for idx, (X_val, y_v) in enumerate(test_load):
    y_hat = nn_classifier(X_val)
    if idx == 0:
        y_hat_val = nn_classifier(X_val)
        y_val = y_v
    else:
        y_hat_val = torch.cat((y_hat_val, nn_classifier(X_val)), dim=0)
        y_val = torch.cat((y_val, y_v), dim=0)

print(torch.cat((y_val.reshape(-1, 1), y_hat_val), dim=1)[0:10])
print("R^2 score:", r2_score(y_hat_val.detach().numpy(), y_val.detach().numpy()))
plt.plot(loss['training'], label="Training set loss")
plt.plot(loss['validation'], label="Validation set loss")
plt.xlabel(f"Epochs\nl={loss['validation'][-1]}")
plt.ylabel("CE")
plt.show()
# plt.savefig(fname="linear_regression.jpg")