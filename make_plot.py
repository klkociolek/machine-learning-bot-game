import matplotlib.pyplot as plt

# Epoki
epochs = [0, 1, 2, 3, 4]

#networks 1st dataset
# train_acc_net1 = [0.583, 0.71, 0.754, 0.782, 0.803]
# val_acc_net1   = [0.672, 0.73, 0.758, 0.771, 0.783]
#
# train_acc_net2 = [0.698, 0.815, 0.847, 0.866, 0.881]
# val_acc_net2   = [0.794, 0.832, 0.849, 0.861, 0.87]
#
# train_acc_net3 = [0.706, 0.826, 0.858, 0.878, 0.893]
# val_acc_net3 = [0.806, 0.838, 0.857, 0.869, 0.881]

# #networks 2nd dataset
# train_acc_net1 = [0.576, 0.702, 0.745, 0.772, 0.792]
# val_acc_net1   = [0.665, 0.727, 0.743, 0.764, 0.776]
#
# train_acc_net2 = [0.685, 0.811, 0.843, 0.863, 0.876]
# val_acc_net2   = [0.777, 0.826, 0.846, 0.858, 0.87]
#
# train_acc_net3 = [0.725, 0.83, 0.859, 0.877, 0.892]
# val_acc_net3 = [0.795, 0.843, 0.861, 0.869, 0.882]

#networks 3rd dataset
# train_acc_net1 = [0.459, 0.636, 0.696, 0.736, 0.768]
# val_acc_net1   = [0.591, 0.655, 0.686, 0.712, 0.732]
#
# train_acc_net2 = [0.499, 0.685, 0.751, 0.79, 0.816]
# val_acc_net2   = [0.597, 0.699, 0.766, 0.793, 0.815]
#
# train_acc_net3 = [0.507, 0.722, 0.782, 0.813, 0.838]
# val_acc_net3 = [0.69, 0.744, 0.774, 0.811, 0.822]

# Tworzymy wykres
plt.figure(figsize=(8, 6))

# network1
plt.plot(epochs, train_acc_net1, 'b-o', label='network1 - training accuracy')
plt.plot(epochs, val_acc_net1,   'b--o', label='network1 - validation accuracy')

# network2
plt.plot(epochs, train_acc_net2, 'r-o', label='network2 - training accuracy')
plt.plot(epochs, val_acc_net2,   'r--o', label='network2 - validation accuracy')

# network3
plt.plot(epochs, train_acc_net3, 'g-o', label='network3 - training accuracy')
plt.plot(epochs, val_acc_net3,   'g--o', label='network3 - validation accuracy')

plt.title('Accuracy - network1 vs. network2 vs. network3')
plt.suptitle('Dataset: 750 uji/ 250 random')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()