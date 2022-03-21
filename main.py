"""Train MLQP network directly on two spirals problem"""

import matplotlib.pyplot as plt
import numpy as np
import mlqp

lrl = 1e0
lrq = 1e0 / 2
decay = 0.95
max_epoch = 8000
norm_d = 6

network_structure = [2, 15, 1]
print_freq = 1


if __name__ == '__main__':
    # Import data
    file = "two_spiral_train_data-1.txt"
    train_data = np.array(np.loadtxt(file))
    file2 = "two_spiral_test_data-1.txt"
    test_data = np.array(np.loadtxt(file2))
    len_train_data = len(train_data)

    # Init MLQP network
    network = mlqp.MLQP(network_structure, lrl, lrq, decay)

    epoch = 0
    loss = []
    errors = []
    total_cnt, correct = 0, 0
    # Train
    while epoch // len_train_data < max_epoch:
        [x1, x2, y] = train_data[np.random.randint(0, len_train_data)]
        o = network.forward([x1 / norm_d, x2 / norm_d])
        network.update(y - o[0][0])
        errors.append(np.abs(y - o[0][0]))
        total_cnt += 1
        correct += np.abs(y - o[0][0]) < 0.5
        epoch += 1
        if epoch % len_train_data == 0:
            print(int(epoch / len_train_data), f"lr={network._lrl}",  "rate = " + str(round(correct / total_cnt * 100, 2)) + "%",
                  "error_mean = " + str(round(np.mean(errors), 6)),
                  "error_std = " + str(round(np.std(errors), 6)))
            if epoch % (len_train_data * 100) == 0:
                loss.append(np.mean(errors))
            errors = []
            total_cnt, correct = 0, 0

    # Evaluate on test data
    total_cnt, correct = 0, 0
    for x1, x2, y in test_data:
        o = network.forward([x1 / norm_d, x2 / norm_d])
        total_cnt += 1
        correct += np.abs(y - o[0][0]) < 0.5
    print("Total correct rate:" + str(round(correct / total_cnt * 100, 4)) + "%")

    # Make plot data
    X, Y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
    Z = np.zeros([256, 256])
    for i in range(len(X)):
        for j in range(len(Y)):
            o = network.forward([X[i][j], Y[i][j]])
            Z[i][j] = o[0][0]

    levels = np.linspace(0, 1, 10)

    # Plot result diagram
    plt.style.use('_mpl-gallery-nogrid')
    fig1, ax1 = plt.subplots()

    data_1 = np.array([x / norm_d for x in test_data if x[2] == 1])
    x = data_1[:, [0]]
    y = data_1[:, [1]]
    ax1.plot(x, y, "o", markersize=2, color="black")
    data_2 = np.array([x / norm_d for x in test_data if x[2] == 0])
    x = data_2[:, [0]]
    y = data_2[:, [1]]
    ax1.plot(x, y, "o", markersize=2, color="grey")
    ax1.contourf(X, Y, Z, levels=levels)
    plt.show()

    # Plot loss function
    plt.style.use('_mpl-gallery')
    fig2, ax2 = plt.subplots()
    x = np.arange(len(loss))
    ax2.plot(x, loss)
    plt.show()
