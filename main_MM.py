"""Train min-max MLQP network on two spirals problem"""

import matplotlib.pyplot as plt
import numpy as np
import mlqp
import math

network_structure = [2, 15, 1]
coef = [1, 3, 9]
colors = ["r", "g", "b"]
lrl = 1
lrq = 0.7
max_epoch = 6000
div_num = 1
div_count = 1
decay_to = 1
random_divide = True

norm_d = 6
print_freq = max_epoch / 50
decay_freq = 100
dec_boundary = [0.47, 0.53]

inMMmode = True     # If to use Min-Max Module Network
if inMMmode:
    network_structure = [2, 15, 1]
    coef = [3]
    lrl = 1
    lrq = 0.7
    max_epoch = 1501
    div_num = 9
    div_count = 3
    decay_to = 0.3
    print_freq = 100
    random_divide = False
  


if __name__ == '__main__':

    def evaluate(network, x):
        o = []
        for i in range(div_num):
            o.append(network[i].forward(x))
        return np.max([np.min(o[div_count * i:div_count * (i + 1)]) for i in range(int(div_num // div_count))])


    # Import data
    file = "two_spiral_train_data-1.txt"
    train_data = np.array(np.loadtxt(file))
    file2 = "two_spiral_test_data-1.txt"
    test_data = np.array(np.loadtxt(file2))
    len_train_data = len(train_data)

    div_train_data = [[] for _ in range(div_num)]
    np.random.shuffle(train_data)
    pos_train_data = [x for x in train_data if x[2] == 1]
    neg_train_data = [x for x in train_data if x[2] == 0]
    m, n = div_num // div_count, div_count

    # Randomly divide the training data into div_num parts
    if random_divide:
        pos_div_data = [[] for _ in range(m)]
        neg_div_data = [[] for _ in range(n)]
        i = 0
        for d in pos_train_data:
            pos_div_data[i].append(d)
            i = (i + 1) % m
        i = 0
        for d in neg_train_data:
            neg_div_data[i].append(d)
            i = (i + 1) % n
        for i in range(m):
            for j in range(n):
                for d in pos_div_data[i]:
                    div_train_data[i * n + j].append(d)
                for d in neg_div_data[j]:
                    div_train_data[i * n + j].append(d)
                np.random.shuffle(div_train_data[i * n + j])
    else:
        pos_div_data = [[] for _ in range(m)]
        neg_div_data = [[] for _ in range(n)]
        i = 0
        angleP = 2 * math.pi / m
        angleN = 2 * math.pi / m
        for d in pos_train_data:
            ang = (math.atan2(d[1], d[0]) + 2 * math.pi) % (2 * math.pi)
            for i in range(m):
                if i * angleP <= ang < (i + 1) * angleP:
                    pos_div_data[i].append(d)
                    break
        for d in neg_train_data:
            ang = (math.atan2(d[1], d[0]) + 2 * math.pi) % (2 * math.pi)
            for i in range(m):
                if i * angleN <= ang < (i + 1) * angleN:
                    neg_div_data[i].append(d)
                    break
        for i in range(m):
            for j in range(n):
                for d in pos_div_data[i]:
                    div_train_data[i * n + j].append(d)
                for d in neg_div_data[j]:
                    div_train_data[i * n + j].append(d)
                np.random.shuffle(div_train_data[i * n + j])

    loss = [[[] for _ in range(div_num)] for _ in range(len(coef))]
    for k in range(len(coef)):
        # Initialize MLQP network
        network = []
        eval_result = []
        for i in range(div_num):
            len_div_train_data = len(div_train_data[i])
            decay_period = max_epoch * len_div_train_data // 100
            decay = decay_to ** (1 / 100)

            network.append(mlqp.MLQP(network_structure, lrl * coef[k], lrq * coef[k], decay, decay_period))

            epoch = 0
            errors = []
            total_cnt, correct = 0, 0
            mss = 1
            while epoch // len_div_train_data < max_epoch:
            # while mss > 0.01 and epoch // len_div_train_data < max_epoch:
                [x1, x2, y] = div_train_data[i][np.random.randint(0, len_div_train_data)]
                o = network[i].forward([x1 / norm_d, x2 / norm_d])
                network[i].update(y - o[0][0])
                errors.append(np.abs(y - o[0][0]))
                total_cnt += 1
                correct += np.abs(y - o[0][0]) < 0.5
                epoch += 1
                if (epoch / len_div_train_data) % print_freq == 0:
                    l1_error = np.mean(errors)
                    mss = np.std(errors)
                    print(f"coef = {coef[k]}", i, int(epoch / len_div_train_data), f"lr = {round(network[i]._lrl, 4)}", "rate = " + str(round(correct / total_cnt * 100, 2)) + "%",
                          "error_mean = " + str(round(l1_error, 6)),
                          "error_std = " + str(round(mss, 6)))
                    # if epoch % (len_div_train_data * (max_epoch / 1)) == 0:
                    loss[k][i].append(mss)
                    errors = []
                    total_cnt, correct = 0, 0

            # Evaluate on test data
            total_cnt, correct = 0, 0
            for x1, x2, y in test_data:
              o = network[i].forward([x1 / norm_d, x2 / norm_d])[0][0]
              total_cnt += 1
              correct += np.abs(y - o) < 0.5
            print(f"{i} Network correct rate:" + str(round(correct / total_cnt * 100, 4)) + "%")

        # Evaluate on test data
        total_cnt, correct = 0, 0
        for x1, x2, y in test_data:
            o = evaluate(network, [x1 / norm_d, x2 / norm_d])
            total_cnt += 1
            correct += np.abs(y - o) < 0.5
        print("Total correct rate:" + str(round(correct / total_cnt * 100, 4)) + "%")


        # Make plot data
        X, Y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        Z = np.zeros([256, 256])
        decX = []
        decY = []
        for i in range(len(X)):
            for j in range(len(Y)):
                o = evaluate(network, [X[i][j], Y[i][j]])
                Z[i][j] = o
                if dec_boundary[0] < o and o < dec_boundary[1]:
                    decX.append(X[i][j])
                    decY.append(Y[i][j])

        levels = np.linspace(0, 1, 3)

        # Plot result diagram
        plt.style.use('_mpl-gallery')
        fig1, ax1 = plt.subplots()
        # Contour
        ct = ax1.contourf(X, Y, Z, levels=levels)
        # plt.clabel(ct, fontsize=10, colors=('k'))
        # Decision boundary
        # ax1.plot(decX, decY, "o", markersize=1, color="black")
        # Sample points
        data_1 = np.array([x / norm_d for x in test_data if x[2] == 1])
        x = data_1[:, [0]]
        y = data_1[:, [1]]
        ax1.plot(x, y, "o", markersize=2, color="red", label="positive")
        data_1 = np.array([x / norm_d for x in test_data if x[2] == 0])
        x = data_1[:, [0]]
        y = data_1[:, [1]]
        ax1.plot(x, y, "o", markersize=2, color="green", label="negative")
        # Set axis label
        ax1.set_ylabel("y-axis", fontsize='xx-large')
        ax1.set_xlabel("x-axis", fontsize='xx-large')
        # Set legend
        plt.legend(loc='upper right', fontsize='xx-small')
        plt.show()

    # Plot loss function
    plt.style.use("classic")
    aaa = plt.style.available
    fig2, ax2 = plt.subplots()
    for i in range(len(coef)):
        for j in range(len(loss[i])):
            x = np.arange(0, len(loss[i][j]) * print_freq, print_freq)
            if not inMMmode:
                ax2.plot(x, loss[i][j], color=colors[i], label=f"learning rate: { lrl * coef[i] }")
            else:
                ax2.plot(x, loss[i][j], label=f"Sub-problem: {j}")
    # Set axis label
    ax2.set_ylabel("Loss Value", fontsize='xx-large')
    ax2.set_xlabel("Epochs", fontsize='xx-large')
    # Set legend
    if not inMMmode:
        plt.legend(loc='upper right', fontsize='xx-large')
    else:
        plt.legend(loc='upper right', fontsize='xx-small')
    plt.show()

    pass


