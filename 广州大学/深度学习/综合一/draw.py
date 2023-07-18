import matplotlib.pyplot as plt
import sys
import os
os.chdir(sys.path[0])


def plot_results(results, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    colors_train = ["#f9791e", "#3dd378", "#f7dc05", "#3d98d3"]
    colors_test = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    markers = ['-', '--', '-.', ':']

    for i, result in enumerate(results):
        train_losses, train_accs, test_losses, test_accs, optimizer_name = result

        color_train = colors_train[i % len(colors_train)]
        color_test = colors_test[i % len(colors_test)]
        marker = markers[i % len(markers)]
        # x 轴刻度为每个 epoch 的索引值
        x = range(1, len(train_losses) + 1)

        ax1.plot(x, train_losses, color=color_train, linestyle=marker,
                 label=f'Train Loss ({optimizer_name})')
        ax1.plot(x, test_losses, color=color_test, linestyle=marker,
                 label=f'Test Loss ({optimizer_name})')
        ax2.plot(x, train_accs, color=color_train, linestyle=marker,
                 label=f'Train Acc ({optimizer_name})')
        ax2.plot(x, test_accs, color=color_test, linestyle=marker,
                 label=f'Test Acc ({optimizer_name})')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper right')
    ax1.set_xticks(x)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')
    ax2.legend(loc='lower right')
    ax2.set_xticks(x)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.savefig('figure/{}.png'.format(title), dpi=300)
    plt.close()
    # plt.show()
    
    
