import matplotlib.pyplot as plt
import numpy as np


def plot_sample(data, labels, label_names, img_per_class=10):
    """Plot a sample if images from data."""
    num_classes = len(set(labels))
    fig, ax = plt.subplots(num_classes, img_per_class, figsize=(10, 10))
    for label in range(num_classes):
        class_imgs = data[labels == label]
        num_examples = class_imgs.shape[0]
        random_indices = np.random.choice(
            num_examples, img_per_class, replace=False)
        for i, idx in enumerate(random_indices):
            ax[label, i].imshow(class_imgs[idx, :, :, :])
            ax[label, i].axis('off')

        ax[label, 0].text(-5, 20, label_names[label], ha='right')

    plt.show();


def plot_projection(data, labels, label_names, title):
    """
    Plot a scatter plot of data with colour map given by labels.

    Assume len(labels) = 10.
    """
    colorbar_formatter = plt.FuncFormatter(lambda val, loc: label_names[loc])

    plt.figure(figsize=(13, 10))
    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap('tab10', 10))
    plt.colorbar(ticks=range(10), format=colorbar_formatter)
    plt.clim(-0.5, 9.5)
    plt.title(title)
    plt.show();


def plot_multiple_tsne_embeddings(
        cnn_codes_tsne_0,
        cnn_codes_tsne_1,
        cnn_codes_tsne_2,
        labels,
        label_names,
        perplexities=(30, 50, 100),
):
    """
    Plot three graphs of t-SNE embeddings for three values of perplexity.

    Assume len(labels) = 10.
    """
    colorbar_formatter = plt.FuncFormatter(lambda val, loc: label_names[loc])

    plt.figure(figsize=(25, 7))
    plt.suptitle('t-SNE 2D embedding', size=20)

    plt.subplot(1, 3, 1)
    plt.scatter(
        cnn_codes_tsne_0[:, 0],
        cnn_codes_tsne_0[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap('tab10', 10))
    plt.title(f'perplexity = {perplexities[0]}')

    plt.subplot(1, 3, 2)
    plt.scatter(
        cnn_codes_tsne_1[:, 0],
        cnn_codes_tsne_1[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap('tab10', 10))
    plt.title(f'perplexity = {perplexities[1]}')

    plt.subplot(1, 3, 3)
    plt.scatter(
        cnn_codes_tsne_2[:, 0],
        cnn_codes_tsne_2[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap('tab10', 10))
    plt.title(f'perplexity = {perplexities[2]}')

    plt.colorbar(ticks=range(10), format=colorbar_formatter)
    plt.clim(-0.5, 9.5)
    plt.show();
