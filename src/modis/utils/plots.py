import os
from typing import Literal

import umap
import colorsys
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

colorblind_safe_colors = [
    '#E6194B',
    '#3CB44B',
    '#4363D8',
    '#42D4F4',
    '#F032E6',
    '#FFE119',
    '#000000',
    '#ADD8E6',
    '#F58231',
    '#911EB4',
    '#FABED4',
    '#A9A9A9'
]

colors = [
    "#FF0000",   # Red
    "#80FF00",   # Lime
    "#0080C0",   # Medium Blue    
    "#F58231",   # Orange
    "#8000FF",   # Indigo
    "#FF80FF",   # Lavender
    "#FFFF00",   # Yellow
    #"#00FFFF",   # Cyan
]

def get_colors(n=5) -> list[str]:
    """Return a list of hexadecimal color codes """
    # assert n <= len(colors), f"Only {len(colors)} available"
    if n > len(colors):
        import matplotlib.colors as mcolors
        colors_from_cmap = plt.cm.rainbow(np.linspace(0, 1, n))
        hex_colors = [mcolors.rgb2hex(c) for c in colors_from_cmap]
        return hex_colors
    return colors[:n]

def generate_colors(hex_color:str, n:int):
    """
    Generate n versions of the hue in hex_color by changing the lightness and saturation

    Args:
        hex_color (str): Color in hex format
        n (int): Number of colors to generate

    Return:
        (list): List of generated colors
    """
    def hex_to_rgb(hex):
        return tuple(int(hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    def rgb_to_hex(rgb):
        return f"#{int(rgb[0]*255):02X}{int(rgb[1]*255):02X}{int(rgb[2]*255):02X}"

    r, g, b = hex_to_rgb(hex_color)
    
    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    step = 0.15
    start_point = 0.5 - (n*step/2)  - step
    if start_point < 0:
        step = 1/(n+2)
        start_point = step
    
    # Generate colors
    colors = []
    for i in range(n):
        # Adjust lightness
        new_l = min(1, max(0, start_point + step + i*step))
        
        # Adjust saturation
        new_s = min(1, max(0, start_point + step + i*step))
        new_s = 1 + start_point - new_s
        new_s = max(0.3, new_s)
        
        # Convert back to RGB and then to hex
        new_rgb = colorsys.hls_to_rgb(h, new_l, new_s)
        colors.append(rgb_to_hex(new_rgb))
    
    return colors

def get_class_per_modality_colors(num_modalities, num_classes):
    """
    Return a list of colors per modality per class
    """
    class_colors = get_colors(num_classes)
    colors_per_modality = list(zip(*[generate_colors(color, num_modalities) for color in class_colors]))
    colors = [color for modality_color in colors_per_modality for color in modality_color]
    return colors

def plot_2d_projection(
    technique: Literal['pca', 'umap'],
    data,
    labels,
    labels_colors: str = None | list[str],
    labels_names: str = None | list[str],
    standardize: bool = True,
    filename: str = None,
    save_path: str = '../saved/reports',
    figsize: tuple = (8, 8),
    text_size: int = 20
) -> None:
    """
    Display and save a 2D PCA or UMAP plot
    
    Args:
        technique (str): Choose between 'pca' or 'umap' dimensionality reduction technique
        labels_colors (None | list[str]): One hex color code for each unique class label
        labels_names (None | list[str]): One name for each unique class label
    """
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data

    if technique == 'pca':
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(data_scaled)
        x_label = 'PC1'
        y_label = 'PC2'
        title = 'PCA'
    elif technique == 'umap':
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
        reduced_data = reducer.fit_transform(data_scaled)
        x_label = 'UMAP component 1'
        y_label = 'UMAP component 2'
        title = 'UMAP'
    else:
        raise ValueError("technique must be either 'pca' or 'umap'")

    plt.figure(figsize=figsize)
    
    # Iterate through unique labels
    unique_labels = set(labels)
    for label in unique_labels:
        # Determine color
        if labels_colors:
            lc = labels_colors[label]
        else:
            lc = colors[label]
        
        # Determine label name
        if labels_names:
            named_label = labels_names[label]
        else:
            named_label = f'class {label}'
        
        # Create mask for current label
        mask = labels == label
        
        plt.scatter(
            reduced_data[mask, 0], 
            reduced_data[mask, 1], 
            c=lc, 
            label=named_label, 
            s=20
        )

    plt.xlabel(x_label, fontsize=text_size)
    plt.ylabel(y_label, fontsize=text_size)
    plt.xticks(fontsize=text_size)
    plt.yticks(fontsize=text_size)
    # plt.title(title, fontsize=text_size)
    
    if len(unique_labels) < 10: ### Find better parameter and use also for 3d plots
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize=text_size)  # bbox_to_anchor=(1.6, 1)

    # Save or show the plot
    if filename is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f'{filename}.svg'), 
            format='svg', 
            bbox_inches='tight'
        )
        plt.close()
    else:
        plt.show()

def plot_3d_projection(
    technique: Literal['pca', 'umap'],
    data,
    labels,
    labels_colors: str = None | list[str],
    labels_names: str = None | list[str],
    standardize: bool = False,
    filename: str = None,
    save_path: str = '../saved/reports',
    width: int = 800,
    height: int = 800
) -> None:
    """
    Display and save a 3D dimensionality reduction plot
    
    Args:
        technique (str): Choose between 'pca' or 'umap' dimensionality reduction technique
        labels_colors (None | list[str]): List of colors for each unique label
        labels_names (None | list[str]): List of names for each unique label
        standardize (boolean): Whether to standardize the data before reduction
        width (int): Width of the plot
        height (int): Height of the plot
    """
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data

    if technique == 'pca':
        reducer = PCA(n_components=3)
        reduced_data = reducer.fit_transform(data_scaled)
        x_label = 'PC1'
        y_label = 'PC2'
        z_label = 'PC3'
        title = 'PCA'
    elif technique == 'umap':
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3)
        reduced_data = reducer.fit_transform(data_scaled)
        x_label = 'UMAP 1'
        y_label = 'UMAP 2'
        z_label = 'UMAP 3'
        title = 'UMAP'
    else:
        raise ValueError("technique must be either 'pca' or 'umap'")

    fig = go.Figure()

    # Iterate through unique labels
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # Determine color
        if labels_colors:
            lc = labels_colors[label]
        else:
            lc = colors[label]
        
        # Determine label name
        if labels_names:
            named_label = labels_names[label]
        else:
            named_label = f'class {label}'
        
        # Create mask for current label
        mask = labels == label
        
        # Add trace for current label
        fig.add_trace(go.Scatter3d(
            x=reduced_data[mask, 0],
            y=reduced_data[mask, 1],
            z=reduced_data[mask, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=lc,
                opacity=0.8,
            ),
            name=named_label
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ),
        margin=dict(r=10, b=10, l=10, t=30),
        legend=dict(
            font=dict(size=14),
            itemsizing='constant'
        ),
        title=title,
        width=width,
        height=height
    )

    # Save or show the plot
    if filename is not None:
        os.makedirs(save_path, exist_ok=True)
        pio.write_image(fig, os.path.join(save_path, f'{filename}.svg'))
    else:
        fig.show()

def plot_training_log(
    logs: list,
    training_mode: str,
    modality_names: list,
    filename: str = None,
    save_path: str = '../saved/reports',
    figsize: tuple = (15, 10)
) -> None:
    """Plot training logs"""
    num_modalities = len(logs[0]['recon_loss_modal'])

    x = range(logs[0]['epoch_idx'] + 1, logs[-1]['epoch_idx'] + 2)

    recon_loss = [item['recon_loss'] for item in logs]
    recon_loss_modal = [[item['recon_loss_modal'][i] for item in logs] for i in range(num_modalities)]
    kl_loss = [item['kl_loss'] for item in logs]
    kl_loss_modal = [[item['kl_loss_modal'][i] for item in logs] for i in range(num_modalities)]
    d_train_loss = [item['d_train_loss'] for item in logs]
    d_loss = [item['d_loss'] for item in logs]
    global_loss = [item['global_loss'] for item in logs]
    d_train_adv_acc = [item['d_train_adv_acc'] for item in logs]
    d_train_aux_acc = [item['d_train_aux_acc'] for item in logs]
    d_train_adv_modal_acc = list(zip(*[item['d_train_adv_modal_acc'] for item in logs]))

    fig = plt.figure(figsize=figsize)

    fig.suptitle(f'Training mode: {training_mode}', y=0.97)

    plt.subplot(2, 3, 1)
    plt.plot(x, recon_loss, label='recon_loss')
    for i in range(num_modalities):
        plt.plot(x, recon_loss_modal[i], label='recon_loss_'+modality_names[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(x, kl_loss, label='kl_loss')
    for i in range(num_modalities):
        plt.plot(x, kl_loss_modal[i], label='kl_loss_'+modality_names[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(x, d_train_loss, label='d_train_loss')
    plt.plot(x, d_loss, label='d_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(x, global_loss, label='global_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(x, d_train_adv_acc, label='d_train_adv_acc')
    plt.plot(x, d_train_aux_acc, label='d_train_aux_acc')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 6)
    for i in range(num_modalities):
        plt.plot(x, d_train_adv_modal_acc[i], label=f'd_adv_acc_{modality_names[i]}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add space for subtitle

    if filename is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f'{filename}.svg'), 
            format='svg', 
            bbox_inches='tight'
        )
        plt.close()
    else:
        plt.show()


from scipy.optimize import linear_sum_assignment
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, jaccard_score, f1_score, normalized_mutual_info_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score


def remap_labels(true_labels, predicted_labels):
    """
    Remap the labels to the permutation maximizing the sum of the diagonal elements in the confusion matrix
    by applying the Hungarian algorithm
    """
    def _make_cost_m(cm):
        s = np.max(cm)
        return (-cm + s)

    cm = confusion_matrix(true_labels, predicted_labels)
    row_indexes, col_indexes = linear_sum_assignment(_make_cost_m(cm))
    #remapped_cm = cm[:, col_indexes]

    names = sorted(np.unique(np.concatenate((true_labels, predicted_labels))))
    remapped_names = [names[i] for i in col_indexes]
    label_mapping = dict(zip(remapped_names, names))
    
    remapped_labels = np.vectorize(label_mapping.get)(predicted_labels)
    
    return remapped_labels #, remapped_cm

def kmeans_clustering(data, true_labels, n_clusters):
    # Regular Kmeans
    #from sklearn.cluster import KMeans
    #kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    #kmeans.fit(data)
    #pred_labels = kmeans.labels_
    #pred_labels_str = [str(l) for l in pred_labels]
    
    # Spherical Kmeans
    from scipy.cluster.vq import kmeans, vq
    #from sklearn.preprocessing import normalize
    #normalized_data = normalize(data)
    centroids, _ = kmeans(data, n_clusters)
    pred_labels, _ = vq(data, centroids)

    remapped_labels = remap_labels(true_labels, pred_labels)

    return remapped_labels

def acc_metrics(true_labels, pred_labels): # : np.ndarray?
    cm = confusion_matrix(true_labels, pred_labels)
    acc = np.trace(cm) / np.sum(cm)

    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)  # average of recall obtained on each class
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ji = jaccard_score(true_labels, pred_labels, average='macro')    
    ari = adjusted_rand_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Inverse frequency weighting accuracy
    class_weights = dict()
    for class_label in np.unique(true_labels):
        class_weights[class_label] = 1 / np.sum(true_labels == class_label)
    sample_weights = np.array([class_weights[y] for y in true_labels])
    weighted_accuracy = accuracy_score(true_labels, pred_labels, sample_weight=sample_weights)

    return {
        'acc': acc.item(),
        'inv-freq-w-acc': weighted_accuracy,
        'balanced-acc': balanced_acc,
        'nmi': nmi if type(nmi) == float else nmi.item(),
        'ji': ji.item(),
        'ari': ari,
        'f1': f1.item()
    }

def plot_confusion_matrix(
    true_labels,
    pred_labels,
    filename: str = None,
    performance_metrics: bool = False,
    save_path: str = '../saved/reports',
    figsize = (16, 8)
) -> None:
    """
    Display a confusion matrix.

    Note: If there are many classes adjust figsize to fix recall and precision plots, add n to each
    
    Args:
        true_labels (array-like of shape (n_samples,))
        pred_labels (array-like of shape (n_samples,))
        performance_metrics: if True, plot recall and precision plots
    """
    metrics = acc_metrics(true_labels, pred_labels)
    matrics_text = f"""
    ACC: {metrics['acc']:.3f}
    IFW-AAC: {metrics['inv-freq-w-acc']:.3f}
    B-AAC: {metrics['balanced-acc']:.3f}
    JI: {metrics['ji']:.3f}
    NMI: {metrics['nmi']:.3f}
    F1: {metrics['f1']:.3f}
    ARI: {metrics['ari']:.3f}
    """

    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) #, display_labels=['num'+str(n) for n in range(10)])
    
    if performance_metrics:
        # Calculate precision and recall
        precision = np.nan_to_num(np.diag(cm) / cm.sum(axis=0))
        recall = np.nan_to_num(np.diag(cm) / cm.sum(axis=1))

        num_classes = len(np.unique(true_labels))

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[num_classes, 1, num_classes],  # (1-(1/num_classes))
            height_ratios=[num_classes, 1],
            hspace = 0.2,
            wspace = 0.1
        )
    
        disp.plot(ax=fig.add_subplot(gs[0, 0]), cmap=plt.cm.Blues, text_kw={'fontsize': 20}, colorbar=False)
        disp.ax_.set_aspect('equal')
        # disp.ax_.set_title('Confusion Matrix', fontsize=20)
        disp.ax_.tick_params(axis='both', which='major', labelsize=20)  # Increase tick label size
        disp.ax_.xaxis.label.set_size(20)
        disp.ax_.yaxis.label.set_size(20)
    
        # Add precision plot
        ax_precision = fig.add_subplot(gs[1, 0])
        precision_2d = precision.reshape(1, -1)
        ax_precision.imshow(precision_2d, cmap=plt.cm.Blues, aspect='equal', vmin=0, vmax=1)
        for j, v in enumerate(precision):
            ax_precision.text(j, 0, f'{v:.2f}', ha='center', va='center', fontsize=20, color='white' if v > 0.7 else 'black')  # plt.cm.Blues(1)
        ax_precision.set_xticks([])
        ax_precision.set_yticks([])
        ax_precision.set_xlabel('Precision', fontsize=20)

        # Add recall plot
        ax_recall = fig.add_subplot(gs[0, 1])
        recall_2d = recall.reshape(-1, 1)
        ax_recall.imshow(recall_2d, cmap=plt.cm.Blues, aspect='equal', vmin=0, vmax=1)
        for i, v in enumerate(recall):
            ax_recall.text(0, i, f'{v:.2f}', ha='center', va='center', fontsize=20, color='white' if v > 0.7 else 'black')
        ax_recall.set_yticks([])
        ax_recall.set_xticks([])
        ax_recall.set_xlabel('Recall', fontsize=20)
    
        # Metrics
        ax_metrics = fig.add_subplot(gs[0:2, 2])
        ax_metrics.text(0.5, 0.5, matrics_text, fontsize=20, color='black', ha='right', va='center', wrap=True)
        ax_metrics.set_xticks([])
        ax_metrics.set_yticks([])
        ax_metrics.set_frame_on(False)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': (4,1)})
        disp.plot(ax=axs[0], cmap=plt.cm.Blues, text_kw={'fontsize': 20}, colorbar=False)
        # axs[0].set_title('Confusion Matrix', fontsize=20)
        axs[0].ax_.xaxis.label.set_size(20)
        axs[0].ax_.yaxis.label.set_size(20)
        axs[0].ax_.tick_params(axis='both', which='major', labelsize=12)
        axs[1].text(0.5, 0.5, matrics_text, fontsize=20, color='black', ha='right', va='center', wrap=True)
        axs[1].set_axis_off()
    
        plt.tight_layout()

    if filename is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f'{filename}.svg'),
            format = 'svg',
            bbox_inches = 'tight'
        ) # dpi=300
        plt.close()
    else:
        plt.show()


import torch
from mk2.utils.data import get_samples_from_dataloader
from mk2.utils.plots import plot_2d_projection, get_class_per_modality_colors
from mk2.utils.plots import kmeans_clustering, plot_confusion_matrix
from mk2.utils.data import get_dataloaders

def make_report(
    logs,
    config,
    save_path: str,
    model,
    train_dataloaders: list[torch.utils.data.DataLoader],
    test_dataloaders: list[torch.utils.data.DataLoader] = None
) -> None:
    """
    Save logs, 2d pca, and accuracy plots
    
    Args:
        save_path (str): Path to model report folder
    """
    num_modalities = len(config.modalities)

    modality_names = [item.name for item in config.modalities]
    plot_training_log(
        logs = logs,
        training_mode = config.training_mode,
        modality_names = modality_names,
        filename = f"{config.model_name}_training_log",
        save_path = save_path
    )

    input_dataloaders = [(train_dataloaders, os.path.join(save_path, 'train'))]  # adjust save_path
    if test_dataloaders is not None:
        input_dataloaders.append((test_dataloaders, os.path.join(save_path, 'test')))

    for (dataloaders, save_path) in input_dataloaders:
        # Generate new dataloader adjusting the drop_last parameter to have all samples
        dataloaders = get_dataloaders(
            [dataloaders[i].dataset for i in range(num_modalities)],
            batch_size = 32,
            drop_last = False,
            shuffle = True
        )

        # Get a list of samples per modality
        x, y = list(zip(*[get_samples_from_dataloader(dataloader, num_samples=None, device=config.device) for dataloader in dataloaders]))

        if config.training_mode == 'semisupervised':
            # Remove unlabeled samples
            x = list(x)
            y = list(y)
            for i in range(num_modalities):
                labeled_mask = torch.tensor([True if label != -1 else False for label in y[i]])
                x[i] = x[i][labeled_mask]
                y[i] = y[i][labeled_mask]

        # Get sample latents
        latents = torch.concat([model.get_latents(x[i], input_modality=i) for i in range(num_modalities)], dim=0).cpu().numpy()

        # Labels
        class_labels = torch.concat(y, dim=0).cpu().numpy()
        modality_labels = torch.concat([torch.full((x[i].shape[0],), i) for i in range(num_modalities)], dim=0).numpy()
        class_per_modality_labels = np.array([cl + mi*config.num_classes for mi, modality_class_labels in enumerate(y) for cl in modality_class_labels.cpu().numpy()])  # Assuming equal number of classes per modality

        class_per_modality_colors = get_class_per_modality_colors(num_modalities=num_modalities, num_classes=config.num_classes)

        class_per_modality_names = [f"{config.modalities[mi].name}_{ci}" for mi in range(num_modalities) for ci in range(config.num_classes)]

        plot_2d_projection(
            technique = 'pca',  # pca, umap
            data = latents,
            labels = class_per_modality_labels,
            labels_colors = class_per_modality_colors,
            labels_names = class_per_modality_names,
            standardize = True,
            filename = f"{config.model_name}_2d_pca",
            save_path = save_path
        )

        # Accuracy

        kmeans_labels = kmeans_clustering(latents, class_labels, config.num_classes)
        plot_confusion_matrix(
            class_labels,
            kmeans_labels,
            filename = f"{config.model_name}_confusion_matrix_kmeans_class",
            save_path = save_path
        )

        kmeans_labels = kmeans_clustering(latents, class_per_modality_labels, n_clusters=num_modalities*config.num_classes)
        plot_confusion_matrix(
            class_per_modality_labels,
            kmeans_labels,
            filename = f"{config.model_name}_confusion_matrix_kmeans_class_per_modality",
            save_path = save_path
        )

        kmeans_out = [kmeans_clustering(latents[modality_labels == i], class_labels[modality_labels == i], config.num_classes) for i in range(num_modalities)]
        kmeans_labels = np.concatenate(kmeans_out)

        for i in range(num_modalities):
            plot_confusion_matrix(
                class_labels[modality_labels == i],
                kmeans_out[i],
                filename = f"{config.model_name}_confusion_matrix_kmeans_class_modality_{i}",
                save_path = save_path
            )

        # Aux classifier - Labels
        class_labels_pred = model.discriminator.predict(torch.tensor(latents).to(config.device)).cpu().numpy()
        aux_modality_class_labels_pred = np.array(list(map(lambda i: class_labels_pred[i] + modality_labels[i]*config.num_classes, range(len(class_labels_pred)))), dtype=np.int32)  # using the modality of origin

        plot_confusion_matrix(
            class_labels,
            class_labels_pred,
            filename = f"{config.model_name}_confusion_matrix_aux_classifier_class",
            save_path = save_path
        )

        plot_confusion_matrix(
            class_per_modality_labels,
            aux_modality_class_labels_pred,
            filename = f"{config.model_name}_confusion_matrix_aux_classifier_class_per_modality",
            save_path = save_path
        )  # equivalent to the previous confusion matrix

        for i in range(num_modalities):
            plot_confusion_matrix(
                class_labels[modality_labels == i],
                class_labels_pred[modality_labels == i],
                filename = f"{config.model_name}_confusion_matrix_aux_classifier_class_modality_{i}",
                save_path = save_path
            )
