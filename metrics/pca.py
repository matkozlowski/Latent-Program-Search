from astlib.ast_util import ASTBuilder
from grammars.arithmetic_grammar import ArithmeticGrammar
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker
from autoencoder.autoencoder_model import load_autoencoder
from autoencoder.configs.standard_conf import standard_autoencoder_config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from autoencoder.ast_dataset import ASTDataset
import torch
from typing import List, Literal
from tqdm import tqdm
import pickle



# fit_latent_reps should have shape [fit samples, features]
# plot_latent_reps should have shape [plot samples, features]
# group_labels should have shape [plot samples]
def plot_pca(fit_latent_reps: np.ndarray, plot_latent_reps: np.ndarray, group_labels: np.ndarray | None, plot_points: np.ndarray | None):
    pca = PCA(n_components=2)
    pca.fit(fit_latent_reps)
    reduced = pca.transform(plot_latent_reps)
    
    if group_labels is not None:
        unique_labels = np.unique(group_labels)
        colors = {label: plt.cm.Paired(i) for i, label in enumerate(unique_labels)}
        #names = ['0', 'x', 'x+2', 'x+1', 'x+3']
        names = ['0', 'x', 'x+2', 'x+1', 'x+3', 'x+4', 'x+5']
        # names = ['x', '3x + 2', '4x - 1', '4x - 6', '4x - 12']
        # names = ['0', '3 - x', '3x', 'x + 9', '7 - x']


        legend_elements = [Line2D([0], [0], marker='o', color='w', label=names[label], markerfacecolor=colors[label], markersize=10) for label in colors]

        for i, point in enumerate(reduced):
            plt.scatter(point[0], point[1], color=colors[group_labels[i]])
        
        plt.legend(handles=legend_elements, loc='best')

    else:
        plt.scatter(reduced[:, 0], reduced[:, 1])

    if plot_points is not None:
        reduced = pca.transform(plot_points)
        plt.scatter(reduced[:, 0], reduced[:, 1], marker='x', color='black', s=100, label='New point')

        label_num = 1
        for point in range(len(plot_points)):
            plt.text(reduced[point, 0] + 0.25, reduced[point, 1] + 0.25, str(label_num), fontsize=12, ha='left', va='bottom')
            label_num += 1

    plt.title('Latent Space PCA of Model with Semantic Ladder Loss')
    plt.savefig('figures/pca.png', dpi=200)
    plt.show()




MODEL_PATH = "autoencoder/models/best_model_2.pt"
TEST_PROGRAMS_FILES = ["autoencoder/data/train_tokenizations.txt", "autoencoder/data/train_tokenizations.txt"]
GROUPS_FILE = "metrics/equivalence_classes.pkl"
POINTS_FILE = "cem/latent_reps.pkl"


def get_test_programs(device):
    programs = []
    for prog_file in TEST_PROGRAMS_FILES:
        dataset = ASTDataset(prog_file, 0, device)
        for program_i in range(len(dataset)):
            program = dataset[program_i]
            programs.append(program)
    return programs


if __name__ == '__main__':
    grammar = ArithmeticGrammar()
    ast_builder = ASTBuilder(grammar)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = ArithmeticSyntaxChecker(device, max_depth=2, inf_mask=True)

    config = standard_autoencoder_config
    config.syntax_checker = sc
    config.is_variational = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_autoencoder(MODEL_PATH, config, device)

    test_programs = get_test_programs(device)
    fit_latent_reps = []
    for program in tqdm(test_programs):
        latent_rep, _, _ = model.encoder(program)
        latent_rep = latent_rep.squeeze()
        fit_latent_reps.append(latent_rep.tolist())
    fit_latent_reps = np.asarray(fit_latent_reps)

    with open(GROUPS_FILE, 'rb') as f:
        groups = pickle.load(f)
    
    plot_latent_reps = []
    group_labels = []
    next_label = 0
    for group in tqdm(groups):
        for program in group:
            latent_rep, _, _ = model.encoder(torch.tensor(program, device=device).unsqueeze(dim=0))
            latent_rep = latent_rep.squeeze()
            plot_latent_reps.append(latent_rep.tolist())
            group_labels.append(next_label)
        next_label += 1
    plot_latent_reps = np.asarray(plot_latent_reps)
    group_labels = np.asarray(group_labels)

    # with open(POINTS_FILE, 'rb') as f:
    #     points = pickle.load(f)
    latent_reps = []
    latent_rep, _, _ = model.encoder(torch.tensor([1, 6, 2], device=device).unsqueeze(dim=0))
    latent_reps.append(latent_rep.squeeze())
    latent_rep, _, _ = model.encoder(torch.tensor([1, 3, 9, 6, 2], device=device).unsqueeze(dim=0))
    latent_reps.append(latent_rep.squeeze())
    latent_rep, _, _ = model.encoder(torch.tensor([1, 3, 9, 3, 6, 7, 2], device=device).unsqueeze(dim=0))
    latent_reps.append(latent_rep.squeeze())

    points = []
    for lr in latent_reps:
        points.append(lr.tolist())
    points = np.asarray(points)

    plot_pca(fit_latent_reps, plot_latent_reps, group_labels, points)
