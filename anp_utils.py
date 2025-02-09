import configparser
import os
import numpy as np
import pandas as pd
import scipy.linalg
from SPARQLWrapper import SPARQLWrapper, JSON
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
import matplotlib.colors as mcolors
import contextily as ctx


def load_config(file_path: str):
    """
    Load and parse the configuration file.

    :param file_path: Path to the configuration file.
    :return: ConfigParser object.
    """

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(file_path)
    root_directory = os.path.dirname(os.path.abspath(__file__))
    config.set('paths', 'root', root_directory)

    return config


def get_metrics(
        endpoint: str
) -> pd.DataFrame:

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery("""PREFIX nemo:<http://www.ebikecityevaluationtool.com/ontology/nemo#>
    PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    SELECT ?metric_type ?method ?thematic_metric ?criteria_type ?representation_feature ?measurement_scale
    WHERE {GRAPH <http://www.ebikecityevaluationtool.com/ontology/nemo/metrics/> { 
    ?metric rdf:type ?metric_type.
    ?metric nemo:usedIn ?method.
    FILTER REGEX(STR(?metric_type),'nemo')
    ?metric_type rdfs:subClassOf ?thematic_metric .
    OPTIONAL{?metric nemo:mapsToFeature/rdf:type ?representation_feature.}
    OPTIONAL{ ?metric nemo:measures ?criteria.
    ?criteria rdf:type ?criteria_type.}
    OPTIONAL {?metric nemo:hasMeasurementScale ?measurement_scale.}}}""")
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    qr = pd.DataFrame(results['results']['bindings'])
    qr = qr.applymap(lambda cell: cell if pd.isnull(cell) else cell['value'])
    prefix = 'http://www.ebikecityevaluationtool.com/ontology/nemo#'

    qr['metric_type'] = qr['metric_type'].str.replace(prefix, '')
    qr['thematic_metric'] = qr['thematic_metric'].str.replace(prefix, '')
    qr['criteria_type'] = qr['criteria_type'].str.replace(prefix, '')
    qr['representation_feature'] = qr['representation_feature'].str.replace(prefix, '')
    qr['measurement_scale'] = qr['measurement_scale'].str.replace(prefix, '')
    qr = qr[qr['representation_feature'] != 'RepresentationFeature']
    qr = qr.sort_values(by='thematic_metric')

    return qr


def filter_metrics(
        metrics: pd.DataFrame,
        occurrence: int,
        remove_columns: list
) -> pd.DataFrame:

    filtered_metrics = metrics[~metrics['criteria_type'].isna()]
    metric_n = filtered_metrics['metric_type'].value_counts()
    filtered_metrics = filtered_metrics[filtered_metrics['metric_type'].isin(metric_n[metric_n >= occurrence].index)]
    filtered_metrics = filtered_metrics[~filtered_metrics['metric_type'].str.contains('Perceived')]
    filtered_metrics = filtered_metrics[~filtered_metrics['metric_type'].isin(remove_columns)]

    return filtered_metrics


def calculate_priority_vector(
        matrix: np.ndarray
) -> np.ndarray:

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    principal_eigvec = np.abs(eigenvectors[:, np.argmax(eigenvalues)])

    return principal_eigvec / principal_eigvec.sum()


def calculate_limit_matrix(
        matrix: pd.DataFrame,
        row_col_names: list,
        max_iter: int = 500,
        tol: float = 1e-6
) -> ndarray | DataFrame:

    prev_matrix = matrix.copy()

    for i in range(max_iter):
        next_matrix = np.dot(prev_matrix, matrix)
        if np.linalg.norm(next_matrix - prev_matrix, ord='fro') < tol:
            print(f"Converged in {i + 1} iterations.")

            return pd.DataFrame(next_matrix, index=row_col_names, columns=row_col_names)
        prev_matrix = next_matrix
    print("WARNING: Limit matrix did not fully converge.")

    return pd.DataFrame(next_matrix, index=row_col_names, columns=row_col_names)


def calculate_consistency_ratio(matrix):
    def get_saaty_ri(n):
        SAATY_RI = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        return SAATY_RI.get(n, 1.98 * (n - 2) / n)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for consistency ratio calculation.")

    eigenvalues, _ = scipy.linalg.eig(matrix)
    lambda_max = max(eigenvalues.real)
    n = matrix.shape[0]

    CI = (lambda_max - n) / (n - 1)

    if abs(CI) < 1e-10:
        CI = 0

    RI = get_saaty_ri(n)
    CR = CI / RI if RI != 0 else 0

    if abs(CR) < 1e-10:
        CR = 0

    #print(f"λ_max: {lambda_max}, CI: {CI}, RI: {RI}, CR: {CR}")

    return CR, lambda_max, CI, RI


def calculate_criteria_metric_interaction_matrix(
        metrics: pd.DataFrame,
        group_col: str,
        target_col: str
) -> pd.DataFrame:

    metric_frequency = metrics.groupby([group_col, target_col]).size().unstack(fill_value=0)
    max_freq = metric_frequency.max().max()
    min_freq = metric_frequency.min().min()
    metric_frequency = 1 + 8 * (metric_frequency - min_freq) / (max_freq - min_freq)

    criteria_to_metric = {}
    for criterion in metric_frequency.columns:
        frequencies = metric_frequency[criterion]
        n = len(frequencies)

        if n < 2:
            priority_vector = np.ones(n) / n
        else:
            matrix = np.ones((n, n))

            for i in range(n):
                for j in range(n):
                    if i != j and frequencies[i] > 0 and frequencies[j] > 0:
                        matrix[i, j] = frequencies[i] / frequencies[j]

            #print(f'Pairwise metric matrix under {criterion}') # uncomment if you want to check Consistency Ratio
            CR, lambda_max, CI, RI = calculate_consistency_ratio(matrix)
            if CR > 0.1:
                raise ValueError(f"Inconsistent PCM (CR = {CR:.2f}). Adjust frequency scaling.")

            priority_vector = calculate_priority_vector(matrix)

        criteria_to_metric[criterion] = priority_vector

    return pd.DataFrame(criteria_to_metric, index=metric_frequency.index)


def calculate_pairwise_comparison(
    metrics: pd.DataFrame,
    group_by_column: str,
    target_column: str
) -> tuple[np.ndarray, list]:

    freq_dict = metrics[group_by_column].value_counts().to_dict()
    min_freq = min(freq_dict.values())
    max_freq = max(freq_dict.values())
    freq_dict = {
        k: 1 + 8 * (v - min_freq) / (max_freq - min_freq) if max_freq != min_freq else 1
        for k, v in freq_dict.items()
    }

    log_freqs = {k: np.log1p(v) for k, v in freq_dict.items()}
    grouped_data = metrics.groupby(group_by_column)[target_column].apply(set).to_dict()
    sorted_keys = sorted(grouped_data.keys())
    n = len(sorted_keys)
    pcm_matrix = np.ones((n, n))

    for i, key1 in enumerate(sorted_keys):
        for j, key2 in enumerate(sorted_keys):
            if i != j:
                freq_ratio = log_freqs.get(key1, 1) / log_freqs.get(key2, 1)
                pcm_matrix[i, j] = freq_ratio
                pcm_matrix[j, i] = 1 / freq_ratio

    CR, lambda_max, CI, RI = calculate_consistency_ratio(pcm_matrix)

    if CR > 0.1:
        raise ValueError(f"Inconsistent PCM (CR = {CR:.2f}). Adjust frequency scaling.")

    return pcm_matrix, sorted_keys


def perform_anp_bikeability_evaluation(
        edges: pd.DataFrame,
        criteria_matrix: np.array,
        criteria_keys: list,
        metric_matrix: np.array,
        metric_keys: list,
        criteria_to_metric: np.array,
        metric_to_criteria: np.array,
        metrics: pd.DataFrame
) -> tuple[np.array, pd.DataFrame]:

    edges_mcda = edges[list(metrics['metric_type'].unique())]
    edges_mcda_norm = edges_mcda.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    edges_mcda_norm = edges_mcda_norm[metric_keys]

    invert_columns = ["AirPolutantConcentration", "MotorisedVehicleCount", "SpeedLimit", 'CarLaneCount',
                      "MotorisedTrafficSpeed", "Slope", 'DistanceToTransitFacility', 'BetweenessCentrality',
                      'NodeDegree']
    invert_columns = list(set(invert_columns).intersection(metrics['metric_type'].unique()))

    edges_mcda_norm[invert_columns] = 1 - edges_mcda_norm[invert_columns]

    n = len(criteria_keys)
    m = len(metric_keys)
    r = len(edges_mcda_norm)
    supermatrix = np.zeros((n + m + r, n + m + r))

    edge_row_names = [f"{i}" for i in edges_mcda_norm.index]
    row_col_names = list(criteria_keys) + list(metric_keys) + edge_row_names

    supermatrix[:n, :n] = criteria_matrix
    supermatrix[n:n + m, :n] = criteria_to_metric
    supermatrix[:n, n:n + m] = metric_to_criteria
    supermatrix[n:n + m, n:n + m] = metric_matrix
    supermatrix[n + m:n + m + r, n:n + m] = edges_mcda_norm.values
    supermatrix[n + m:n + m + r, n + m:n + m + r] = np.identity(r)

    supermatrix_df = pd.DataFrame(supermatrix, index=row_col_names, columns=row_col_names)
    norm_supermatrix_df = supermatrix_df.div(supermatrix_df.sum(axis=0, skipna=True), axis=1)
    norm_supermatrix_df.fillna(0, inplace=True)

    limit_matrix_df = calculate_limit_matrix(norm_supermatrix_df, row_col_names)

    edge_rankings = limit_matrix_df.loc[edge_row_names, :].iloc[:, :n + m].mean(axis=1)
    edge_rankings /= edge_rankings.sum()
    edge_rankings.index = edges.index
    bikeability_index = edge_rankings.fillna(0)

    return bikeability_index, limit_matrix_df


def get_edge_ranking(edges, metrics):

    criteria_matrix, criteria_keys = calculate_pairwise_comparison(metrics, 'criteria_type', 'metric_type')
    metric_matrix, metric_keys = calculate_pairwise_comparison(metrics, 'metric_type', 'criteria_type')
    criteria_to_metric = calculate_criteria_metric_interaction_matrix(metrics, 'metric_type', 'criteria_type')
    metric_to_criteria = calculate_criteria_metric_interaction_matrix(metrics, 'criteria_type', 'metric_type')

    edge_rankings, limit_matrix_df = perform_anp_bikeability_evaluation(edges,
                                                                        criteria_matrix,
                                                                        criteria_keys,
                                                                        metric_matrix,
                                                                        metric_keys,
                                                                        criteria_to_metric,
                                                                        metric_to_criteria,
                                                                        metrics)
    return edge_rankings, limit_matrix_df


def permutate_dropped_elements(
        metrics: pd.DataFrame,
        edges: pd.DataFrame,
        element_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = {}

    for i in metrics[element_col].unique():
        shuffled_metrics = metrics[metrics[element_col] != i].copy()
        edge_rankings, limit_matrix_df = get_edge_ranking(edges, shuffled_metrics)
        indexes[i] = edge_rankings
        bi_col = f"BI dropped {i}"
        edges[bi_col] = edge_rankings

    rankings_df = pd.DataFrame(indexes)

    return rankings_df, edges


# Visualization

def save_plot(fig, directory, filename):
    os.makedirs(directory, exist_ok=True)
    fig.savefig(os.path.join(directory, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_priority_weights(df, criteria_keys, metric_keys, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title, color in zip(
            axes,
            [df.iloc[:len(criteria_keys), 0], df.iloc[len(criteria_keys):len(criteria_keys) + len(metric_keys), 0]],
            ["Criteria Priorities", "Metric Priorities"],
            ["#3950A1", "#BB1526"]
    ):
        data.sort_values().plot(kind="barh", ax=ax, color=color, alpha=0.8)
        ax.set(title=title, xlabel="Priority Weight")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_plot(fig, save_dir, "priority_weights.png")


def plot_bikeability_map(edges, zurich_boundary, bi_col, save_dir, crs=2056):
    edges_plot = edges.copy()
    edges_plot[bi_col] = edges_plot[bi_col].clip(upper=edges_plot[bi_col].quantile(0.95))

    norm = mcolors.Normalize(vmin=edges_plot[bi_col].min(), vmax=edges_plot[bi_col].max())
    fig, ax = plt.subplots(figsize=(6, 6))
    edges_plot.plot(ax=ax, column=bi_col, cmap="RdYlBu", linewidth=1.2, alpha=0.9, norm=norm, legend=False)
    zurich_boundary.plot(ax=ax, edgecolor='black', lw=1.8, linestyle="dashed", facecolor='none')
    #ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12, crs=crs)
    sm = plt.cm.ScalarMappable(cmap="RdYlBu", norm=norm)
    sm._A = []
    cbar_ax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.05, ax.get_position().width, 0.02])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"ANP {bi_col}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    save_plot(fig, save_dir, "bikeability_map.png")

    plt.show()


def plot_permutations(original_bikeability, rankings_df1, rankings_df2, save_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, rankings_df, title in zip(
            axes,

            [rankings_df1, rankings_df2],
            ['Metric Type', 'Criteria Type']):

        ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='black', zorder=0)
        ax.errorbar(original_bikeability, rankings_df.mean(axis=1),
                    yerr=rankings_df.std(axis=1) * 2, fmt='o', color="#FED687", label='Mean BI', zorder=1)
        ax.scatter(original_bikeability, rankings_df.max(axis=1), marker='^', color='#3950A1', label='Max BI', zorder=2)
        ax.scatter(original_bikeability, rankings_df.min(axis=1), marker='v', color='#BB1526', label='Min BI', zorder=3)

        ax.set_xlim(0, rankings_df.max().max() * 1.15)
        ax.set_ylim(0, rankings_df.max().max() * 1.15)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=14)

    fig.text(0.00, 0.5, 'Permuted Bikeability Values', va='center', ha='center', rotation='vertical', fontsize=16,
             transform=fig.transFigure)
    axes[-1].set_xlabel('Original Bikeability Values', fontsize=16)
    plt.tight_layout()
    save_plot(fig, save_dir, "permutations.png")

