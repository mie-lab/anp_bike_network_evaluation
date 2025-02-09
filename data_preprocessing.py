import networkx as nx
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from sklearn.neighbors import KernelDensity
import rasterio.features
import pandas as pd
from typing import Union
from scipy.stats import entropy


def calculate_buffer(
        df: gpd.GeoDataFrame,
        buffer_size: float
) -> gpd.GeoDataFrame:

    buff_df = df.copy()[['geometry', 'length', 'index']]
    buff_df['geometry'] = buff_df.buffer(buffer_size)
    buff_df['buff_area'] = buff_df.area

    return buff_df


def merge_spatial_boolean(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        threshold_col: str,
        merge_col: str = "index",
        threshold: float = 0
) -> gpd.GeoDataFrame:

    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    overlap_sums = overlaps.groupby(merge_col)['overlap_length'].sum().reset_index(name='overlap_length_sum')
    edges = edges.merge(overlap_sums, on=merge_col, how='left')

    # Fill NaN values in overlap_length_sum with 0 (no overlap)
    edges['overlap_length_sum'] = edges['overlap_length_sum'].fillna(0)
    edges[target_col] = (edges['overlap_length_sum'] / edges[threshold_col]) * 100 > threshold
    edges = edges.drop(columns=['overlap_length_sum'])

    return edges


def merge_spatial_attribute(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        attribute_cols: Union[str, list],
        target_cols: Union[str, list] = None,
        merge_col: str = "index"
) -> gpd.GeoDataFrame:

    # if only one column
    if not isinstance(attribute_cols, list):
        attribute_cols = [attribute_cols]

    if target_cols is None:
        target_cols = attribute_cols

    if not isinstance(target_cols, list):
        target_cols = [target_cols]

    # Perform spatial overlay
    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    idx = overlaps.groupby(merge_col)['overlap_length'].idxmax()
    max_overlaps = overlaps.loc[idx]

    merge_cols = [merge_col] + attribute_cols
    edges = edges.merge(max_overlaps[merge_cols], on=merge_col, how="left")
    rename_dict = dict(zip(attribute_cols, target_cols))
    edges = edges.rename(columns=rename_dict)

    return edges


def calculate_count(
        df: gpd.GeoDataFrame,
        other_df: gpd.GeoDataFrame,
        grouping_column: str,
        count_column: str = None
) -> gpd.GeoDataFrame:

    overlaps = gpd.sjoin(df, other_df, how='inner', predicate='intersects')

    if count_column:
        result = overlaps.groupby(grouping_column)[count_column].sum()
    else:
        result = overlaps.groupby(grouping_column).size()

    return df[grouping_column].map(result).fillna(0)


def merge_spatial_share(
        edges: gpd.GeoDataFrame,
        buffer: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        divider_col: str,
        percent: bool = False,
        merge_col: str = "index",
) -> gpd.GeoDataFrame:

    overlaps = gpd.overlay(buffer, spatial_data, how="intersection", keep_geom_type=False)

    if overlaps.empty:
        edges[target_col] = 0
        return edges

    if edges.geometry.type[0] == 'LineString':
        overlaps['overlap'] = overlaps.geometry.length
    elif edges.geometry.type[0] == 'Polygon':
        overlaps['overlap'] = overlaps.geometry.area

    overlap_sums = overlaps.groupby(merge_col)['overlap'].sum().rename('overlap_sum')
    edges = edges.merge(overlap_sums, on=merge_col, how='left')

    edges[target_col] = (edges['overlap_sum'] / buffer[divider_col]).fillna(0)
    if percent:
        edges[target_col] *= 100

    edges.drop(columns=['overlap_sum'], inplace=True)

    return edges


def merge_distance_to_nearest(
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        merge_col: str = "index",
        how: str = "intersection"):

    nearest_stops = gpd.sjoin_nearest(edges, spatial_data, how=how, distance_col=target_col)
    edges = edges.merge(nearest_stops[[merge_col, target_col]], on=merge_col)

    return edges


def merge_spatial_count(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        agg_col: str = None,
        agg_func: str = "size",
        merge_col: str = "index",
) -> gpd.GeoDataFrame:
    """

    :rtype: object
    """
    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)

    if agg_func in ['size', 'sum', 'min', 'max']:
        aggregation = overlaps.groupby(merge_col)[agg_col].agg(agg_func) if agg_col else overlaps.groupby(
            merge_col).agg(agg_func)
    else:
        aggregation = overlaps.groupby(merge_col).apply(agg_func)

    aggregation_aligned = buff_edges[merge_col].map(aggregation).fillna(0)
    edges[target_col] = aggregation_aligned

    return edges





def compute_air_pollution_heatmap(gdf, column, boundary_gdf, resolution=300, bandwidth=500):
    """
    Compute a KDE-based density grid for NO2 concentrations and return a GeoDataFrame.

    Parameters:
    - gdf: GeoDataFrame with point data and NO2 values.
    - column: Column name in `gdf` for the NO2 values.
    - boundary_gdf: GeoDataFrame representing the boundary for masking.
    - resolution: Grid resolution for the KDE (default: 300).
    - bandwidth: Bandwidth for KDE (default: 500).

    Returns:
    - heatmap_gdf: GeoDataFrame with polygons representing grid cells and a density attribute.
    """
    # Calculate bounds
    x_min, y_min, x_max, y_max = boundary_gdf.total_bounds
    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    values = gdf[column].values

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    grid_coords = np.c_[xx.ravel(), yy.ravel()]

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(coords, sample_weight=values)
    log_density = kde.score_samples(grid_coords)
    density = np.exp(log_density).reshape(xx.shape)

    # Rescale density to match original NO2 values
    density *= gdf[column].sum()  # Scale by total NO2 values
    density *= (gdf[column].max() / density.max())  # Normalize to the max NO2 value

    # Mask using boundary
    mask = np.array([
        boundary_gdf.contains(Point(x, y)).any()
        for x, y in grid_coords
    ]).reshape(xx.shape)
    density[~mask] = np.nan
    density = density.astype('float32')

    # Create polygons
    transform = rasterio.transform.from_bounds(
        x_min, y_max,
        x_max, y_min,
        density.shape[1], density.shape[0]
    )

    shapes = rasterio.features.shapes(density, transform=transform)

    polygons = []
    values = []
    for geom, value in shapes:
        if not np.isnan(value):
            polygons.append(shape(geom))
            values.append(value)

    heatmap_gdf = gpd.GeoDataFrame({column: values}, geometry=polygons, crs=boundary_gdf.crs)

    return heatmap_gdf


def count_characters(value, chars):
    if pd.isna(value):
        return 0  # Return 0 if the value is NaN
    return sum([1 for char in value if char in chars])


def calculate_land_use_mix(buffer, edges, landuse, landuse_col):
    def calculate_shannon_entropy(proportions):
        return entropy(proportions, base=np.e)

    intersected = gpd.overlay(landuse, buffer, how='intersection')
    intersected['area'] = intersected.geometry.area
    land_use_areas = intersected.groupby(['index', 'typ'])['area'].sum().reset_index()
    total_area = land_use_areas.groupby('index')['area'].sum().rename('total_area')
    land_use_areas = land_use_areas.join(total_area, on='index')
    land_use_areas['proportion'] = land_use_areas['area'] / land_use_areas['total_area']

    edges_entropy = land_use_areas.groupby('index')['proportion'].apply(calculate_shannon_entropy).reset_index()
    edges_entropy = edges_entropy .rename(columns={'proportion': landuse_col})
    edges = edges.merge(edges_entropy , on='index', how='left')
    edges[landuse_col] = edges[landuse_col].fillna(0)

    return edges


def calculate_edge_betweenness(edges, target_col, normalize=True):

    G = nx.from_pandas_edgelist(edges, 'u', 'v')
    edge_btw = nx.edge_betweenness_centrality(G, normalized=normalize)
    edge_btw_df = pd.DataFrame([
        {'u': u, 'v': v, target_col: bc}
        for (u, v), bc in edge_btw.items()
    ])

    edges = edges.merge(edge_btw_df, on=['u', 'v'], how='left')

    return edges


def calculate_intersection_density(edges, buffer, intersection_col):

    all_points = edges['geometry'].apply(lambda geom: [Point(geom.coords[0]), Point(geom.coords[-1])]).explode()
    unique_points = gpd.GeoSeries(pd.unique(all_points), crs=edges.crs)
    intersections = gpd.GeoDataFrame(geometry=unique_points)
    edges = merge_spatial_count(buffer, edges, intersections, intersection_col)

    return edges


def calculate_bikelane_density(edges, buffer, bike_lanes, bikelane_col):

    overlaps = gpd.overlay(buffer, bike_lanes, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    overlap_sums = overlaps.groupby('index')['overlap_length'].sum().rename('overlap_length_sum')
    edges = edges.merge(overlap_sums, on='index', how='left')
    edges[bikelane_col] = (edges['overlap_length_sum'] / buffer['buff_area']).fillna(0)

    return edges


def calculate_node_degrees(edges, target_col):

    G = nx.from_pandas_edgelist(edges, 'u', 'v')
    degree_dict = dict(G.degree())
    degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['degree']).reset_index().rename(
        columns={'index': 'node'})
    edges = edges.merge(degree_df, left_on='u', right_on='node', how='left').rename(
        columns={'degree': 'u_degree'}).drop(columns=['node'])
    edges = edges.merge(degree_df, left_on='v', right_on='node', how='left').rename(
        columns={'degree': 'v_degree'}).drop(columns=['node'])
    edges[target_col] = edges[['u_degree', 'v_degree']].mean(axis=1)

    return edges


def assign_points(series, bins):
    """Assign points based on thresholds."""
    return np.digitize(series, bins, right=True).clip(1, 5)


def average_pop(group):
    """Calculate average population."""
    return group['PERS_N'].sum() / len(group)


def avg_no2(group):
    """Calculate average no2 emissions"""
    return group['no2'].sum() / len(group)