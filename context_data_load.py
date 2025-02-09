import geopandas as gpd
import pandas as pd
from typing import Dict
from shapely.ops import unary_union
from data_preprocessing import compute_air_pollution_heatmap

SPEED_COL = 'temporeg00'
TRAFFIC_COL = 'AADT_all_veh'
HEAVY_VEH_COLS = ['AADT_delivery_veh', 'AADT_truck_veh', 'AADT_articulated_truck_veh']
LANDUSE_COL = 'typ'
SURFACE_COL = 'belagsart'
SLOPE_COL = 'steigung'
POPULATION_COL = 'PERS_N'
TREES_COL = 'hoehe'
AIR_COL = 'no2'
STREET_COL = 'strassentyp'
NOISE_COL = "lre_tag"


def load_speed_limits(path: str) -> gpd.GeoDataFrame:
    """
    Loads and processes speed limit data.

    :param path: Path to the speed limits file (GeoJSON, Shapefile, etc.).
    :return: A GeoDataFrame containing a column with processed speed limit data.
    """

    speed_limits = gpd.read_file(path).explode(index_parts=True)
    speed_limits[SPEED_COL] = speed_limits[SPEED_COL].str.replace(r'^T|N0$|N30$', '', regex=True).astype(int)

    return speed_limits


def load_traffic_volume(path: str, mapping: Dict[str, str]) -> gpd.GeoDataFrame:
    """
    Loads and processes traffic volume data.

    :param path: Path to the traffic volume data file (GeoJSON, Shapefile, etc.).
    :param mapping: A dictionary mapping German column names to English column names.
    :return: A GeoDataFrame with processed traffic volume data.
    """

    traffic_volume = gpd.read_file(path)
    traffic_volume.rename(columns=mapping, inplace=True)

    return traffic_volume


def load_location(path: str):

    location = gpd.read_file(path)

    return location


def load_landuse(path: str, mapping: Dict[str, str]) -> gpd.GeoDataFrame:
    """
    Loads and processes land use data, mapping land use types to desired labels.

    :param path: Path to the land use data file (GeoJSON, Shapefile, etc.).
    :param mapping: A dictionary mapping original land use type values to desired labels.
    :return: A GeoDataFrame containing land use data with a mapped `typ` column.
    """

    landuse = gpd.read_file(path).assign(typ=lambda df: df[LANDUSE_COL].map(mapping))

    return landuse


def load_surface(path: str, layer: str, mapping: Dict[str, str]) -> gpd.GeoDataFrame:
    """
    Loads and processes surface data and maps surface types to specified values.

    :param path: Path to the surface data file (e.g., GeoPackage, Shapefile, etc.).
    :param layer: The layer name within the file to load.
    :param mapping: A dictionary mapping original surface type values to desired labels.
    :return: A GeoDataFrame containing surface data with surface types mapped.
    """

    surface = gpd.read_file(path, layer=layer)
    surface[SURFACE_COL] = surface[SURFACE_COL].map(mapping)

    return surface


def load_slope(path: str, layer: str) -> gpd.GeoDataFrame:

    slope = gpd.read_file(path, layer=layer)

    return slope


def load_housing_units(path: str, layer: str) -> gpd.GeoDataFrame:

    housing = gpd.read_file(path, layer=layer)

    return housing


def load_street_lighting(path: str) -> gpd.GeoDataFrame:

    street_lighting = gpd.read_file(path)

    return street_lighting


def load_population(path: str) -> gpd.GeoDataFrame:
    """
    Loads and processes population data, replacing negative values with zero.

    :param path: Path to the population data file (GeoJSON, Shapefile, etc.).
    :return: A GeoDataFrame containing population data with invalid values replaced by 0.
    """

    population = gpd.read_file(path)
    population[POPULATION_COL].replace(-999.0, 0, inplace=True)

    return population


def load_green_space(path: str) -> gpd.GeoDataFrame:
    """
    Loads and processes green space data, ensuring same geometry types.

    :param path: Path to the green space data file (GeoJSON, Shapefile, etc.).
    :return: A GeoDataFrame containing individual geometries of green spaces.
    """

    green_spaces = gpd.read_file(path).explode(index_parts=True)

    return green_spaces


def load_trees(path: str) -> gpd.GeoDataFrame:
    """
    Loads and processes tree data, creating canopy coverage geometries.

    :param path: Path to the tree data file (GeoJSON, Shapefile, etc.).
    :return: A GeoDataFrame containing buffered tree geometries and their calculated areas.
    """

    trees = gpd.read_file(path)
    trees['geometry'] = trees.buffer(trees[TREES_COL].mean())
    unary_trees = unary_union(trees['geometry'])
    trees = gpd.GeoDataFrame(geometry=list(unary_trees.geoms), crs=trees.crs)
    trees['area'] = trees.area

    return trees


def load_pt_stops(path: str, location: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Loads and processes public transport stop data, filtering to stops within the specified location.

    :param path: Path to the public transport stops file.
    :param location: A GeoDataFrame representing the geographic boundary to filter stops within.
    :return: A GeoDataFrame containing the geometry of public transport stops within the specified location.
    """

    pt_stops = pd.read_csv(path)
    points = gpd.points_from_xy(pt_stops.stop_lon, pt_stops.stop_lat)

    pt_stops = gpd.GeoDataFrame(pt_stops, geometry=points, crs=4326).to_crs(location.crs)
    pt_stops = gpd.sjoin(pt_stops, location, how="inner", predicate='within')
    pt_stops = pt_stops[['geometry']]

    return pt_stops


def load_air_quality(path: str, location: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Loads and processes air quality data, filtering for the latest year and computing a heatmap.

    :param path: Path to the air quality data file (GeoJSON, Shapefile, etc.).
    :param location: A GeoDataFrame representing the geographic boundary to process the data within.
    :return: A GeoDataFrame containing the computed air pollution heatmap for the specified location.
    """

    air_quality = gpd.read_file(path)
    air_quality = air_quality[air_quality['jahr'].isin([air_quality['jahr'].max()])]
    air_quality_df = compute_air_pollution_heatmap(air_quality, AIR_COL, location, resolution=300, bandwidth=500)

    return air_quality_df


def load_noise_pollution(path: str, road_mapping: Dict[str, str]) -> gpd.GeoDataFrame:
    """
    Loads and processes noise pollution data.

    :param path: Path to the noise pollution data file (GeoJSON, Shapefile, etc.).
    :param road_mapping: A dictionary mapping road types to desired labels.
    :return: A GeoDataFrame containing processed noise pollution data.
    """

    noise_pollution = gpd.read_file(path)
    noise_pollution[STREET_COL] = noise_pollution[STREET_COL].map(road_mapping)

    return noise_pollution

