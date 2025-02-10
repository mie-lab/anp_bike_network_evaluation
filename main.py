import osmnx as ox
from shapely.geometry import LineString
import logging

# Custom modules
from context_data_load import *
from data_preprocessing import *
from anp_utils import *
import constants


def enrich_edge_df(
        edges_df: gpd.GeoDataFrame,
        buffer: gpd.GeoDataFrame,
        speed_limits: gpd.GeoDataFrame,
        traffic_volume: gpd.GeoDataFrame,
        green_spaces: gpd.GeoDataFrame,
        air_poll: gpd.GeoDataFrame,
        surface: gpd.GeoDataFrame,
        landuse: gpd.GeoDataFrame,
        population: gpd.GeoDataFrame,
        trees: gpd.GeoDataFrame,
        pt_stops: gpd.GeoDataFrame,
        slope: gpd.GeoDataFrame,
        b_parking: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        bike_speed: int
) -> gpd.GeoDataFrame:
    """
    Enriches road network data with  contextual information as specific metric values.
    Args:
        edges_df: road network GeoDataFrame.
        buffer:  road network GeoDataframe after buffering it.
        speed_limits: DataFrame with Zurich road speed limits.
        traffic_volume: DataFrame with Average Daily Traffic volumes.
        green_spaces: GeoDataFrame with green spaces in Zurich.
        air_poll: air pollution GeoDataFrame after generating Kernel Density.
        surface: GeoDataFrame with surface conditions of Zurich's road network.
        landuse: GeoDataFrame with landuse across Zurich.
        population: GeoDataFrame with population counts in Zurich.
        trees: GeoDataFrame of tree inventory in Zurich.
        pt_stops: GeoDataFrame of Google OT stops in Zurich.
        slope: GeoDataFrame with road slopes across Zurich.
        b_parking: GeoDataFrame with Bike Parking locations in Zurich.
        pois: OSM points of interest.
        bike_speed: default bike speed set for the workflow.

    Returns: GeoDataFrame with edges enriched with contextual data.
    """

    SPEED_COL = 'temporeg00'
    TRAFFIC_COL = 'AADT_all_veh'
    LANDUSE_COL = 'typ'
    SURFACE_COL = 'belagsart'
    SLOPE_COL = 'steigung'
    POP_COL = 'PERS_N'
    AIR_COL = 'no2'
    BIKE_PARKING_COL = "anzahl_pp"
    BIKELANE_WIDTH_COL = 'ln_desc_width_cycling_m'
    BIKELANE_COL = 'ln_desc'

    # GREEN SPACE SHARE
    edges_df = merge_spatial_share(edges_df, buffer, green_spaces, 'GreenSpaceShare', 'length', percent=True)

    # GREENERY PRESENCE
    edges_df['GreeneryPresence'] = np.where(edges_df['GreenSpaceShare'] > 25, 1, 0)

    # TREE COVERAGE
    edges_df = merge_spatial_share(edges_df, buffer, trees, 'TreeCanopyCoverage', 'buff_area', percent=True)

    # TRAFFIC VOLUME
    edges_df = merge_spatial_attribute(buffer, edges_df, traffic_volume, TRAFFIC_COL, 'MotorisedVehicleCount')

    # SPEED LIMIT
    edges_df = merge_spatial_attribute(buffer, edges_df, speed_limits, SPEED_COL, 'SpeedLimit')

    # MOTORIZED TRAFFIC SPEED
    edges_df['MotorisedTrafficSpeed'] = edges_df['SpeedLimit'] * 0.9

    # SLOPE
    edges_df = merge_spatial_attribute(buffer, edges_df, slope, SLOPE_COL, 'Slope')
    edges_df['Slope'] = abs(edges_df['Slope'])

    # SURFACE
    edges_df = merge_spatial_attribute(buffer, edges_df, surface, SURFACE_COL, 'BikeLaneSurfaceCondition').replace(0,
                                                                                                                   np.nan)

    # AIR POLLUTANT CONCENTRATION
    edges_df = merge_spatial_count(buffer, edges_df, air_poll, 'AirPolutantConcentration', agg_col=AIR_COL,
                                   agg_func=avg_no2)

    # POPULATION DENSITY
    edges_df['PopulationDensity'] = calculate_count(buffer, population, 'index', POP_COL) / (buffer['buff_area'] / 1e6)

    # RESIDENTIAL LANDUSE PRESENCE
    residential_landuse = landuse[landuse[LANDUSE_COL] == 'residential']
    edges_df = merge_spatial_boolean(buffer, edges_df, residential_landuse, 'ResidentialAreaPresence', 'length',
                                     threshold=75)
    edges_df['ResidentialAreaPresence'] = np.where(edges_df['ResidentialAreaPresence'] == True, 1, 0)

    # DISTANCE TO TRANIST FACILITY
    edges_df = merge_distance_to_nearest(edges_df, pt_stops, 'DistanceToTransitFacility', merge_col='index', how='left')

    # TRANSIT FACILITY DENSITY
    edges_df['TransitFacilityDensity'] = calculate_count(buffer, pt_stops, 'index') / (buffer['buff_area'] / 1e6)

    # BIKE PARKING COUNT
    edges_df = merge_spatial_count(buffer, edges_df, b_parking, 'BikeParkingDensity', agg_col=BIKE_PARKING_COL,
                                   agg_func="sum")

    # BIKELANE PRESENCE
    edges_df['BikeLanePresence'] = edges_df[BIKELANE_COL].apply(
        lambda x: 1 if pd.notna(x) and any(char in "P" for char in x) else 0)

    # BIKE LANE WIDTH
    edges_df['BikeLaneWidth'] = edges_df.loc[
        (edges_df[BIKELANE_WIDTH_COL] == 0) & (edges_df['BikeLanePresence'] == 1), 'BikeLaneWidth'] = 1.5

    # SINUOSITY
    edges_df['Sinuosity'] = edges_df['length'] / edges_df.geometry.apply(
        lambda geom: LineString([geom.coords[0], geom.coords[-1]]).length)
    edges_df['Linearity'] = edges_df['Sinuosity']

    # CAR LANE COUNT
    edges_df['CarLaneCount'] = edges_df[BIKELANE_COL].apply(lambda x: count_characters(x, "HMT")).replace(0, np.nan)

    # LANDUSE MIX
    edges_df = calculate_land_use_mix(buffer, edges_df, landuse, 'LandUseMix')

    # INTERSECTION DENSITY
    edges_df = calculate_intersection_density(edges_df, buffer, 'IntersectionDensity')

    # BIKE LANE DENSITY
    bike_lanes = edges_df[edges_df[BIKELANE_COL].str.contains("P", na=False)][['geometry', 'length']]
    edges_df = calculate_bikelane_density(edges_df, buffer, bike_lanes, 'BikeLaneDensity')

    # DESTINATION DENSITY
    edges_df['DestinationDensity'] = calculate_count(buffer, pois, 'index') / (buffer['buff_area'] / 1e6)

    # BIKE AND CAR TRAVEL TIME RATIO
    edges_df['length_km'] = edges_df['length'] / 1000
    edges_df['BikeAndCarTravelTimeRatio'] = (edges_df['length_km'] / int(bike_speed)) / (
                edges_df['length_km'] / edges_df['MotorisedTrafficSpeed'])

    # BETWEENESS CENTRALITY
    G = nx.from_pandas_edgelist(edges_df, 'u', 'v', edge_attr=True)
    edges_df['BetweenessCentrality'] = nx.betweenness_centrality(G, normalized=True, k=500)

    # NODE DEGREE
    edges_df = calculate_node_degrees(edges_df, 'NodeDegree')

    return edges_df


def run_anp_workflow(edges_df, spatial_bounds, kg_endpoint, output_dir):
    metrics = get_metrics(kg_endpoint)
    remove_columns = [
        'BuildingCondition', 'ObstaclePresence', 'DistanceToBikeLaneWithoutExclusivess',
        'SafetySupportedIntersectionShare', 'IllegalSideParkingDensity', 'SignageRatio',
        'SidewalkWidth', 'SeparatedBikeLaneDensity', 'BikeLaneType', 'RoadType',
        'BusAndCarTrafficVolumeRatio', 'BikeParkingType', 'OfficialBikeNetworkShare',
        'DetourFactor'
    ]
    metrics = filter_metrics(metrics, occurrence=2, remove_columns=remove_columns)
    logging.info("Metrics from KG instance loaded.")

    # Status Quo bikeability
    criteria_keys = sorted(metrics['criteria_type'].unique())
    metric_keys = sorted(metrics['metric_type'].unique())

    logging.info("Initiating ANP supermatrix construction.")
    edge_rankings, limit_matrix_df = get_edge_ranking(edges_df, metrics)
    edges_df["BI"] = edge_rankings
    logging.info("Limit matrix created.")

    # Plotting
    plot_priority_weights(limit_matrix_df, criteria_keys, metric_keys, output_dir)
    plot_bikeability_map(edges_df, spatial_bounds, "BI", output_dir)

    # Sensitivity Analysis
    logging.info("Starting sensitivity analysis permutations. It might take a couple minutes.")
    rankings_df1, edges_df = permutate_dropped_elements(metrics, edges_df, 'metric_type')
    rankings_df2, edges_df = permutate_dropped_elements(metrics, edges_df, 'criteria_type')

    plot_permutations(edges_df["BI"], rankings_df1, rankings_df2, output_dir)
    logging.info("Analysis results stored in the output directory.")


def main():
    config_file = "config.ini"
    config = load_config(config_file)

    endpoint = config.get('endpoints', 'kg')
    out_dir = config.get('paths', 'out_dir')
    bike_speed = int(config.get('constants', 'bike_speed'))

    edges = gpd.read_file(config.get('paths', 'ebikecity_network'))
    edges['index'] = edges.index
    buffer = calculate_buffer(edges.copy(), int(config.get('constants', 'buffer')))

    logging.info("Loading contextual data. It might take a couple of minutes.")
    zurich_boundary = load_location(config.get('paths', 'zurich_boundary_path'))
    zurich_district = load_location(config.get('paths', 'zurich_districts'))[:1]
    speed_limits = load_speed_limits(config.get('paths', 'speed_limit_path'))
    traffic_volume = load_traffic_volume(config.get('paths', 'traffic_volume_path'), constants.TRAFFIC_VOLUME_MAPPING)
    green_spaces = load_green_space(config.get('paths', 'green_spaces_path'))
    air_poll = load_air_quality(config.get('paths', 'air_pollution_path'), zurich_boundary)
    surface = load_surface(config.get('paths', 'road_network_path'), config.get('paths', 'surface_layer'),
                           constants.SURFACE_MAPPING)
    landuse = load_landuse(config.get('paths', 'landuse_path'), constants.ZONE_MAPPING)
    population = load_population(config.get('paths', 'population_path'))
    trees = load_trees(config.get('paths', 'trees_path'))
    pt_stops = load_pt_stops(config.get('paths', 'pt_stops_path'), zurich_boundary)
    slope = load_slope(config.get('paths', 'road_network_path'), config.get('paths', 'slope_layer'))
    b_parking = gpd.read_file(config.get('paths', 'bike_parking_path'))
    pois = ox.geometries_from_place(config.get('constants', 'place_name'), {"amenity": True})
    pois = pois[pois.geometry.type == 'Point'].to_crs(int(config.get('constants', 'crs')))
    logging.info(" Contextual data loaded")

    # cache sindex
    _, _, _ = trees.sindex, buffer.sindex, edges.sindex

    logging.info("Enriching edge network with contextual data. It might take a couple of minutes.")
    edges = enrich_edge_df(edges, buffer, speed_limits, traffic_volume, green_spaces, air_poll,
                           surface, landuse, population, trees, pt_stops, slope, b_parking, pois, bike_speed)
    # Limit to Oerlikon area
    edges = edges.drop(columns=['index']).fillna(0)
    edges = gpd.sjoin(edges, zurich_district, how="inner", predicate="intersects").drop(columns=["index_right"])

    run_anp_workflow(edges, zurich_district, endpoint, out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
