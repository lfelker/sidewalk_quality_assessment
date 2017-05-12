import json
import os
import click
import geopandas as gpd
from geopandas import GeoSeries
import shapely
from shapely import ops

# for visualizations
import matplotlib.pyplot as plt
import matplotlib
import fiona
import pandas as pd

import networkx as nx

import osmnx as ox

import xml.etree.ElementTree

crs_utm = {'init': 'epsg:26910'}
crs_merc = {'init': 'epsg:4326'}

# combination of all of these should get us matching lines.
# in theory each one doesn't have to be super precise since
# for a match the two lines need similar length, slope, and at least one endpoint
LEN_COMP_THRESH = 0.3 # length fraction that signifies a segment too short compared to other line lengthto be matched
LEN_CHECK_THRESH = 8 # any segment less than this length is stripped out of matching attempt (gets rid of links)
END_DIST = 12 # segments need at least one end point within this distance of each other
SLOPE_DIFFERENCE = 0.5 # allowed difference for a match when absolute value of one slope is less than 5
HIGH_SLOPE_CUTOFF = 5 # slopes higher than this are assumed to be the same
SWK_BUFFER = 0.4572 # this is in meters (1.5 feet) this extends sidewalks to 3 foot width

def main():
	click.echo("Loading Data")

	streets = gpd.read_file('./data/sdot/streets.shp').to_crs(crs_utm)

	# NEIGHBORHOOD PREP
	neighborhoods = gpd.read_file('./data/neighborhoods/Neighborhoods.shp')
	neighborhoods = neighborhoods.to_crs(crs_utm)
	neighborhood_name = 'University District'
	neighborhood_escname = neighborhood_name.replace(' ', '_')
	mask = neighborhoods['S_HOOD'] == neighborhood_name
	udistrict = neighborhoods.loc[mask, 'geometry'].iloc[0]

	# STREET POLYGON FOR CHECKING SWK OVERLAP
	streets_poly_gdf = gpd.read_file('./data/seattle_street_polygon.shp').to_crs(crs_utm)
	streets_poly_gdf = clip_data(streets_poly_gdf, udistrict)

	# OPEN STREET MAP DATA PREP
	click.echo("Preparing OSM Data")
	osm_gdf = json_to_gdf("./data/osm/udistrict.geojson")
	osm_gdf = osm_gdf.loc[osm_gdf['geometry'].geom_type == 'LineString']
	osm_gdf_copy = osm_gdf.copy()

	# SDOT PREP
	click.echo("Preparing SDOT Data")
	sdot_json = open("./data/sdot/SDOT_swk.json").read()
	sdot_json = json.loads(sdot_json)
	sdot_data = sdot_json['data']
	sdot_geo = convert_to_geo(sdot_data)
	sdot_gdf = gpd.GeoDataFrame.from_features(sdot_geo['features'])
	sdot_gdf.crs = crs_merc
	sdot_gdf = sdot_gdf.to_crs(crs_utm)
	sdot_gdf = clip_data(sdot_gdf, udistrict)

	# ACCESS MAP PREP
	click.echo("Preparing Access Map Data")
	am_gdf = json_to_gdf("./data/access_map/sidewalks.geojson")
	am_gdf = clip_data(am_gdf, udistrict)

	# To Avoid Data Prep
	# am_gdf.to_file("./data/am_gdf.shp")
	# sdot_gdf.to_file("./data/sdot_gdf.shp")
	# osm_gdf.to_file("./data/osm_gdf.shp")

	# SPLIT BLOCKS
	click.echo("Polyganizing Blocks")
	blocks = blocks_subtasks(streets)
	blocks = filter_blocks_by_poly(blocks, udistrict)

	layers = {
		'osm_gdf': osm_gdf,
		'am_gdf': am_gdf,
		'sdot_gdf': sdot_gdf
	}

	# mappings between datasets
	osm_to_other = []
	sdot_no_matches = []
	am_no_matches = []

	# PROCESS DATA BLOCK BY BLOCK
	blocks_gdfs = split_geometry_into_tasks(layers, blocks)

	for block_num in blocks_gdfs:
		osm_gdf = blocks_gdfs[block_num]['osm_gdf']
		if len(osm_gdf) > 0: # only examine blocks that have osm data
			click.echo("preparing block number: " + str(block_num))

			# EXTRACTING DATA FOR CURRENT BLOCK
			osm_gdf_filtered = osm_gdf.loc[osm_gdf['geometry'].length > LEN_CHECK_THRESH]
			sdot_gdf = blocks_gdfs[block_num]['sdot_gdf']
			am_gdf = blocks_gdfs[block_num]['am_gdf']


			# FIND STREET SIDEWALK OVERLAP BY BLOCK
			curr_block = blocks_gdfs[block_num]['polygon']
			streets_poly_gdf_copy = streets_poly_gdf.copy()
			overlap_results = overlap_computations(osm_gdf_filtered, sdot_gdf, am_gdf, curr_block, streets_poly_gdf)
			blocks_gdfs[block_num]['overlap_results'] = overlap_results
			

			# IN PROGRESS: FINDING MAPPINGS BETWEEN DATASETS
			# precalculate slope of lines for one to one.
			am_gdf['slope'] = am_gdf['geometry'].apply(slope)
			osm_gdf_filtered['slope'] = osm_gdf_filtered['geometry'].apply(slope)
			sdot_gdf['slope'] = sdot_gdf['geometry'].apply(slope)

			# FINDING MAPINGS TO OSM
			for osm_row in osm_gdf_filtered.iterrows():
				if type(osm_row[1]['geometry']) == shapely.geometry.LineString:
					osm_to_other.append(find_matches(osm_row, sdot_gdf, am_gdf, curr_block))

			# FINDING SDOT WITHOUT MAPPING
			for sdot_row in sdot_gdf.iterrows():
				if not is_osm_match(sdot_row, osm_gdf_filtered):
					sdot_no_matches.append(sdot_row[1]['geometry'])

			# FINDING AM WITHOUT MAPPING
			for am_row in am_gdf.iterrows():
				if not is_osm_match(am_row, osm_gdf_filtered): # we have no corresponding ground truth sidewalk
					am_no_matches.append(am_row[1]['geometry'])

			graph_results = graph_computations(osm_gdf, osm_gdf_filtered, sdot_gdf, am_gdf)
			blocks_gdfs[block_num]['graph_results'] = graph_results

	# FOR VISUALIZING DATA MAPPINGS
	# osm_to_other_df = pd.DataFrame(osm_to_other)
	# print(osm_to_other_df.head())
	# sdot_no_matches = gpd.GeoDataFrame(geometry=sdot_no_matches)
	# am_no_matches 	= gpd.GeoDataFrame(geometry=am_no_matches)
	# print(str(len(sdot_no_matches)) + " " + str(len(am_no_matches)))
	# visualize(osm_gdf_copy, buff=udistrict, title="No Matches for SDOT", extras=[sdot_no_matches])
	# visualize(osm_gdf_copy, buff=udistrict, title="No Matches for AM", extras=[am_no_matches])
	# sdot_no_matches.to_file('./data/generated/sdot_no_matches.shp')
	# am_no_matches.to_file('./data/generated/am_no_matches.shp')
	# sdot_no_matches = gpd.read_file('./data/generated/sdot_no_matches.shp')
	# am_no_matches =  gpd.read_file('./data/generated/am_no_matches.shp')
	# visualize(osm_gdf_copy, buff=udistrict, title="No Matches for SDOT", extras=[sdot_no_matches])
	# visualize(osm_gdf_copy, buff=udistrict, title="No Matches for AM", extras=[am_no_matches])

	click.echo('PREPARING OUTPUT CSV')
	output_to_csv(blocks_gdfs, "./data/output.csv")

# creates a series from the given data with indexes i
def s(data, i):
	return pd.Series(data, index=i)

# for the given three datasets, computes overlap with streets, overlap with 'ground truth' when given buffer of SWK_BUFFER, and overall SWK area when buffered.
def overlap_computations(osm_gdf_filtered, sdot_gdf, am_gdf, curr_block, streets_poly_gdf_copy):
	streets_poly_gdf_copy = streets_poly_gdf_copy.loc[streets_poly_gdf_copy.intersects(curr_block)]
	streets_poly_gdf_copy['geometry'] = streets_poly_gdf_copy.intersection(curr_block)

	# OVERLAP WITH STREETS
	osm_overlap  = calculate_overlap(osm_gdf_filtered, streets_poly_gdf_copy)
	sdot_overlap = calculate_overlap(sdot_gdf, streets_poly_gdf_copy)
	am_overlap   = calculate_overlap(am_gdf, streets_poly_gdf_copy)

	# OVERLAP WITH "GOUND TRUTH" OSM SWKS
	sdot_ground_overlap = calculate_swk_overlap(sdot_gdf, osm_gdf_filtered)
	am_ground_overlap = calculate_swk_overlap(am_gdf, osm_gdf_filtered)
	osm_ground_overlap = sanity_check(osm_gdf_filtered)

	results = {}
	results['osm_ground_overlap'] = sum(osm_ground_overlap.area) if len(osm_ground_overlap) > 0 else 0
	results['sdot_ground_overlap'] = sum(sdot_ground_overlap.area) if len(sdot_ground_overlap) > 0 else 0
	results['am_ground_overlap'] = sum(am_ground_overlap.area) if len(am_ground_overlap) > 0 else 0

	results['osm_overlap'] = sum(osm_overlap.area) if len(osm_overlap) > 0 else 0
	results['sdot_overlap'] = sum(sdot_overlap.area) if len(sdot_overlap) > 0 else 0
	results['am_overlap'] = sum(am_overlap.area) if len(am_overlap) > 0 else 0\

	results['osm_swk_area'] = calculate_sidewalk_area(osm_gdf_filtered)
	results['sdot_swk_area'] = calculate_sidewalk_area(sdot_gdf)
	results['am_swk_area'] = calculate_sidewalk_area(am_gdf)
	return results

# calculates overlap betweeen given gdf and street polygon when the given gdf is buffered by SWK_BUFFER
def calculate_overlap(gdf, streets_poly):
	sidewalks = gdf.geometry.apply(lambda g: g.buffer(SWK_BUFFER, cap_style=2))
	sidewalks = gpd.GeoDataFrame(geometry=sidewalks)
	overalp = gpd.overlay(sidewalks, streets_poly, how="intersection")
	return overalp

# returns overlap of given geodataframe and given "ground" source when each dataset is buffered by SWK_BUFFER
def calculate_swk_overlap(gdf, ground):
	sidewalks = gdf.geometry.apply(lambda g: g.buffer(SWK_BUFFER, cap_style=2))
	sidewalks = gpd.GeoDataFrame(geometry=sidewalks)
	ground = ground.geometry.apply(lambda g: g.buffer(SWK_BUFFER, cap_style=2))
	ground = gpd.GeoDataFrame(geometry=ground)
	overlap = gpd.overlay(sidewalks, ground, how="intersection")
	return overlap

# produces overlap with itself
def sanity_check(ground):
	ground = ground.geometry.apply(lambda g: g.buffer(SWK_BUFFER, cap_style=2))
	ground = gpd.GeoDataFrame(geometry=ground)
	overlap = gpd.overlay(ground, ground, how="intersection")
	return overlap

# sums area of gdf with each segment getting a buffer of SWK_BUFFER
def calculate_sidewalk_area(gdf):
	sidewalks = gdf.geometry.apply(lambda g: g.buffer(SWK_BUFFER, cap_style=2))
	sidewalks = gpd.GeoDataFrame(geometry=sidewalks)
	return sum(sidewalks.area)

# for a given osm_row (line), returns matches in sdot and access map data for that block
def find_matches(osm_row, sdot_gdf, am_gdf, curr_block):
	sdot_matches = []
	for sdot_row in sdot_gdf.iterrows():
		if is_match(osm_row, sdot_row):
			sdot_matches.append(sdot_row[1]['geometry'])

	am_matches = []
	for am_row in am_gdf.iterrows():
		if is_match(osm_row, am_row):
			am_matches.append(am_row[1]['geometry'])		

	data = {
		'osm_line': osm_row[1]['geometry'],
		'sdot_matches': sdot_matches,
		'am_matches': am_matches
	}

	# VISUALIZE MAPPINGS
	# target_gdf = gpd.GeoDataFrame(geometry=[osm_row[1]['geometry']])
	# am_matches_gdf = gpd.GeoDataFrame(geometry=am_matches)
	# sdot_matches_gdf = gpd.GeoDataFrame(geometry=sdot_matches)

	# if len(sdot_matches) > 0:
	# 	visualize(target_gdf, buff=curr_block, title="Matches Found: " + str(len(sdot_matches)), extras=[sdot_matches_gdf])
	# else:
	# 	visualize(target_gdf, buff=curr_block, title="Matches Found: " + str(len(sdot_matches)), extras=[sdot_gdf])

	# if len(am_matches) > 0:
	# 	visualize(target_gdf, buff=curr_block, title="Matches Found: " + str(len(am_matches)), extras=[am_matches_gdf])
	# else:
	# 	visualize(target_gdf, buff=curr_block, title="Matches Found: " + str(len(am_matches)), extras=[am_gdf])

	return data

# returns true if given row has a match to osm data for that block
def is_osm_match(row, osm_gdf_filtered):
	match = False
	for osm_row in osm_gdf_filtered.iterrows():
		if type(osm_row[1]['geometry']) == shapely.geometry.LineString and is_match(osm_row, row):
			match = True
			break
	return match

# returns true if lines are matches
def is_match(row1, row2):
	close_end = compare_ends(row1, row2)
	close_slope = compare_slopes(row1, row2)
	close_length = compare_lengths(row1, row2)
	return close_end and close_slope and close_length

# returns true if length ratio between segments is greater then LEN_COMP_THRESH
def compare_lengths(row1, row2):
	line1 = row1[1]['geometry']
	line2 = row2[1]['geometry']
	if line1.length < line2.length:
		return line1.length / line2.length > LEN_COMP_THRESH
	else:
		return line2.length / line1.length > LEN_COMP_THRESH

# returns true if the lines have an end point match and other end points are within min length of each other
def compare_ends(row1, row2):
	length_min = min(row1[1]['geometry'].length, row2[1]['geometry'].length)

	line1_coords = row1[1]['geometry'].coords
	one_start = shapely.geometry.Point(line1_coords[0])
	one_end = shapely.geometry.Point(line1_coords[len(line1_coords) - 1])

	line2_coords = row2[1]['geometry'].coords
	two_start = shapely.geometry.Point(line2_coords[0])
	two_end = shapely.geometry.Point(line2_coords[len(line2_coords) - 1])

	# if one end point is within END_DIST of other
	# we require other end points must at least be within the min length of segment from each other
	# otherwise the directionality of the ways from those points is oposite.
	# we could use slope here too
	def compare_points(x, y, x_other, y_other):
		if x.distance(y) < END_DIST:
			#  os------oe  ts-----te account for this scenario
			return x_other.distance(y_other) < length_min
		else:
			return False
	a = compare_points(one_start, two_start, one_end, two_end)
	b = compare_points(one_start, two_end, one_end, two_start)
	c = compare_points(one_end, two_start, one_start, two_end)
	d = compare_points(one_end, two_end, one_start, two_start)
	return a or b or c or d

# compares slope of two lines
# returns true if both lines have slope greater than HIGH_SLOPE_CUTOFF
# or if difference in slope is less than SLOPE_DIFFERENCE
def compare_slopes(row1, row2):
	slope1 = row1[1]['slope']
	slope2 = row2[1]['slope']
	if abs(slope1) > HIGH_SLOPE_CUTOFF and abs(slope2) > HIGH_SLOPE_CUTOFF:
		return True
	else:
		return abs(row1[1]['slope'] - row2[1]['slope']) < SLOPE_DIFFERENCE

# account for lines being reversed too. always does slope left to right returns None if x are equal
def slope(line):
	slope = None
	end_one = line.coords[0]
	end_two = line.coords[len(line.coords) - 1]
	if end_one[0] == end_two[0]:
		return None
	elif end_one[0] < end_two[0]:
		return (end_two[1] - end_one[1]) / (end_two[0] - end_one[0])
	else:
		return (end_one[1] - end_two[1]) / (end_one[0] - end_two[0])

# completes graph computations for the given datasets
# computes conected components length of sidewalks and number of segments
def graph_computations(osm_gdf, osm_gdf_filtered, sdot_gdf, am_gdf):
	results = {}

	# COMPARING GRAPH CONNECTIVITY
	# keep links for graph creation, so use non filtered osm_gdf
	G_osm = graph_construct(osm_gdf)
	results['osm_graph_cc'] = len(list(nx.connected_components(G_osm)))
	# use filtered dataset for seg num and len calculation
	results['osm_swk_len'] = sum(osm_gdf_filtered.length)
	results['osm_swk_seg_num'] = len(osm_gdf_filtered)
	# connected_components = len(list(nx.connected_components(G_osm)))
	#visualize(osm_gdf, title=str(block_num) + " cc = " + str(connected_components))

	# CREATE SDOT GRAPH
	# Unary Union Creates Points at Intersection of SDOT Lines
	lines = shapely.ops.unary_union(sdot_gdf["geometry"])
	# convert unaryunion result to only linestrings
	if type(lines) == shapely.geometry.MultiLineString:
		lines = explodeMultiLineStrings(lines)
	elif type(lines) == shapely.geometry.LineString:
		lines = [lines]
	else:
		raise ValueError("unary_union returned non line string data")

	sdot_gdf = gpd.GeoDataFrame(geometry=lines)
	G_sdot = graph_construct(sdot_gdf)
	results['sdot_graph_cc'] = len(list(nx.connected_components(G_sdot)))
	results['sdot_swk_len'] = sum(sdot_gdf.length)
	results['sdot_swk_seg_num'] = len(sdot_gdf)
	# connected_components = len(list(nx.connected_components(G_sdot)))
	#visualize(sdot_gdf, title=str(block_num) + " cc = " + str(connected_components))

	# CREATE ACCESS MAP GRAPH
	G_am = graph_construct(am_gdf)
	results['am_graph_cc'] = len(list(nx.connected_components(G_am)))
	results['am_swk_len'] = sum(am_gdf.length)
	results['am_swk_seg_num'] = len(am_gdf)
	# connected_components = len(list(nx.connected_components(G_am)))
	#visualize(am_gdf, title=str(block_num) + " cc = " + str(connected_components))

	return results

# creates a graph from the linestring data in the given geodataframe
def graph_construct(gdf):
	G = nx.Graph()
	for line in gdf['geometry']:
		# we have curb ramp data, so check for only linestrings
		if type(line) == shapely.geometry.LineString:
			for i in range(1, len(line.coords)):
				G.add_edge(line.coords[i - 1], line.coords[i])
	return G

def explodeMultiLineStrings(lines):
	result = []
	for line in lines:
		result.append(line)
	return result

def split_geometry_into_tasks(layers_gdfs, tasks):
	click.echo('spliting geometires into separate tasks')

	seen_it = {
		'osm_gdf': set(),
		'am_gdf': set(),
		'sdot_gdf': set()
	}

	tasks_gdfs = {}

	for idx, task in tasks.iterrows():
		click.echo('Processed task {} of {}'.format(idx, tasks.shape[0]))
		tasks_gdfs[idx] = {
			'polygon': task.geometry
		}

		for key, value in layers_gdfs.items():
			# FIXME: need to remove redundant data (use poly_id!)
			# Extract
			data = value.loc[value.intersects(task.geometry)].copy()

			# Check the set of IDs we've seen and remove the features if we've
			# already processed them into a task. Add the new IDs to the ID set for
			# this layer.
			data = data.loc[~data.index.isin(seen_it[key])]
			for layer_idx in list(data.index):
				seen_it[key].add(layer_idx)

			tasks_gdfs[idx][key] = data
	click.echo('Done')
	return tasks_gdfs

def blocks_subtasks(streets):
    '''Given a street network as the input, generated polygons that roughly
    correspond to street blocks.

    Returns a GeoDataFrame of those blocks, numbered from 0 to n - 1.

    '''
    # Generate blocks by polygonizing streets. Not perfect, but pretty good.
    polygons = list(ops.polygonize(list(streets.geometry)))
    blocks = gpd.GeoDataFrame(geometry=polygons)
    blocks.crs = streets.crs
    blocks['poly_id'] = blocks.index

    return blocks

# this function isn't used
def blocks_poly_boundary_subtasks(streets, polygon):
    boundary = polygon.boundary
    geoms = list(streets.geometry)
    geoms.append(boundary)
    polygons = list(ops.polygonize(geoms))
    blocks = gpd.GeoDataFrame(geometry=polygons)
    blocks.crs = streets.crs
    blocks['poly_id'] = blocks.index
    new_blocks = blocks.loc[blocks.intersects(polygon)].copy()
    return new_blocks

def filter_blocks_by_poly(blocks, polygon):
    '''Given a GeoDataFrame of polygons (or anything, really), return it
    filtered by that polygon:
    1) Remove block polygons that don't intersect the new polygon
    2) Alter the shape of the block polygons to the intersection between the
       block and the polygon of interest. e.g. trim to the neighborhood.

    '''
    # Initialize the spatial index, if it wasn't already
    blocks.sindex

    # Find the blocks that intersect the polygon (bounding box intersection)
    query = blocks.sindex.intersection(polygon.bounds, objects=True)
    ids = [x.object for x in query]
    bbox_ixn = blocks.loc[ids]

    # Find the blocks that intersect the polygon (actual intersection)
    new_blocks = bbox_ixn.loc[bbox_ixn.intersects(polygon)].copy()

    # Alter the blocks to the shape of the enclosing polygon
    new_blocks['geometry'] = new_blocks.intersection(polygon)

    # Recreate the index + poly_id
    new_blocks.reset_index(drop=True, inplace=True)
    new_blocks['poly_id'] = new_blocks.index

    return new_blocks

# this function isn't used
def isolate_osm_sidewalks(streets):
	osm_gdf = json_to_gdf("./data/osm/roads.geojson")
	footway = osm_gdf.loc[osm_gdf["type"] == "footway"]
	footway = clip_data(footway, udistrict)
	streets = clip_data(streets, udistrict)

	columns_data = []
	geoms = []
	for line in footway["geometry"]:
		inter = False
		for street in streets["geometry"]:
			intersection = line.intersection(street)
			#print(type(intersection))
			if type(intersection) == shapely.geometry.Point:
				print("HERE")
				inter = True
				break
		if not inter:
			geoms.append(line)

	all_intersection = gpd.GeoDataFrame(geometry=geoms)
	all_intersection.to_file('./data/osm/sidewalks.shp')

# converts json file to geodataframe
def json_to_gdf(file_path):
	json_data = open(file_path).read()
	json_data = json.loads(json_data)
	gdf = gpd.GeoDataFrame.from_features(json_data['features'])
	gdf.crs = crs_merc
	gdf = gdf.to_crs(crs_utm)
	return gdf

def clip_data(data, buff):
	intersection = data.loc[data.intersects(buff)]
	return intersection

def plot_buffer(data, buffa):
	cliped_data = clip_data(data, buffa)
	visualize(cliped_data, buff=buffa)

# for visualizing geodataframes
# you can pass a buffer, title, and extra geodataframes to visualize
def visualize(data, buff=None, title=None, extras=[]):
	#plt.ion()
	plot_ref = None
	if buff == None:
		plot_ref = data.plot(color='grey')
	else: 
		buffers = []
		buffers.append(buff)
		plot_ref = GeoSeries(buffers).plot()
		data.plot(ax=plot_ref, color='grey')
	if title != None:
		plt.title(title)
	if len(extras) > 0:
		for layer in extras:
			layer.plot(ax=plot_ref, color='blue')
	plt.show()

# converts SDOT json format to geojson
def convert_to_geo(data):
	geo_j = { 
		"type": "FeatureCollection",
		"features": []
	}

	def fill_element(el_data):

		location = el_data[50][5]
		if 'paths' not in location:
			return None

		feature = {
			"type": "Feature",
			"geometry": {
				"type": "LineString",
				"coordinates": el_data[50][5]["paths"][0]
			},
			"properties": {
				"id": el_data[0]
			}
		}

		return feature

	features = []
	none_count = 0
	for element in data:
		res = fill_element(element)
		if res == None:
			none_count += 1
		else:
			features.append(fill_element(element))
	click.echo("SDOT Features Without Data: " + str(none_count))

	geo_j["features"] = features
	return geo_j

# converts block_gdfs datastructure to a data frame then outputs to given file path
# this is very ugly, but works
def output_to_csv(blocks_gdfs, file_path):
	i = [] # block numbers for dataframe creation
	osm_cc = []
	osm_swk_len = []
	osm_swk_seg_num = []
	osm_overlap = []
	osm_swk_area = []
	osm_ground_overlap = []

	sdot_cc = []
	sdot_swk_len = []
	sdot_swk_seg_num = []
	sdot_overlap = []
	sdot_swk_area = []
	sdot_ground_overlap = []

	am_cc = []
	am_swk_len = []
	am_swk_seg_num = []
	am_overlap = []
	am_swk_area = []
	am_ground_overlap = []

	for block_num in blocks_gdfs:
		osm_gdf = blocks_gdfs[block_num]['osm_gdf']
		if len(osm_gdf) > 0:
			block = blocks_gdfs[block_num]
			graph_results = block['graph_results']
			overlap_results = block['overlap_results']

			i.append(block_num)

			osm_cc.append(graph_results['osm_graph_cc'])
			osm_swk_len.append(graph_results['osm_swk_len'])
			osm_swk_seg_num.append(graph_results['osm_swk_seg_num'])
			osm_overlap.append(overlap_results['osm_overlap'])
			osm_swk_area.append(overlap_results['osm_swk_area'])
			osm_ground_overlap.append(overlap_results['osm_ground_overlap'])

			sdot_cc.append(graph_results['sdot_graph_cc'])
			sdot_swk_len.append(graph_results['sdot_swk_len'])
			sdot_swk_seg_num.append(graph_results['sdot_swk_seg_num'])
			sdot_overlap.append(overlap_results['sdot_overlap'])
			sdot_swk_area.append(overlap_results['sdot_swk_area'])
			sdot_ground_overlap.append(overlap_results['sdot_ground_overlap'])

			am_cc.append(graph_results['am_graph_cc'])
			am_swk_len.append(graph_results['am_swk_len'])
			am_swk_seg_num.append(graph_results['am_swk_seg_num'])
			am_overlap.append(overlap_results['am_overlap'])
			am_swk_area.append(overlap_results['am_swk_area'])
			am_ground_overlap.append(overlap_results['am_ground_overlap'])

	d = {
		'block_num': s(i, i),
		'osm_cc': s(osm_cc, i),
		'osm_swk_len': s(osm_swk_len, i),
		'osm_swk_seg_num': s(osm_swk_seg_num, i),
		'osm_overlap': s(osm_overlap, i),
		'osm_swk_area': s(osm_swk_area, i),
		'osm_ground_overlap': s(osm_ground_overlap, i),

		'sdot_cc': s(sdot_cc, i),
		'sdot_swk_len': s(sdot_swk_len, i),
		'sdot_swk_seg_num': s(sdot_swk_seg_num, i),
		'sdot_overlap': s(sdot_overlap, i),
		'sdot_swk_area': s(sdot_swk_area, i),
		'sdot_ground_overlap': s(sdot_ground_overlap, i),

		'am_cc': s(am_cc, i),
		'am_swk_len': s(am_swk_len, i),
		'am_swk_seg_num': s(am_swk_seg_num, i),
		'am_overlap': s(am_overlap, i),
		'am_swk_area': s(am_swk_area, i),
		'am_ground_overlap': s(am_ground_overlap, i),
	}

	df = pd.DataFrame(d)

	print(df)

	df.to_csv(file_path)

if __name__ == "__main__":
	main()