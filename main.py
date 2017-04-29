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

def main():
	click.echo("Loading Data")
	regenerate = False
	# regenerate = True

	streets = gpd.read_file('./data/sdot/streets.shp').to_crs(crs_utm)

	# NEIGHBORHOOD PREP
	neighborhoods = gpd.read_file('./data/neighborhoods/Neighborhoods.shp')
	neighborhoods = neighborhoods.to_crs(crs_utm)
	neighborhood_name = 'University District'
	neighborhood_escname = neighborhood_name.replace(' ', '_')
	mask = neighborhoods['S_HOOD'] == neighborhood_name
	udistrict = neighborhoods.loc[mask, 'geometry'].iloc[0]

	# OSM PREP
	click.echo("Preparing OSM Data")
	osm_gdf = json_to_gdf("./data/osm/udistrict_sidewalks.geojson")
	osm_gdf_copy = osm_gdf.copy()
	# plot_buffer(osm_gdf, udistrict)

	if regenerate:

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

		# SPLIT BLOCKS
		click.echo("Polyganizing Blocks")
		blocks = blocks_subtasks(streets)
		blocks = filter_blocks_by_poly(blocks, udistrict)
		# visualize(blocks)

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

				osm_gdf_filtered = osm_gdf.loc[osm_gdf['geometry'].length > LEN_CHECK_THRESH]
				osm_gdf_filtered['slope'] = osm_gdf_filtered['geometry'].apply(slope)
				sdot_gdf = blocks_gdfs[block_num]['sdot_gdf']
				sdot_gdf['slope'] = sdot_gdf['geometry'].apply(slope)
				am_gdf = blocks_gdfs[block_num]['am_gdf']
				am_gdf['slope'] = am_gdf['geometry'].apply(slope)
				visualize(osm_gdf_filtered, title=str(block_num), extras=[sdot_gdf])

				# FINDING MAPPINGS TO OSM DATA
				for osm_row in osm_gdf_filtered.iterrows():
					if type(osm_row[1]['geometry']) == shapely.geometry.LineString:

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

						osm_to_other.append(data)

						# VISUALIZE MAPPINGS
						target_gdf = gpd.GeoDataFrame(geometry=[osm_row[1]['geometry']])
						am_matches_gdf = gpd.GeoDataFrame(geometry=am_matches)
						sdot_matches_gdf = gpd.GeoDataFrame(geometry=sdot_matches)

						if len(sdot_matches) > 0:
							visualize(target_gdf, buff=blocks_gdfs[block_num]['polygon'], title="Matches Found: " + str(len(sdot_matches)), extras=[sdot_matches_gdf])
						else:
							visualize(target_gdf, buff=blocks_gdfs[block_num]['polygon'], title="Matches Found: " + str(len(sdot_matches)), extras=[sdot_gdf])

						# if len(am_matches) > 0:
						# 	visualize(target_gdf, buff=blocks_gdfs[block_num]['polygon'], title="Matches Found: " + str(len(am_matches)), extras=[am_matches_gdf])
						# else:
						# 	visualize(target_gdf, buff=blocks_gdfs[block_num]['polygon'], title="Matches Found: " + str(len(am_matches)), extras=[am_gdf])

				# FINDING SDOT WITHOUT MAPPING
				for sdot_row in sdot_gdf.iterrows():
					match = False
					for osm_row in osm_gdf_filtered.iterrows():
						if type(osm_row[1]['geometry']) == shapely.geometry.LineString and is_match(osm_row, sdot_row):
							match = True
							break
					if not match:
						sdot_no_matches.append(sdot_row[1]['geometry'])

				# FINDING AM WITHOUT MAPPING
				for am_row in am_gdf.iterrows():
					match = False
					for osm_row in osm_gdf_filtered.iterrows():
						if type(osm_row[1]['geometry']) == shapely.geometry.LineString and is_match(osm_row, am_row):
							match = True
							break
					if not match: # we have no corresponding ground truth sidewalk
						am_no_matches.append(am_row[1]['geometry'])

				# COMPARING GRAPH CONNECTIVITY
				G_osm = graph_construct(osm_gdf)
				blocks_gdfs[block_num]['osm_graph'] = G_osm
				connected_components = len(list(nx.connected_components(G_osm)))
				#visualize(osm_gdf, title=str(block_num) + " cc = " + str(connected_components))

				# Create SDOT Graph
				# First must add intersection nodes of linestrings in sdot data to simulate connectivity
				lines = shapely.ops.unary_union(sdot_gdf["geometry"])
				if type(lines) == shapely.geometry.MultiLineString:
					lines = explodeMultiLineStrings(lines)
				elif type(lines) == shapely.geometry.LineString:
					lines = [lines]
				else:
					print("THIS HAPPENED!")
				sdot_gdf = gpd.GeoDataFrame(geometry=lines)
				G_sdot = graph_construct(sdot_gdf)
				blocks_gdfs[block_num]['sdot_graph'] = G_sdot
				connected_components = len(list(nx.connected_components(G_sdot)))
				#visualize(sdot_gdf, title=str(block_num) + " cc = " + str(connected_components))

				# Create Access Map Graph
				G_am = graph_construct(am_gdf)
				blocks_gdfs[block_num]['am_graph'] = G_am
				connected_components = len(list(nx.connected_components(G_am)))
				#visualize(am_gdf, title=str(block_num) + " cc = " + str(connected_components))

		osm_to_other_df = pd.DataFrame(osm_to_other)
		print(osm_to_other_df.head())
		sdot_no_matches = gpd.GeoDataFrame(geometry=sdot_no_matches)
		am_no_matches 	= gpd.GeoDataFrame(geometry=am_no_matches)
		print(str(len(sdot_no_matches)) + " " + str(len(am_no_matches)))
		visualize(osm_gdf_copy, buff=udistrict, title="No Matches for SDOT", extras=[sdot_no_matches])
		visualize(osm_gdf_copy, buff=udistrict, title="No Matches for AM", extras=[am_no_matches])
		sdot_no_matches.to_file('./data/generated/sdot_no_matches.shp')
		am_no_matches.to_file('./data/generated/am_no_matches.shp')
	else:
		sdot_no_matches = gpd.read_file('./data/generated/sdot_no_matches.shp')
		am_no_matches =  gpd.read_file('./data/generated/am_no_matches.shp')
		visualize(osm_gdf_copy, buff=udistrict, title="No Matches for SDOT", extras=[sdot_no_matches])
		visualize(osm_gdf_copy, buff=udistrict, title="No Matches for AM", extras=[am_no_matches])


	# TODO: calculate one to one comparison with slope and start and end point
	# TODO: calculate overalp with roads and buildings

def is_match(row1, row2):
	close_end = compare_ends(row1, row2)
	close_slope = compare_slopes(row1, row2)
	close_length = compare_lengths(row1, row2)
	return close_end and close_slope and close_length

def compare_lengths(row1, row2):
	line1 = row1[1]['geometry']
	line2 = row2[1]['geometry']
	if line1.length < line2.length:
		return line1.length / line2.length > LEN_COMP_THRESH
	else:
		return line2.length / line1.length > LEN_COMP_THRESH


# returns 0, 1, or 2 depending on number of matches the end points have.
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

	# extra code from attempting to parse osm.xml

	# city = ox.gdf_from_place('Seattle, WA')
	# ud = ox.gdf_from_place('University District, Seattle, WA')
	# data = ud.loc[0]
	# print(data)
	# G = ox.graph_from_bbox(data["bbox_north"], data["bbox_south"], data["bbox_east"], data["bbox_west"], network_type='walk')

	# print(type(G))
	# #G_projected = ox.project_graph(G)
	# #print(type(G_projected))

	# for n,nbrsdict in G.adjacency_iter():
	# 	for nbr,keydict in nbrsdict.items():
	# 		for key,eattr in keydict.items():
	# 			print(eattr)
	# #ox.plot_graph(G_projected)

	# click.echo("OSM Load Complete")
	#G = ox.graph_from_polygon(mission_shape, network_type='drive')
	#ox.plot_graph(G)

	# load access map
	# load SDOT data
	# Get the streets shapefile



	# oms_xml = xml.etree.ElementTree.parse('./data/osm/seattle.osm').getroot()

	# print("we got the xml")


	# for way in oms_xml.findall('way'):
	# 	print(type(way))
	# 	for tag in way.iter('tag'):
	# 		key = tag.get('k')
	# 		value = tag.get('v')
	# 		if value == "sidewalk":
	# 			wayjson=way.ExportToJson(as_object=True)
	# 			print(wayjson)
	# 			print("WE FOUND ONE")

	# features = [x for x in layer]
	# print(len(features))

	# data_list = []
	# for feature in features:
	#     data = feature.ExportToJson(as_object=True)
	#     coords = data['geometry']['coordinates']
	#     shapely_geo = Point(coords[0],coords[1])
	#     name = data['properties']['name']
	#     highway = data['properties']['highway']
	#     other_tags = data['properties']['other_tags']
	#     if other_tags and 'amenity' in other_tags:
	#         feat = [x for x in other_tags.split(',') if 'amenity' in x][0]
	#         amenity = feat[feat.rfind('>')+2:feat.rfind('"')]
	#     else:
	#         amenity = None
	#     data_list.append([name,highway,amenity,shapely_geo])
	# gdf = gpd.GeoDataFrame(data_list,columns=['Name','Highway','Amenity','geometry'],crs={'init': 'epsg:4326'}).to_crs(epsg=3310)

	#print(intersection)


if __name__ == "__main__":
	main()