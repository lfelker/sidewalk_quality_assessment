# This repo contains code to calculate metrics comparing sidewalk datasets.

# How To Run
- you need python3 and all of the packages declared at the top of main.py installed

- you need the data to compare
1) SDOT swk json dataset
2) Access Map swidewalks.geojson
3) OSM geojson of sidewalks in the area you want to compare to

- data used for comparisons
1) seattle streets linestring shapefile
2) seattle street polygon shapefile
3) seattle neighborhood shapefile

you can modify paths in main.py to these data sources, or replicate the paths and file names in main.py

# Output
## file: ./data/output.csv
## data columns:
- 'block_num': corresponding block number

- 'osm_cc': osm connected components per block
- 'osm_swk_len': length of sidewalks per block with links removed
- 'osm_swk_seg_num': number of segments per block with links removed
- 'osm_overlap': area of osm sidewalk with buffer of 3 ft width overlaping with the street network polygons per block
- 'osm_swk_area': area of osm sidewalk with buffer of 3 ft width per block
- 'osm_ground_overlap': area of osm sidewalk with buffer of 3 ft width per block overlaping with "ground truth" osm dataset with 3 ft width.

- 'sdot_cc': sdot connected components per block
- 'sdot_swk_len': length of sidewalks per block
- 'sdot_swk_seg_num': number of segments per block
- 'sdot_overlap': area of sdot sidewalk with buffer of 3 ft width overlaping with the street network polygons per block
- 'sdot_swk_area': area of sdot sidewalk with buffer of 3 ft width per block
- 'sdot_ground_overlap': area of sdot sidewalk with buffer of 3 ft width per block overlaping with "ground truth" osm dataset with 3 ft width.

- 'am_cc': am connected components per block
- 'am_swk_len': length of sidewalks per block
- 'am_swk_seg_num': number of segments per block
- 'am_overlap': area of am sidewalk with buffer of 3 ft width overlaping with the street network polygons per block
- 'am_swk_area': area of am sidewalk with buffer of 3 ft width per block
- 'am_ground_overlap': area of am sidewalk with buffer of 3 ft width per block overlaping with "ground truth" osm dataset with 3 ft width.

