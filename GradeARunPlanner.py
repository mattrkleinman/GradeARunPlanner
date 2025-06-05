import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium, folium_static
import csv
import os
import inspect

def get_address_loc(loc):
    # Get latitude and longitude for user-specified address
    geolocator = Nominatim(user_agent="grade_a_run_planner")
    geo = RateLimiter(geolocator.geocode, min_delay_seconds=2)  
    location = geo(loc)
    if location is None:
        point = []
        success = 0
        return point, success
    else:
        point = (location.longitude, location.latitude) 
        success = 1
        return point, success

def make_graph(loc,graphpath,rundist):
    # Generate graph, with an extra 500m padding
    G = ox.graph_from_address(address=loc, dist=(rundist+500)/2, dist_type="network", network_type="walk", simplify=True)
        
    # Add elevations to the graph
    original_elevation_url = ox.settings.elevation_url_template
    ox.settings.elevation_url_template = (
        "https://api.opentopodata.org/v1/aster30m?locations={locations}"
    )
    G = ox.elevation.add_node_elevations_google(G, batch_size=100, pause=1)
    G = ox.elevation.add_edge_grades(G)
    ox.settings.elevation_url_template = original_elevation_url

    # Project graph and consolidate nodes
    G_proj = ox.project_graph(G)
    G_c = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)

    # By default, just save in current directory; can change later
    ox.io.save_graphml(G_c, graphpath)
    return G_c

def read_graph(graph_file):
    # Read in generated graph file
    G_c = ox.io.load_graphml(graph_file)
    return G_c

def make_routes(G_c, point, filepref):
    # Find all the potential routes from starter node
    point_geom_proj, crs = ox.projection.project_geometry(Point(point), to_crs=G_c.graph['crs'])
    start_node = ox.distance.nearest_nodes(G_c, point_geom_proj.x, point_geom_proj.y)

    # Now calculate the distances between starting nodes and others
    start_nodes = np.linspace(start_node, start_node, len(G_c))
    dest_nodes = G_c.nodes
    if __name__ == "__main__":
        routes = ox.routing.shortest_path(G_c, start_nodes, dest_nodes, weight="length", cpus=None)

    # Save out the routes
    with open(filepref+'.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for n in np.arange(0,len(routes)):
            csvwriter.writerow(routes[n])

    return routes

def read_routes(routes_file):
    # Read csv file that contains all routes from starter node
    file = open(routes_file, "r")
    read_routes = list(csv.reader(file, delimiter=","))
    file.close()
    
    read_routes_float = []
    for row in read_routes:
        currline = []
        for item in row:
            currline.append(float(item))
        read_routes_float.append(currline)

    return read_routes_float 

if __name__ == "__main__":
    # Query user for relevant input
    # Other potential titles: Grade A Run Planner
    st.title('Grade A Run Planner')
    
    st.write("Provide a starting address, distance, and outbound grade, and Grade A Run Planner will find a matching route. Routes with a large degree of overlap will be removed, ensuring meaningfully distinct potential routes will be found.")

    st.write("For the first run, use the buttons to:")
    st.write("(1) Find coordinates of the starting address")
    st.write("(2) Generate a graph file") 
    st.write("(3) Upload it in the graph upload box")
    st.write("(4) Generate a routes file")
    st.write("(5) Upload it in the routes upload box")
    st.write("(6) Click the evaluate routes button")
    st.write("After this, the desired distance and grade parameters can be changed and the new best matching routes will be plotted after clicking the 'Evaluate routes' button. Initial map and routes generation can take awhile, especially for distances >10 miles.")

    st.write("For the smoothest operation, generate a unique map and routes file for a given starting location and distance. Unexpected errors can occur when trying to use map and routes files generated with different distance radii from the starting address, due to non-overlapping nodes and routes. Once a map is generated, the desired distance can be adjusted to suit shorter length runs, and matching routes re-evaluated.")

    # Initialize session state variables
    if "good_routes" not in st.session_state:
        st.session_state.good_routes = []
    if "routes" not in st.session_state:
        st.session_state.routes = []
    if "G_c" not in st.session_state:
        st.session_state.G_c = []
    if "success" not in st.session_state:
        st.session_state.success = 0
    if "location" not in st.session_state:
        st.session_state.location = []
    if "graph_file" not in st.session_state:
        st.session_state.graph_file = []
    if "routes_file" not in st.session_state:
        st.session_state.routes_file = []   
    if "reevaluate_routes" not in st.session_state:
        st.session_state.reevaluate_routes = 0
    if "current_dir" not in st.session_state:
        st.session_state.current_dur = []

    if len(st.session_state.current_dur)==0:
        st.session_state.current_dur = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    loc = st.text_input('Full starting street address: ')
    st.session_state.location = loc
    if st.button('Find coordinates of starting location'):
        st.session_state.point, st.session_state.success = get_address_loc(loc)
        if st.session_state.success == 0:
            st.write("Address geolocation failed, try a different starting point.")
            
    rundist_miles = st.number_input('Desired distance, in miles: ') 
    rundist = rundist_miles*1609.34 # convert to meters
    rungrade_percent = st.number_input('Desired grade of run, in percent: ')
    rungrade = rungrade_percent/100 # convert to decimal

    # For saving files, remove commas, replace spaces with underscore
    fname = loc.replace(', ', '_')
    fname = fname.replace(' ', '_') #catch any missed spaces
    filepref = st.session_state.current_dur+'\\data_files\\'+fname+'_'+str(rundist_miles)+'_miles'
    graphpath = filepref+'.graphml'

    # Users will start here, making and saving a graph file
    if st.button('Make new graph map from starting address and route distance. This may take awhile...'):
        # Check that the address gets geolocated accurately
        if st.session_state.success:
            st.session_state.G_c = make_graph(loc, graphpath, rundist)   
        else:
            st.write("Address geolocation failed, try a different starting point.")

    # Make graph upload box to upload saved graph
    graph_file = st.file_uploader("Choose a graphml graph file to upload.", type="graphml")
    if (graph_file is not None) and (len(st.session_state.G_c)==0):
        st.session_state.G_c = read_graph(st.session_state.current_dur+'\\data_files\\'+graph_file.name)
        st.session_state.graph_file = graph_file
    # User changed the map they uploaded
    if (graph_file is not None) and (graph_file != st.session_state.graph_file): 
        st.session_state.G_c = read_graph(st.session_state.current_dur+'\\data_files\\'+graph_file.name)
        st.session_state.graph_file = graph_file

    # Users will continue here, generating and saving start node to destination node routes
    if st.button('Calculate new routes from uploaded graphml. This may take awhile...'):
        if graph_file is not None:
            st.session_state.routes = make_routes(st.session_state.G_c, st.session_state.point, filepref)
            
    # Make routes upload box
    routes_file = st.file_uploader("Choose a csv routes file to upload.", type="csv")
    if (routes_file is not None) and (len(st.session_state.routes)==0):
        st.session_state.routes = read_routes(st.session_state.current_dur+'\\data_files\\'+routes_file.name)   
        st.session_state.routes_file = routes_file
    # User changed the routes they uploaded
    if (routes_file is not None) and (routes_file != st.session_state.routes_file): 
        st.session_state.routes = read_routes(st.session_state.current_dur+'\\data_files\\'+routes_file.name) 
        st.session_state.routes_file = routes_file
        st.session_state.reevaluate_routes = 1

    # Calculate the route metrics for the first time
    if st.button('Evaluate routes with current distance and grade inputs.'):
        if (routes_file is not None) and (len(st.session_state.routes)>0) and (len(st.session_state.G_c)>0):
            routes = st.session_state.routes
            G_c = st.session_state.G_c
            st.session_state.reevaluate_routes = 0
    
            # Evaluate all of the potential routes
            route_metrics = []
            for r in np.arange(0,len(routes)):
                
                # calculate distance and elevation change of the route
                dist = 0
                elev = 0
                for n in np.arange(1,len(routes[r])):
                    dist = dist + G_c[routes[r][n-1]][routes[r][n]][0]['length']
                    elev = elev + (G_c[routes[r][n-1]][routes[r][n]][0]['length']*G_c[routes[r][n-1]][routes[r][n]][0]['grade']) 
                    # If we only want to count uphill portions towards elevation change, comment line above and uncomment if statement below
                    # if G_c[routes[r][n-1]][routes[r][n]][0]['grade'] > 0:
                    #     elev = elev + (G_c[routes[r][n-1]][routes[r][n]][0]['length']*G_c[routes[r][n-1]][routes[r][n]][0]['grade'])  
            
                if dist > 0:
                    # Need to weigh grade difference more heavily; alternatively, could normalize these, but working
                    # in raw distance and grade is more intuitive
                    # Currently, 1km distance error and 1% grade error are equivalent
                    score = ((dist-rundist/2)/1000)**2 + (100*(rungrade-elev/dist))**2
                    route_metrics.append([r, dist, elev/dist, score])
            
            rm = np.array(route_metrics)
            rm = rm[rm[:, 3].argsort()]
    
            # Now go through routes that most match criteria, and exclude those too similar to better-matching routes
            previous_routes = list()
            good_routes = np.array([-1,-1,-1,-1])
            bestroute = list(map(int,routes[int(rm[0,0])]))
            previous_routes.append(bestroute.copy())
            for n in np.arange(1,len(routes)-1):
                too_similar = 1
                thisroute = list(map(int,routes[int(rm[n,0])]))
                any_too_similar = 0
                for j in np.arange(0,len(previous_routes)):
                    compare = list(set(thisroute).difference(previous_routes[j]))         
                    if len(compare)<0.5*len(thisroute):
                        any_too_similar = 1
            
                too_similar = any_too_similar 
                if too_similar == 0:
                    previous_routes.append(thisroute.copy())
                    good_routes = np.vstack([good_routes, rm[n,:]])
        
            good_routes = np.delete(good_routes, 0, 0)
            st.session_state.good_routes = good_routes
   
    # Now that routes are calculated, we can display whichever ones we want
    st.write('After routes are calculated, they will be plotted in batches of five, starting with the Nth best route specified below.')
    startroute = st.number_input('Which route to begin plot (1 matches criteria best)?', value=1)
    st.write(f'{len(st.session_state.good_routes)} total distinct routes')
    
    if (len(st.session_state.good_routes)>0) and (st.session_state.reevaluate_routes==0):
        routes = st.session_state.routes
        st.session_state.disp_routes = []
        curr_routes = st.session_state.good_routes
        
        if startroute+4<=len(curr_routes):
            curr_routes = curr_routes[startroute-1:startroute+4,:]
        else:
            startroute = len(curr_routes)-4
            curr_routes = curr_routes[-5:,:]

        ordered = np.arange(startroute,startroute+5,dtype=int).reshape(-1, 1)
        ordered = np.hstack([ordered, curr_routes])
        disp_routes = pd.DataFrame(ordered[:,[0,2,3,0]], columns=['Route Rank','Distance (miles)','Grade (%)','Color'])
        disp_routes.loc[:,'Distance (miles)'] = 2*disp_routes.loc[:,'Distance (miles)']/1609.34
        disp_routes.loc[:,'Grade (%)'] = 100*disp_routes.loc[:,'Grade (%)']
        
        plot_routes = []
        for n in np.arange(0,len(curr_routes)):
            plot_routes.append(list(routes[int(curr_routes[n][0])]))
    
        colors = ['yellow','orange','red','purple','blue']
        def route_cols(val): 
          color_hex = colors[int(val-startroute)] 
          return f'background-color: {color_hex}; color: {color_hex}' 
        
        st_df = st.dataframe(disp_routes.style.applymap(route_cols,subset='Color'), hide_index=True)
        
        # Initialize a minimal graph to center the folium map
        G_min = ox.graph_from_address(address=loc, dist=100, dist_type="network", network_type="walk", simplify=True)
        G_min = ox.project_graph(G_min)
        rem = list(G_min.nodes)[2:]
        G_min.remove_nodes_from(rem)
        G_min.remove_edges_from(list(G_min.edges))
        G_min.add_edge(list(G_min.nodes)[0],list(G_min.nodes)[1])
        
        m = ox.convert.graph_to_gdfs(G_min,edges=False).explore(zoom_start=13, tiles="cartodbdarkmatter")
        
        route_gdfs = (ox.routing.route_to_gdf(st.session_state.G_c, route) for route in plot_routes)
        rcount = 0
        for route_edges in route_gdfs:
            m = route_edges.explore(m=m, color=colors[0], style_kwds={"weight": 8-rcount/2, "opacity": 0.3-rcount*0.03}, legend=True)    
            rcount += 1
            colors.pop(0)

        # st_folium(m, returned_objects=[]) # may be necessary in the future
        folium_static(m, width=1600, height=900)

        


    














    
    
