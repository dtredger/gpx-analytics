import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from ipywidgets import Button, HTML, HBox, VBox, Checkbox, FileUpload, Label, Output, IntSlider, Layout, Image, link
from ipyleaflet import basemaps, FullScreenControl, LayerGroup, Map, MeasureControl, Polyline, Marker, CircleMarker, WidgetControl, AntPath, Popup

from bqplot import Axis, Figure, Lines, LinearScale, DateScale
from bqplot.interacts import IndexSelector

import seaborn as sns

"""
Methods available for charting are:
    - render_polar
    - ipyleaflet_chart
    - bqplot_graph
    - link_chart_graph
    - seaborn_violin
    - [deprecated] render_plotly_speed_graph
    - [deprecated] render_mapbox_chart
"""


def segment_is_upwind(rows):
    return True if np.mean(rows.upwind) >= 0 else False


def speed_fraction(df, rows):
    mean = np.mean(df['sog_kts'])
    if np.mean(rows.sog_kts) > mean:
        return 4
    else:
        return 2

    
# Match coordinates pair to a point in the dataframe
# Latitudes will be from the clicked location on the map, 
# and not match points specifically. Clicked map points return
# 14 decimals (eg: 43.634773071709375), vakaros provides 6: (43.640782)
# create a range (via `jiggle`) to match at least one point
def match_coords(df, coords, jiggle=0.00001):
    lat, lon = [round(pt, 6) for pt in coords]

    lat_floor, lat_ceil = [lat-jiggle, lat+jiggle]
    lon_floor, lon_ceil = [lon-jiggle, lon+jiggle]

    return df[ (df['latitude'].between(lat_floor, lat_ceil)) & 
               (df['longitude'].between(lon_floor, lon_ceil)) ]


def point_from_coords(df, coords):
    matches = match_coords(df, coords)
    if len(matches) == 0:
        matches = match_coords(df, coords, jiggle=0.00010)
        if len(matches) == 0:
            matches = match_coords(df, coords, jiggle=0.00100)
    return matches.iloc[0]


def html_table(point):
    return HTML(value=f"""<table>
                            <tr>
                                <td><b>Timestamp</b></td>
                                <td>{point.timestamp.strftime('%X')}</td>
                            </tr>
                            <tr>
                                <td><b>Reading No., Segment</b></td>
                                <td>{point.name} - {int(point.segment)}</td>
                            </tr>
                            <tr>
                                <td><b>Speed (kts)</b></td>
                                <td>{point.sog_kts}</td>
                            </tr>
                            <tr>
                                <td><b>Heading</b></td>
                                <td>{point.hdg_true}</td>
                            </tr>
                            <tr>
                                <td><b>Course over Ground</b></td>
                                <td>{point.cog}</td>
                            </tr>
                            <tr>
                                <td><b>Roll & Pitch</b></td>
                                <td>{point.roll}, {point.pitch}</td>
                            </tr>
                            <tr>
                                <td><b>VMG (kts)</b></td>
                                <td>{point.vmg_kts}</td>
                            </tr>
                            <tr>
                                <td><b>Cumulative Distance (m)</b></td>
                                <td>{round(point.distance_cumulative, 2)}</td>
                            </tr>
                            <tr>
                                <td><b>lat/lon pair</b></td>
                                <td>[{point.latitude}, {point.longitude}]</td>
                            </tr>
                          </table>""")




def ipyleaflet_chart(session, height_pix='500'):
    """
    Plot the GPS trace on an ipyleaflet map.
    Include waypoints if provided.

    Colours will be separated by tack. Dashed lines (AntPath) for downwind
    """
    mean_lat = session.filtered_df.median(numeric_only=True).latitude
    mean_lng = session.filtered_df.median(numeric_only=True).longitude

    # create the map
    m = Map(center=(mean_lat, mean_lng), 
            zoom=15,
            layout=Layout(width='100%', height=f"{height_pix}px"),
            basemap=basemaps.CartoDB.Positron) #Stamen.Terrain)

    markers = []
    def line_click_handler(**kwargs):
        point = point_from_coords(session.filtered_df, kwargs['coordinates'])

        popup = Popup(
            location=[point.latitude, point.longitude],
            child=html_table(point),
            close_button=False,
            auto_close=False,
            close_on_escape_key=False
        )
        m.add_layer(popup)

        # line_marker = Marker(location=[point.latitude, point.longitude],
        #                      draggable=False, popup=html_table(point))
        # m.add_layer(line_marker)

        
    segments = {}        
    for x in range(int(session.filtered_df.iloc[1].segment), int(session.filtered_df.iloc[-1].segment)+1):
        segments[x] = []

    for ix, row in session.filtered_df.iterrows():
        segments[row.segment] += [[row.latitude, row.longitude]]
        
    for seg_key in segments:
        rows = session.df[session.df['segment'] == seg_key]
        # tacks are indicated by 1 / -1 for port / stbd
        seg_colour = 'red' if rows.iloc[0].tack > 0 else 'green'
        
        # use antpath (striped line) for downwind
        if segment_is_upwind(rows):
            line = Polyline(locations=segments[seg_key], 
                            color = seg_colour, fill=False, weight=speed_fraction(session.filtered_df, rows))
        else:  
            line = AntPath(locations=segments[seg_key], color = seg_colour, 
                             fill=False, weight=speed_fraction(session.filtered_df, rows), paused=True)
    
        line.on_click(line_click_handler)
        m.add_layer(line)
    
    # session.tacks (and gybes) contain the index of the first point
    # on the new tack.
    # Use reddish for stbd-> port
    # greenish for port->stbd
    def transition_colour(tack, type='tack'):
        reddish = '#f56042'
        greenish = '#36e036'
        green_gybe = '#36e0b8'
        red_gybe = '#e08536'
        if type == 'gybe':
            return green_gybe if tack > 0 else red_gybe
        else:
            return greenish if tack > 0 else reddish
    
    # add markers
    waypoints = []

    # tacks
    for ix in session.tacks:
        if ix in session.filtered_df.index:
            row = session.filtered_df.loc[ix]
            marker = CircleMarker(location=(row.latitude, row.longitude),
                                  title='Tack', 
                                  radius=2,
                                  color=transition_colour(row.tack),
                                  popup=HTML(value=f"Tack to start segment {row.segment}"), 
                                  draggable=False)
            waypoints += [marker]
        
    # gybes
    for ix in session.gybes:
        if ix in session.filtered_df.index:
            row = session.filtered_df.loc[ix]
            marker = CircleMarker(location=(row.latitude, row.longitude), 
                                  title='Gybe', 
                                  radius=2,
                                  color=transition_colour(row.tack, 'gybe'),
                                  popup=HTML(value=f"Gybe to start segment {row.segment}"), 
                                  draggable=False)
            waypoints += [marker]
        
    waypoints_layer = LayerGroup(layers=waypoints)
    m.add_layer(waypoints_layer)
    
    
    # legend
    # legend = LegendControl({"low":"#FAA", "medium":"#A55", "High":"#500"}, 
    #                        name="Legend", position="bottomright")
    # m.add_control(legend)

    
    # add a checkbox to show / hide waypoints
    waypoints_checkbox = Checkbox(value=True, description='Show Maneuvers')
    
    def update_visible(change):
        for p in waypoints:
            p.stroke = change['new']
    
    waypoints_checkbox.observe(update_visible, 'value')
    waypoint_control = WidgetControl(widget=waypoints_checkbox, position='bottomright')
    m.add_control(waypoint_control)
    
    # enable full screen mode
    m.add_control(FullScreenControl())
        
    # add measure control
    measure = MeasureControl(
        position='bottomleft',
        active_color = 'orange',
        primary_length_unit = 'kilometers'
    )
    m.add_control(measure)
    
    # def handle_click(**kwargs):
    #     # print(kwargs)
    #     if kwargs.get('type') == 'click':
    #         m.add_layer(Marker(location=kwargs.get('coordinates')))
    # m.on_interaction(handle_click)
        
    return m


def render_polar(session, bin_deg_size=3):
    """ 
    Polar diagram showing speed at different angles. TWA is displayed with gray wedge
    """
    filtered_df = session.filtered_df
    wind_dir = session.wind_dir
    
    pf = pd.DataFrame(columns=['cog', 'top', 'mean', 'tack_raw'])
    bin_mod = round((bin_deg_size - 1) / 2)

    for x in range(0, 361):
        rows = filtered_df.loc[filtered_df["cog"].isin([x-bin_mod, x, x+bin_mod])]
        pf.loc[x] =  { 'cog': x, 
                       'top': rows.sog_kts.quantile(0.95), 
                       'mean': rows.sog_kts.quantile(0.5),
                       'tack_raw': rows.tack_raw.mean() }

    fig = go.Figure()
    fig.update_layout(autosize=False, 
                      width=700, 
                      height=700, 
                      polar_angularaxis_direction='clockwise')
    
    fig.add_trace(go.Barpolar(
        r=[10],
        theta=[wind_dir],
        width=[20],
        marker_color=["gray"],
        marker_line_color="black",
        marker_line_width=2,
        opacity=0.9,
        name="True Wind Direction"
    ))
    
    # stbd traces
    add_traces_to_fig(fig, pf.loc[pf["tack_raw"] > 0])
    # port traces
    add_traces_to_fig(fig, pf.loc[pf["tack_raw"] < 0])

    fig.update_layout(
        title = 'Speed Polar',
        showlegend = True,
        polar=dict(angularaxis=dict(rotation=90+wind_dir))
    ).show()

#  Add mean and 95% to polar chart for port and stbd
def add_traces_to_fig(fig, frame):
    if frame.tack_raw.mean() > 0:
        colour = { "mean": "#6AFF33", "top": "#46FF01" }
    else:
        colour = { "mean": "#C02002", "top": "#FF2800" }

    for quantile in ["top", "mean"]:
        fig.add_trace(go.Scatterpolar(r = frame[quantile],
                                      theta = frame['cog'],
                                      mode = 'lines+markers',
                                      name = quantile,
                                      line_color = colour[quantile],
                                      # connectgaps=True,
                                      # hoverinfo=['theta', 'r'],
                                      fill='tonext'))


def bqplot_graph(session, metrics=['sog_kts'], x_scale='time'):
    """
    Return a graph for the given metric(s) based on the session gps trace

    Currently only sog_kts vs time is supported
    TODO: legend does not display??
    """
    if x_scale == 'time':
        px = session.filtered_df['timestamp']
        x_scale = DateScale()
    else:
        px = session.filtered_df['distance_cumulative']
        x_scale = LinearScale()
        
    py = session.filtered_df[metrics]
    # py_2 = session.filtered_df['hdg_true']
    
    y_scale = LinearScale()
    x_scale.allow_padding = False
    
    x_ax = Axis(label='Time', scale=x_scale)
    y_ax = Axis(label=metrics[0], scale=y_scale, orientation='vertical')
    
    lines = Lines(x=px, y=py, scales={'x': x_scale, 'y': y_scale}, stroke_width=2,
                  colors=['blue'], fill='bottom', fill_opacities=[.8])

    # lines_2 = Lines(x=px, y=py_2, scales={'x': x_scale, 'y': y_scale}, colors=['green'])
    
    graph = Figure(title='Speed Chart', axes=[x_ax, y_ax], marks=[lines], #, lines_2],
                  legend_location="top-right", display_legend=True)
    graph.layout.width = 'auto'
    graph.layout.height = 'auto'
    graph.layout.min_height = '500px'
    graph.layout.fill='bottom'
    graph.interaction = IndexSelector(scale=x_scale)
    return graph


def link_chart_graph(chart, graph, session):
    """
    Link the chart (ipyleaflet map) and graph (bqplot).
    Changing the selection on the graph will update the
    marker on the map
    """
    points = session.filtered_df
    
    # TODO - use timestamp or time_elapsed_sec for times?
    # full_distance = session.filtered_df.iloc[-1].distance_cumulative
    
    def find_point(value, column='timestamp'):
        """
        Find the matching row matching a timestamp from the graph.
        The graph seems to invent its own stamps, so match to the closest second.
        This assumes update rate > 1hz

        Find a point given the value in given column. Return the
        first match (there should never be >1)
        """
        timestamp = pd.Timestamp(value)
        df_rows = session.filtered_df.loc[
                    (session.filtered_df['timestamp'] > timestamp.floor(freq='S')) & 
                    (session.filtered_df['timestamp'] < timestamp.ceil(freq='S')) ]
        if len(df_rows) == 0:
            df_rows = session.filtered_df

        return df_rows.iloc[0]
    
    # add a checkbox to auto center
    autocenter = Checkbox(value=False, description='Auto Center')
    autocenter_control = WidgetControl(widget=autocenter, position='bottomright')
    chart.add_control(autocenter_control)
    
    # mark the current position on the map
    start = find_point(pd.Timestamp(0))
    marker = CircleMarker(visible=False, location=(start.latitude, start.longitude),
                          radius=10, color="green", fill_color="green")
    chart.add_layer(marker)
    brushintsel = graph.interaction

    def update_range(change):
        """
        Update the position on the map when the graph selector changes

        On interaction, the graph will provide a `selected`, which looks like
        ['2023-06-30T18:39:58.366'] for timestamp ( np.datetime obj)
        """
        if brushintsel.selected.shape != (1,):
            return
        selected = brushintsel.selected[0] #  * 1000  # convert from km to m
         
        point = find_point(selected)

        marker.visible = True
        marker.location = (point.latitude, point.longitude)
        if autocenter.value:
            chart.center = marker.location
        
    brushintsel.observe(update_range, 'selected')


def seaborn_violin(session, metric='cog', dots_enabled=False, omit_stationary=True):
    """
    display violin diagram of metrics split by segment and colour coded 
    by upwind/downwind.
    Includes ability to render indivdual metric points with `dots_enabled`
    omit_stationary leaves out segments with speed < min_sailing_kts

    Use slider to set filtered_df because otherwise the graph is overstuffed
    matplotlib.show() call is required to display 
    """
    if omit_stationary:
        df = session.filtered_df[session.filtered_df['upwind'] != 0]
    else:
        df = session.filtered_df

    violin = sns.catplot(data=df, kind='violin', inner=None, 
                         palette="pastel", aspect=4, hue='upwind',
                         x="segment", y=metric, )
    if dots_enabled:
        sns.stripplot(data=df, color="k", jitter=True, size=0.5,
                      x="segment", y=metric, ax=violin.ax)
    plt.grid()
    plt.show()


# *** Deprecated Methods ***

def render_mapbox_chart(session, crash_enabled=False, boat_enabled=False):
    """
    * Deprecated: use the ipyleaflet map instead (it allows linking graph to
    scrub through session) *

    default trace color is stbd/port tack
    markers for tacks, gybes, and crashes
    uses session.filtered_df for data, so modify df with widgets to use subset
    of points

    possible chart styles are:
        'open-street-map', 
        'white-bg', 
        'carto-positron', 
        'carto-darkmatter', 
        'stamen- terrain', 
        'stamen-toner', 
        'stamen-watercolor'.
    """
    filtered_df = session.filtered_df

    fig = px.line_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="tack",
        line_group="segment",
        mapbox_style='carto-positron', #"open-street-map",
        hover_data=["timestamp", "sog_kts", "cog", "distance_cumulative", "upwind"],
        zoom=16,
        height=1000
    )
    # Filter out tacks not present in filtered dataFrame
    filtered_tacks = session.filtered_df.index.intersection(session.tacks)
    fig.add_trace(go.Scattermapbox(
        lat=filtered_df["latitude"][filtered_tacks],
        lon=filtered_df["longitude"][filtered_tacks],
        mode="markers",
        marker=go.scattermapbox.Marker(size=10),
        name="Tack"
    ))
        # Filter out tacks not present in filtered dataFrame        
    filtered_gybes = session.filtered_df.index.intersection(session.gybes)
    fig.add_trace(go.Scattermapbox(
        lat=filtered_df["latitude"][filtered_gybes],
        lon=filtered_df["longitude"][filtered_gybes],
        mode="markers",
        marker=go.scattermapbox.Marker(size=10),
        name="Gybe"
    ))
    if boat_enabled:
        fig.add_trace(go.Scattermapbox(
            lat=[filtered_df.iloc[0]["latitude"]],
            lon=[filtered_df.iloc[0]["longitude"]],
            mode="markers",
            marker=go.scattermapbox.Marker(size=15), 
                # symbol="bus"#,slaughterhouse"#,
                # anglesrc=point["cog"]
            name="Boat"
        ))
    if crash_enabled:
        # Filter out tacks not present in filtered dataFrame
        filtered_crashes = session.filtered_df.index.intersection(session.crashes)
        fig.add_trace(go.Scattermapbox(
            lat=filtered_df["latitude"][filtered_crashes],
            lon=filtered_df["longitude"][filtered_crashes],
            mode="markers",
            marker=go.scattermapbox.Marker(size=10),
            name="Crash"
        ))
    fig.show()
    return fig


def render_plotly_speed_graph(session):
    """
    * Deprecated. Use bqplot instead, since it allows linking to map *

    Basic graph of speed vs time
    """
    fig = px.line(session.df, x="timestamp", y="sog_kts", title="Speed (kts)")
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode=mode).show()