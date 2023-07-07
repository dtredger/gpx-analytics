import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from ipywidgets import Button, HTML, HBox, VBox, Checkbox, FileUpload, Label, Output, IntSlider, Layout, Image, link
from ipyleaflet import basemaps, FullScreenControl, LayerGroup, Map, MeasureControl, Polyline, Marker, CircleMarker, WidgetControl

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


def ipyleaflet_chart(session):
    """
    Plot the GPS trace on an ipyleaflet map.
    Include waypoints if provided
    """
    lat_lng_pts = session.filtered_df[['latitude', 'longitude']].values
    mean_lat = session.filtered_df.median().latitude
    mean_lng = session.filtered_df.median().longitude

    # create the map
    m = Map(center=(mean_lat, mean_lng), 
            zoom=15, 
            basemap=basemaps.CartoDB.Positron) #Stamen.Terrain)

    # show trace
    # Attribute Default Value Doc
    # locations [[]] List of list of points of the polygon
    # stroke True Set it to False to disable borders
    # color “#0033FF” Stroke color
    # opacity 1.0 Stroke opacity
    # weight 5 Stroke width in pixels
    # fill True Whether to fill the polyline or not
    # fill_color “#0033FF”
    # fill_opacity 0.2
    # dash_array
    # line_cap “round”
    # line_join “round”
    line = Polyline(locations=[[[p[0], p[1]] for p in lat_lng_pts],],
                    color = "red", fill=False, weight=1)
    m.add_layer(line)

    # add markers
    waypoints = [
        Marker(location=(lat_lng_pts[0], lat_lng_pts[1]), title='TEST TITLE',
               popup=HTML(value='TEST'), draggable=False)
        for point in [] # TODO
    ]
    waypoints_layer = LayerGroup(layers=waypoints)
    m.add_layer(waypoints_layer)
    
    # add a checkbox to show / hide waypoints
    waypoints_checkbox = Checkbox(value=True, description='Show Waypoints')
    
    def update_visible(change):
        for p in waypoints:
            p.visible = change['new']
    
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
                                      mode = 'lines',
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
    y_ax = Axis(label='SOG (kts)', scale=y_scale, orientation='vertical')
    
    lines = Lines(x=px, y=py, scales={'x': x_scale, 'y': y_scale}, stroke_width=1,
                  colors=['blue'], fill='bottom', fill_opacities=[.7])

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


def seaborn_violin(session, metric='cog'):
    """
    display violin diagram of metrics split by segment.
    Use slider to set filtered_df because otherwise the graph is overstuffed
    matplotlib.show() call is required to display 
    """
    violin = sns.catplot(data=session.filtered_df, kind='violin', inner=None, 
                         palette="pastel", aspect=4, #hue='upwind',
                         x="segment", y=metric, ) 
    sns.stripplot(data=session.filtered_df, color="k", jitter=True, size=0.5,
                  x="segment", y=metric, ax=violin.ax)
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