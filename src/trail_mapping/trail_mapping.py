import datetime
import json
import os
from io import StringIO
from statistics import mean

import gpxpy
import srtm

from bqplot import Axis, Figure, Lines, LinearScale
from bqplot.interacts import IndexSelector
from ipyleaflet import basemaps, FullScreenControl, LayerGroup, Map, MeasureControl, Polyline, Marker, CircleMarker, WidgetControl
from ipywidgets import Button, HTML, HBox, VBox, Checkbox, FileUpload, Label, Output, IntSlider, Layout, Image, link


def parse_data(file):
    """
    Parse a GPX file and add elevations
    """
    gpx = gpxpy.parse(file)
    elevation_data = srtm.get_data()
    elevation_data.add_elevations(gpx, smooth=True)
    return gpx


def plot_map(gpx):
    """
    Plot the GPS trace on a map
    """
    points = [p.point for p in gpx.get_points_data(distance_2d=True)]
    mean_lat = mean(p.latitude for p in points)
    mean_lng = mean(p.longitude for p in points)

    # create the map
    m = Map(center=(mean_lat, mean_lng), zoom=12, basemap=basemaps.Stamen.Terrain)

    # show trace
    line = Polyline(locations=[[[p.latitude, p.longitude] for p in points],],
                    color = "red", fill=False)
    m.add_layer(line)

    # add markers
    waypoints = [
        Marker(location=(point.latitude, point.longitude), title=point.name,
               popup=HTML(value=point.name), draggable=False)
        for point in gpx.waypoints
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


def plot_stats(gpx):
    """
    Compute statistics for a given trace
    """
    lowest, highest = gpx.get_elevation_extremes()
    uphill, downhill = gpx.get_uphill_downhill()
    points = gpx.get_points_data(distance_2d=True)

    _, distance_from_start, *rest = points[-1]

    stat_layout = Layout(margin="10px", padding="10px", border="1px solid black",
                         flex_flow='column', align_items='center')

    start_time = gpx.get_time_bounds().start_time
    duration = gpx.get_duration()
    stats = [
        ('Date', start_time.strftime("%Y-%m-%d") if start_time else '-'),
        ('Distance', f"{round(distance_from_start / 1000, 2)} km"),
        ('Duration', str(datetime.timedelta(seconds=duration)) if duration else '-'),
        ('Lowest', f"{int(lowest)} m" if lowest else '-'),
        ('Highest', f"{int(highest)} m" if highest else '-'),
        ('Uphill', f"{int(uphill)} m" if uphill else '-'),
        ('Downhill', f"{int(downhill)} m" if downhill else '-'),
    ]

    stats_formatted = [
        VBox([
            HTML(value=f"<strong>{title}</strong>"),
            Label(value=value)
        ], layout=stat_layout)
        for title, value in stats
    ]
    return HBox(stats_formatted, layout=Layout(flex_flow='row', align_items='center'))

def plot_elevation(gpx):
    """
    Return an elevation graph for the given gpx trace
    """
    points = gpx.get_points_data(distance_2d=True)
    px = [p.distance_from_start / 1000 for p in points]
    py = [p.point.elevation for p in points]

    # if there is no elevation data, create zeros
    for i, v in enumerate(py):
        py[i] = 0 if v == None else v
    
    x_scale, y_scale = LinearScale(), LinearScale()
    x_scale.allow_padding = False
    x_ax = Axis(label='Distance (km)', scale=x_scale)
    y_ax = Axis(label='Elevation (m)', scale=y_scale, orientation='vertical')

    lines = Lines(x=px, y=py, scales={'x': x_scale, 'y': y_scale})

    elevation = Figure(title='Elevation Chart', axes=[x_ax, y_ax], marks=[lines])
    elevation.layout.width = 'auto'
    elevation.layout.height = 'auto'
    elevation.layout.min_height = '500px'

    elevation.interaction = IndexSelector(scale=x_scale)
    return elevation

def link_trace_elevation(trace, elevation, gpx, debug):
    """
    Link the trace the elevation graph.
    Changing the selection on the elevation will update the
    marker on the map
    """
    points = gpx.get_points_data(distance_2d=True)
    _, distance_from_start, *rest = points[-1]
    n_points = len(points)

    def find_point(distance):
        """
        Find a point given the distance
        """
        progress = min(1, max(0, distance / distance_from_start))
        position = int(progress * (n_points - 1))
        return points[position].point

    # add a checkbox to auto center
    autocenter = Checkbox(value=False, description='Auto Center')
    autocenter_control = WidgetControl(widget=autocenter, position='bottomright')
    trace.add_control(autocenter_control)

    # mark the current position on the map
    start = find_point(0)
    marker = CircleMarker(visible=False, location=(start.latitude, start.longitude),
                          radius=10, color="green", fill_color="green")
    trace.add_layer(marker)

    brushintsel = elevation.interaction

    def update_range(change):
        """
        Update the position on the map when the elevation
        graph selector changes
        """
        if brushintsel.selected.shape != (1,):
            return
        marker.visible = True
        selected = brushintsel.selected * 1000  # convert from km to m
        point = find_point(selected)
        marker.location = (point.latitude, point.longitude)

        if autocenter.value:
            trace.center = marker.location

        position = max(0, int((selected / distance_from_start) * len(points)))

    brushintsel.observe(update_range, 'selected')


def plot_gpx(gpx_filename):
    with open(gpx_filename) as gpx_file:
        gpx = parse_data(gpx_file)

    stats = plot_stats(gpx)
    trace = plot_map(gpx)
    elevation = plot_elevation(gpx)
    debug = Label(value='')

    display(stats)
    display(trace)
    display(elevation)
    display(debug)

    link_trace_elevation(trace, elevation, gpx, debug)
