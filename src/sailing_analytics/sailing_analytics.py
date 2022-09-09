from collections import deque
from geopy import distance
import gpxpy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import windrose
import time


# Code from https://gist.github.com/jeromer/2005586
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


# is the craft deemed to be moving? Determined from avg speed within a given
# window of time vs the avg of stopped_threshold_ms and moving_threshold_ms
# @param speed_window [Array]
# @param params [Dict]
def is_craft_moving(speed_window, params):
    return (np.mean(speed_window) >= (params["stopped_threshold_ms"] + params["moving_threshold_ms"]) / 2)


# create count of maneuvers including:
# "is_moving": is_moving,
# "tack": tack,
# "crashes": crashes,
# "crash_data": crash_data,
# "port_crashes": port_crashes,
# "starboard_crashes": starboard_crashes,
# "stopped_segments": stopped_segments,
# "moving_segments": moving_segments}
def maneuvers(speed_ms, tack_raw, upwind, params):
    cnt = 0
    stopped_segments = []
    # how many moving points in a row has there been
    moving_segments = []
    # list of gpx points where crashes occur
    crashes = []
    starboard_crashes = []
    port_crashes = []
    crash_data = []

    # the window is a range of points (equal to window_size) to smooth out speeds
    # and other averages
    window = range(params["window_size"])
    # Set all is_moving up until the index of window_size to 1 if the craft is
    # moving, 0 otherwise
    is_moving = np.zeros(len(speed_ms))
    is_moving[:params["window_size"]] = 1 if is_craft_moving(speed_ms[window], params) else 0
    # Set all points in the window to the median tack in the current window
    tack = np.zeros(len(speed_ms))
    tack[:params["window_size"]] = np.median(tack_raw[window])

    # Iterate over every speed point
    for i in range(1, len(speed_ms)):
        start = min(i, len(speed_ms) - params["window_size"])
        end = min(i + params["window_size"], len(speed_ms))
        # Adjust the window to the current point and the `window_size` points ahead
        window = range(start, end)
        # Average speed within the current window
        window_speed_ms = np.mean(speed_ms[window])
        # was the craft moving at the last recorded point?
        if is_moving[i - 1] == 1:
            is_moving[i] = 1
            if (tack[i - 1] != tack_raw[i]) and (len(set(tack_raw[window])) == 1):
                # only update tack if consistent across the window
                # if we're updating tack, scroll back that update earlier in time
                tack[max(i - params["window_size"] + 1, 0):i] = np.median(tack_raw[window])
            tack[i] = tack[i - 1]
            # craft is not below `stopped` threshold
            if window_speed_ms > params["stopped_threshold_ms"]:
                cnt += 1
            # craft is `stopped`
            else:
                is_moving[i] = 0
                moving_segments.append(cnt)
                crashes.append(i)
                is_moving[
                max(i - 2 * params["window_size"], 0):i] = 0  # Reset time before crash to not moving to clean up data

                tack_before_crash = tack[max(i - 2 * params["window_size"], 0)]
                upwind_before_crash = upwind[max(i - 2 * params["window_size"], 0)]
                if tack_before_crash == 0:
                    tack_before_crash = np.median([x for x in tack[max(i - 2 * params["window_size"], 0):i] if x != 0])
                if tack_before_crash == 1:  # port tack
                    port_crashes.append(i)
                else:
                    starboard_crashes.append(i)

                crash_data.append([i,
                                   "Port" if tack_before_crash == 1 else "Starboard",
                                   "Upwind" if upwind_before_crash == 1 else "Downwind"])

                tack[max(i - 2 * params["window_size"],
                         0):i] = 0  # Reset time before crash to not on a tack to clean up data
                cnt = 1
        # the craft was not moving at the previously recorded point
        else:
            if window_speed_ms < params["moving_threshold_ms"]:
                cnt += 1
            else:
                is_moving[i] = 1
                stopped_segments.append(cnt)
                cnt = 1

    return {
        "is_moving": is_moving,
        "tack": tack,
        "crashes": crashes,
        "crash_data": crash_data,
        "port_crashes": port_crashes,
        "starboard_crashes": starboard_crashes,
        "stopped_segments": stopped_segments,
        "moving_segments": moving_segments}


# This chooses a wind direction that is in the middle of the range of directions that are almost-never-travelled
def wind_direction(bearing, speed):
    (bearing_density, bins) = np.histogram(bearing, bins=range(360))
    # Normalize so that if I spent an equal time going in each direction, each bin would have a density of 1.0
    bearing_density = list(bearing_density / np.sum(bearing_density) * 360)

    # Pull out maximum and start it at 0 degrees for simplicity - this way I don't have to worry about the wind direction crossing 0
    offset = np.where(bearing_density == np.max(bearing_density))[0][0]
    bearing_density_offset = bearing_density[offset:] + bearing_density[:offset]

    # state 0 - a direction regularly travelled
    # state 1 - a direction not regularly travelled (and thus a candidate for the wind direction)
    state = 0
    cnt = 0
    start_loc = 0

    segments = []

    for i in range(len(bearing_density_offset)):
        if state == 0:
            # 0.5 - arbitrary parameter between "direction regularly traveled" and "direction irregularly travelled"
            if bearing_density_offset[i] < 0.5:
                state = 1
                cnt = 1
                start_loc = i
        else:
            if bearing_density_offset[i] < 0.5:
                cnt += 1
            else:
                state = 0
                segments.append((start_loc, cnt))

    segments = sorted(segments, key=lambda x: x[1], reverse=True)
    candidate = (segments[0][0] + offset + int(segments[0][1] / 2)) % 360

    if 0.5 * segments[0][1] < segments[1][1]:
        candidate_2 = (segments[1][0] + offset + int(segments[1][1] / 2) + 180) % 360
        candidate = int((candidate + candidate_2) / 2)

    # top N% downwind vmg should be greater than top N% upwind vmg
    # if that's not true, then I probably have my wind direction reversed
    vmg = np.cos((bearing - candidate) * np.pi / 180) * speed
    top2percent_loc = int(len(bearing) / 100)
    partition = np.partition(vmg, [top2percent_loc, -top2percent_loc])
    top_vmg_upwind = partition[-top2percent_loc]
    top_vmg_downwind = partition[top2percent_loc]
    if np.abs(top_vmg_downwind) < np.abs(top_vmg_upwind):
        candidate = (candidate + 180) % 360

    return candidate


# Create a new session from a GPX file
class SailingSession:
    default_params = {
        # "wind_dir":              ,
        "stopped_threshold_ms": 0.1,  # 3.5,
        "moving_threshold_ms": 0.2,  # 4.5,
        "window_size": 5,
        "kts_speed_cap": 15,  # 40,
        "timezone": "US/Eastern"
    }

    def __init__(self, gpx_file, params=None):
        self.params = self.default_params

        with open(gpx_file, "r") as f:
            gpx = gpxpy.parse(f)

        self.gpx = gpx
        self.process_gpx_file()

    def get_gpx_points_list(self, gpx):
        if gpx.get_track_points_no() == 0:
            raise IOError("GPX File has no tracks")

        gpx_points = gpx.tracks[0].segments[0].points
        point_list = [[point.longitude,
                       point.latitude,
                       point.elevation,
                       point.time.astimezone(tz=pytz.timezone(self.params["timezone"]))] for point in gpx_points]
        return gpx_points, point_list

    def create_dataframe(self, data, point_list, columns=['lon', 'lat', 'alt', 'time']):
        params = self.params
        df = pd.DataFrame(data=point_list, columns=columns)

        # distance from previous coordinate, in meters
        df["distance_step"] = [0] + [distance.distance((data[i - 1].latitude, data[i - 1].longitude),
                                                       (data[i].latitude, data[i].longitude)).m
                                     for i in range(1, len(data))]
        # add total distance from all previous steps
        df["distance_cumulative"] = np.cumsum(df["distance_step"])
        # gap between this point and previous point, in seconds
        df["time_diff"] = [1] + [(data[i].time - data[i - 1].time).total_seconds() for i in range(1, len(data))]
        # cumulative time from first point
        df["time_elapsed_sec"] = np.cumsum(df["time_diff"]) - 1
        # compass bearing between this point and previous point
        # TODO: should there be a check to verify impossibly abrupt changes of bearing?
        df["bearing"] = [0] + [calculate_initial_compass_bearing((data[i - 1].latitude, data[i - 1].longitude),
                                                                 (data[i].latitude, data[i].longitude))
                               for i in range(1, len(data))]
        # speed in meters per second. 1 m/s = 1.94384 kts, or 3.6 km/h
        df["speed_ms"] = df["distance_step"] / df["time_diff"]
        df["speed_kts"] = df["speed_ms"] * 1.94384
        df["speed_kts_capped"] = df["speed_kts"]
        # replace any row faster than kts_speed_cap with that value
        # TODO (should there be a max acceleration, to determine when the speed has
        # changed too dramatically between consecutive points?
        df.loc[df["speed_kts"] > params["kts_speed_cap"], "speed_kts_capped"] = params["kts_speed_cap"]
        return df

    # set specified and computed wind directions
    def set_wind_directions(self, df):
        params = self.params
        if "wind_dir" in params:
            self.wind_dir = params["wind_dir"]
        # Only points with speeds faster than moving_threshold_ms will be used
        # to calculate wind_direction
        moving_locs = df["speed_ms"] > params["moving_threshold_ms"]
        self.calculated_wind_dir = wind_direction(df[moving_locs]["bearing"].values, df[moving_locs]["speed_ms"].values)

    # set VMG (based on straight upwind, straight downwind), and what tack we are on (1 for Starboard)
    def set_vmg_tack_direction(self, df, use_computed=True):
        if use_computed is True:
            wind_dir = self.calculated_wind_dir
        else:
            wind_dir = self.wind_dir

        df["vmg_ms"] = np.cos((df["bearing"] - wind_dir) * np.pi / 180) * df["speed_ms"]
        df["vmg_kts"] = df["vmg_ms"] * 1.94384
        # negative VMG indicates moving downwind
        df["upwind"] = np.sign(df["vmg_ms"])
        df["tack_raw"] = [1 if ((bearing - wind_dir) % 360) < 180 else -1 for bearing in df["bearing"]]

    # Infer maneuvers from gpx points
    def set_maneuver_data(self, df):
        params = self.params
        m = maneuvers(df["speed_ms"].values, df["tack_raw"].values, df["upwind"].values, params)

        df["is_moving"] = m["is_moving"]
        df["tack"] = m["tack"]

        self.crashes = m["crashes"]
        self.port_crashes = m["port_crashes"]
        self.starboard_crashes = m["starboard_crashes"]
        self.stopped_segments = m["stopped_segments"]
        self.moving_segments = m["moving_segments"]

        self.crashes_df = pd.DataFrame(data=m["crash_data"], columns=["Loc", "Tack", "Upwind"])

    # Organize tracks into segments, numbered sequentially, separated by maneuvers
    # Modify upwind/downwind to set not-moving segments
    def set_segments(self, df):
        segment = np.ones(len(df))
        this_segment = 1
        for i in range(1, len(df)):
            # if tack changed from previous point, a new segment starts
            if df.loc[i - 1, "tack"] != df.loc[i, "tack"]:
                this_segment += 1
            segment[i] = this_segment
        df["segment"] = segment
        # 1 is Upwind, -1 is Downwind, and 0 is not moving
        df["upwind"] = df["is_moving"] * df["upwind"]

    # Set transition data
    def set_transitions(self, df):
        transitions_port = []
        transitions_starboard = []
        transitions = []

        tack = df["tack"][0]

        tacks = []
        jibes = []

        for i in range(len(df)):
            if tack != df["tack"][i]:
                if tack == 1 and df["tack"][i] == -1:
                    transitions_port.append(i)

                    if df["upwind"][i] == 1:
                        tacks.append(i)
                        transitions.append([i, "Port", "Tack"])
                    elif df["upwind"][i] == -1:
                        jibes.append(i)
                        transitions.append([i, "Port", "Jibe"])
                if tack == -1 and df["tack"][i] == 1:
                    transitions_starboard.append(i)

                    if df["upwind"][i] == 1:
                        tacks.append(i)
                        transitions.append([i, "Starboard", "Tack"])
                    elif df["upwind"][i] == -1:
                        jibes.append(i)
                        transitions.append([i, "Starboard", "Jibe"])

                tack = df["tack"][i]

        self.transitions_df = pd.DataFrame(data=transitions, columns=["Loc", "Tack", "Maneuver"])
        self.tacks = tacks
        self.jibes = jibes
        self.transitions_port = transitions_port
        self.transitions_starboard = transitions_starboard

    # set stats object
    def set_stats(self):
        df = self.df
        stats = {}

        with np.errstate(invalid="ignore"):
            stats["start_time"] = df["time"].min()
            stats["stop_time"] = df["time"].max()
            stats["duration"] = df["time"].max() - df["time"].min()

            num_crashes = len(self.stopped_segments)
            stats["num_crashes"] = num_crashes
            stats["num_starboard_tack_crashes"] = len(self.starboard_crashes)
            stats["num_port_tack_crashes"] = len(self.port_crashes)

            time_elapsed_minutes = (max(df["time"]) - min(df["time"])).seconds / 60.0
            stats["time_elapsed_minutes"] = time_elapsed_minutes

            successful_transitions = len(self.transitions_port) + len(self.transitions_starboard)
            stats["transition_success_percent"] = np.float64(100) * successful_transitions / (
                        successful_transitions + num_crashes)
            stats["starboard_transition_success_percent"] = np.float64(100) * len(self.transitions_starboard) / (
                        len(self.starboard_crashes) + len(self.transitions_starboard))
            stats["port_transition_success_percent"] = np.float64(100) * len(self.transitions_port) / (
                        len(self.port_crashes) + len(self.transitions_port))
            stats["num_starboard_to_port_transitions"] = len(self.transitions_starboard)
            stats["num_port_to_starboard_transitions"] = len(self.transitions_port)

            transitions_df = self.transitions_df
            crashes_df = self.crashes_df
            stats["port_tack_success_percent"] = np.float64(100) * len(
                transitions_df[(transitions_df["Maneuver"] == "Tack") & (transitions_df["Tack"] == "Port")]) / (
                                                         len(transitions_df[(transitions_df["Maneuver"] == "Tack") & (
                                                                     transitions_df["Tack"] == "Port")]) +
                                                         len(crashes_df[(crashes_df["Upwind"] == "Upwind") & (
                                                                     crashes_df["Tack"] == "Port")]))
            stats["starboard_tack_success_percent"] = np.float64(100) * len(
                transitions_df[(transitions_df["Maneuver"] == "Tack") & (transitions_df["Tack"] == "Starboard")]) / (
                                                              len(transitions_df[
                                                                      (transitions_df["Maneuver"] == "Tack") & (
                                                                                  transitions_df[
                                                                                      "Tack"] == "Starboard")]) +
                                                              len(crashes_df[(crashes_df["Upwind"] == "Upwind") & (
                                                                          crashes_df["Tack"] == "Starboard")]))
            stats["port_jibe_success_percent"] = np.float64(100) * len(
                transitions_df[(transitions_df["Maneuver"] == "Jibe") & (transitions_df["Tack"] == "Port")]) / (
                                                         len(transitions_df[(transitions_df["Maneuver"] == "Jibe") & (
                                                                     transitions_df["Tack"] == "Port")]) +
                                                         len(crashes_df[(crashes_df["Upwind"] == "Downwind") & (
                                                                     crashes_df["Tack"] == "Port")]))
            stats["starboard_jibe_success_percent"] = np.float64(100) * len(
                transitions_df[(transitions_df["Maneuver"] == "Jibe") & (transitions_df["Tack"] == "Starboard")]) / (
                                                              len(transitions_df[
                                                                      (transitions_df["Maneuver"] == "Jibe") & (
                                                                                  transitions_df[
                                                                                      "Tack"] == "Starboard")]) +
                                                              len(crashes_df[(crashes_df["Upwind"] == "Downwind") & (
                                                                          crashes_df["Tack"] == "Starboard")]))
        self.stats = stats

    # set all dataframe columns
    def process_gpx_file(self):
        gpx = self.gpx
        params = self.params

        t = time.time()
        gpx_data, point_list = self.get_gpx_points_list(gpx)
        df = self.create_dataframe(gpx_data, point_list)
        self.df = df

        self.set_wind_directions(df)
        self.set_vmg_tack_direction(df)
        self.set_maneuver_data(df)
        self.set_segments(df)
        self.set_transitions(df)
        self.set_stats()

    def windrose(self, use_calculated_wind_dir=True):
        if use_calculated_wind_dir is True:
            wind_dir = self.calculated_wind_dir
        else:
            wind_dir = self.wind_dir

        n = int(len(self.df))
        wind = list(range(wind_dir - 2, wind_dir + 3))
        ax = windrose.plot_windrose(
            self.df[self.df["is_moving"] == 1],
            var_name="speed_kts",
            direction_name="bearing",
            kind="contourf",
            bins=[x for x in range(10, 30, 2)],
            nsector=180
        )

        ax.legend(title="Speed (kts)")
        ax.contourf(wind * n, [30] * n * len(wind), nsector=360)
        text_offset = -5 if wind_dir >= 270 or wind_dir <= 90 else 5
        ax.text((90 - wind_dir + text_offset) / 180 * np.pi, ax.get_ylim()[1] * 0.75, "Wind Direction", fontsize=10)

    def map(self):
        fig = px.line_mapbox(
            self.df,
            lon="lon",
            lat="lat",
            color="tack",
            line_group="segment",
            mapbox_style="open-street-map",
            hover_data=["speed_kts", "bearing", "tack_raw", "distance_cumulative", "upwind"],
            zoom=14,
            height=1000
        )
        fig.add_trace(go.Scattermapbox(
            lat=self.df["lat"][self.tacks],
            lon=self.df["lon"][self.tacks],
            mode="markers",
            marker=go.scattermapbox.Marker(size=10),
            name="Tack"
        ))
        fig.add_trace(go.Scattermapbox(
            lat=self.df["lat"][self.jibes],
            lon=self.df["lon"][self.jibes],
            mode="markers",
            marker=go.scattermapbox.Marker(size=10),
            name="Jibe"
        ))
        fig.add_trace(go.Scattermapbox(
            lat=self.df["lat"][self.crashes],
            lon=self.df["lon"][self.crashes],
            mode="markers",
            marker=go.scattermapbox.Marker(size=10),
            name="Crash"
        ))
        fig.show()

    def set_minimum_speeds(self):
        """
        Unless stopped_threshold_ms, moving_threshold_ms, and kts_speed_cap are explicitly specified in the params,
        infer them from the data.
        Speeds in the bottom 5% are deemed to be stopped, and the
        :return:
        """
