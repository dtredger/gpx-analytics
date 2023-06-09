
import time
# from collections import deque
import numpy as np
import pandas as pd
from geopy import distance
import gpxpy

from utils import *


np.seterr('raise')


# Create a new session from a file
class SailingSession:
    default_params = {
        "debug": True,
        # "wind_dir":
        "min_sailing_kts": 1.0,
        "window_size": 5, # TODO: should be time, not number of points?
        # "kts_speed_cap": 40,
        # "timezone": "US/Eastern"
    }

    ### Load data from file. 
    # If data is not included, speed and course data can be calculated.
    # There is no way to infer: hdg_true, roll, or pitch
    def __init__(self, file_path, params={}):
        self.params = self.default_params | params
        self.debug = self.params["debug"]

        if file_path.endswith('.csv'):
            self.from_vakaros_csv(file_path)
        else:       
            with open(file_path, "r") as f:
                gpx = gpxpy.parse(f)
                self.process_gpx_file(gpx)

        # Set basic dataframe params
        self.add_distances()
        self.add_time_steps()
        self.calculate_cog()
        self.calculate_sog_kts()

        # # Infer more complex things
        self.set_inferred_wind_dir()
        self.set_vmg_tack_direction()
        self.set_maneuver_data()
        self.set_segments()
        self.set_transitions()
        self.set_stats()


    ### Vakaros includes the following fields:
    # timestamp, latitude, longitude, sog_kts, cog, hdg_true, roll, pitch
    def from_vakaros_csv(self, file_path):
        ### Format for time string: '%Y-%m-%dT%H:%M:%S.%f'
        self.df = pd.read_csv(file_path, parse_dates=['timestamp'])     


    ### Generic GPX Files contain: 
    # time, latitude, longitude, elevation 
    # Only the first segment of the first track is analyzed.
    def process_gpx_file(self, gpx):
        # gpx_points = self.gpx.routes[0].points
        gpx_points = gpx.tracks[0].segments[0].points
        self.df = pd.DataFrame.from_records([
                (pt.time, pt.latitude, pt.longitude) for pt in gpx_points
            ], columns=['timestamp', 'latitude', 'longitude'])


    # Calculate the distance from one lat/lon point to the next, in meters,
    # and the cumulative distance. Return calculation time.
    #
    # takes ~35s for a dataframe with 50,000 rows
    def add_distances(self):
        start_time = time.time()

        self.df["distance_step"] = [0] + [distance.distance(
            (self.df.loc[i - 1].latitude, self.df.loc[i - 1].longitude),
            (self.df.loc[i].latitude, self.df.loc[i].longitude)).m
        for i in range(1, len(self.df))]
        
        self.df["distance_cumulative"] = np.cumsum(self.df["distance_step"])
        if self.debug: 
            print(f'add_distances elapsed: {time.time() - start_time} s.')


    # Add time gap between this point and previous point, and time elapsed at point
    #
    # takes ~15s for a dataframe with 50,000 rows
    def add_time_steps(self):
        start_time = time.time()

        self.df["time_diff"] = [1] + [
            (self.df.loc[i].timestamp - self.df.loc[i - 1].timestamp).total_seconds() 
        for i in range(1, len(self.df))]

        # cumulative time from first point
        self.df["time_elapsed_sec"] = np.cumsum(self.df["time_diff"])

        if self.debug: 
            print(f'add_time_steps elapsed: {time.time() - start_time} s.')


    # Infer course over ground by comparing two lat/lon points.
    # 
    # TODO: should there be a check to verify impossibly abrupt changes of bearing?    
    def calculate_cog(self):
        # Don't create if this data is already provided
        if 'cog' in self.df.keys():
            print('cog already present in dataframe.')
            return

        heading_decimals = 1
        start_time = time.time()
        # compass bearing between this point and previous point
        self.df["cog"] = [0.0] + [round(calculate_initial_compass_bearing(
            (self.df.loc[i - 1].latitude, self.df.loc[i - 1].longitude),
            (self.df.loc[i].latitude, self.df.loc[i].longitude)), heading_decimals)
        for i in range(1, len(self.df))]

        if self.debug: 
            print(f'calculate_cog elapsed: {time.time() - start_time} s.')
    
    # Infer speed by calculating meters travelled in the time-step between readings.
    # If sog already present, do not calculate.
    def calculate_sog_kts(self):
        # Don't create if this data is already provided
        if 'sog_kts' in self.df.keys():
            print('sog_kts already present in dataframe.')
            return

        start_time = time.time()
        # meters per second converted to kts
        self.df["sog_kts"] = (self.df["distance_step"] / self.df["time_diff"]) * 1.94384

        if self.debug: 
            print(f'calculate_sog_kts elapsed: {time.time() - start_time} s.')


    ### set specified and computed wind directions
    # Wind direction can be set manually with a `wind_dir` session param 
    # 
    # Direction is compass heading set to `self.calculated_wind_dir` 
    def set_inferred_wind_dir(self):
        if "wind_dir" in self.params:
            if self.debug:
                print(f"using preset wind_dir {self.params['wind_dir']}")
            self.preset_wind_dir = self.params["wind_dir"]

        # Only points with speeds faster than min_sailing_kts will be used
        # to calculate wind_direction
        moving_locs = self.df["sog_kts"] > self.params["min_sailing_kts"]
        self.calculated_wind_dir = wind_direction(
                self.df[moving_locs]["cog"].values, 
                self.df[moving_locs]["sog_kts"].values
            )
        if self.debug:
            print(f"calculated wind_dir as {self.calculated_wind_dir}")


    # set VMG (based on straight upwind, straight downwind), and what tack we are on (1 for Starboard)
    def set_vmg_tack_direction(self, use_computed=True):
        wind_dir = self.calculated_wind_dir if use_computed else self.preset_wind_dir

        self.df["vmg_kts"] = np.cos((self.df["cog"] - wind_dir) * np.pi / 180) * self.df["sog_kts"]
        # negative VMG indicates moving downwind
        self.df["upwind"] = np.sign(self.df["vmg_kts"])
        self.df["tack_raw"] = [1 if ((cog - wind_dir) % 360) < 180 else -1 for cog in self.df["cog"]]


    # *** un-cleaned up methods ***

    # Infer maneuvers from data points
    #
    # TODO: why a separate dataframe for this? and some arrays attached to self?
    def set_maneuver_data(self):
        """

        """
        maneuvers = parse_maneuvers(self.df["sog_kts"].values, 
                                    self.df["tack_raw"].values, 
                                    self.df["upwind"].values, 
                                    self.params)

        self.df["is_moving"] = maneuvers["is_moving"]
        self.df["tack"] = maneuvers["tack"]

        self.crashes = maneuvers["crashes"]
        self.port_crashes = maneuvers["port_crashes"]
        self.starboard_crashes = maneuvers["starboard_crashes"]
        self.stopped_segments = maneuvers["stopped_segments"]
        self.moving_segments = maneuvers["moving_segments"]

        self.crashes_df = pd.DataFrame(data=maneuvers["crash_data"], 
                                       columns=["Loc", "Tack", "Upwind"])


    # Organize tracks into segments, numbered sequentially, separated by maneuvers
    # Modify upwind/downwind to set not-moving segments
    def set_segments(self):
        segment = np.ones(len(self.df))
        this_segment = 1
        for i in range(1, len(self.df)):
            # if tack changed from previous point, a new segment starts
            if self.df.loc[i - 1, "tack"] != self.df.loc[i, "tack"]:
                this_segment += 1
            segment[i] = this_segment
        self.df["segment"] = segment
        # 1 is Upwind, -1 is Downwind, and 0 is not moving
        self.df["upwind"] = self.df["is_moving"] * self.df["upwind"]

    # Set transition data
    # Creates transitions_df dataframe
    def set_transitions(self):
        df = self.df
        transitions_port = []
        transitions_starboard = []
        transitions = []

        tack = df["tack"][0]

        tacks = []
        gybes = []

        for i in range(len(df)):
            if tack != df["tack"][i]:
                if tack == 1 and df["tack"][i] == -1:
                    transitions_port.append(i)

                    if df["upwind"][i] == 1:
                        tacks.append(i)
                        transitions.append([i, "Port", "Tack"])
                    elif df["upwind"][i] == -1:
                        gybes.append(i)
                        transitions.append([i, "Port", "Gybe"])
                if tack == -1 and df["tack"][i] == 1:
                    transitions_starboard.append(i)

                    if df["upwind"][i] == 1:
                        tacks.append(i)
                        transitions.append([i, "Starboard", "Tack"])
                    elif df["upwind"][i] == -1:
                        gybes.append(i)
                        transitions.append([i, "Starboard", "Gybe"])

                tack = df["tack"][i]

        self.transitions_df = pd.DataFrame(data=transitions, columns=["Loc", "Tack", "Maneuver"])
        self.tacks = tacks
        self.gybes = gybes
        self.transitions_port = transitions_port
        self.transitions_starboard = transitions_starboard

    # set stats object
    def set_stats(self):
        """
        Crashes are from a kitefoil-focused source, and not really relevant for conventional
        dinghies, but present here anyway. Anytime the boat goes from moving to stopped,
        that is interpreted as a Crash.
        """
        df = self.df
        stats = {}

        with np.errstate(invalid="ignore"):
            stats["start_time"] = df["timestamp"].min()
            stats["stop_time"] = df["timestamp"].max()
            stats["duration"] = df["timestamp"].max() - df["timestamp"].min()

            num_crashes = len(self.stopped_segments)
            stats["num_crashes"] = num_crashes
            stats["num_starboard_tack_crashes"] = len(self.starboard_crashes)
            stats["num_port_tack_crashes"] = len(self.port_crashes)

            time_elapsed_minutes = (max(df["timestamp"]) - min(df["timestamp"])).seconds / 60.0
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
            stats["port_gybe_success_percent"] = np.float64(100) * len(
                transitions_df[(transitions_df["Maneuver"] == "Gybe") & (transitions_df["Tack"] == "Port")]) / (
                                                         len(transitions_df[(transitions_df["Maneuver"] == "Gybe") & (
                                                                     transitions_df["Tack"] == "Port")]) +
                                                         len(crashes_df[(crashes_df["Upwind"] == "Downwind") & (
                                                                     crashes_df["Tack"] == "Port")]))
            stats["starboard_gybe_success_percent"] = np.float64(100) * len(
                transitions_df[(transitions_df["Maneuver"] == "Gybe") & (transitions_df["Tack"] == "Starboard")]) / (
                                                              len(transitions_df[
                                                                      (transitions_df["Maneuver"] == "Gybe") & (
                                                                                  transitions_df[
                                                                                      "Tack"] == "Starboard")]) +
                                                              len(crashes_df[(crashes_df["Upwind"] == "Downwind") & (
                                                                          crashes_df["Tack"] == "Starboard")]))
        self.stats = stats










