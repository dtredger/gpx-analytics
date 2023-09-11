
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
        "start":0,
        "end": None,
        # "wind_dir":
        "min_sailing_kts": 1.0,
        "window_size": 5, # TODO: should be time, not number of points?
        # "kts_speed_cap": 40,
        # "timezone": "US/Eastern"
    }

    def __init__(self, file_path, params={}):
        """
        Load data from file. 
        If data is not included, speed and course data can be calculated.
        There is no way to infer: hdg_true, roll, or pitch
        """
        self.params = self.default_params | params
        self.debug = self.params["debug"]

        if file_path.endswith('.csv'):
            self.from_vakaros_csv(file_path)
        else:       
            with open(file_path, "r") as f:
                gpx = gpxpy.parse(f)
                self.process_gpx_file(gpx)

        # Set Sample rate
        self.set_sample_rate()

        # Set basic dataframe params
        self.add_distances()
        self.add_time_steps()
        self.calculate_cog()
        self.calculate_sog_kts()
        self.calculate_leeway()

        # Add ability to filter df with widgets
        self.filtered_df = self.df
        self.course_points = []
        self.course_axis = None

        # Infer more complex things
        self.infer_data()
        self.filtered_df = self.df

    def infer_data(self, start=0, end=None):
        """
        Use data to make inferences about other data or actions.
        When the dataframe is filtered, these inferences should be 
        recalculated.
        """
        self.df = self.filtered_df.loc[start:end].copy()
        self.infer_wind_dir()
        self.infer_course_axis()
        self.infer_vmg_tack_direction()
        self.set_maneuver_data()
        self.set_segments()
        self.set_transitions()
        self.set_stats()



    ### Vakaros includes the following fields:
    # timestamp, latitude, longitude, sog_kts, cog, hdg_true, roll, pitch
    def from_vakaros_csv(self, file_path):
        ### Format for time string: '%Y-%m-%dT%H:%M:%S.%f'
        self.raw_df = pd.read_csv(file_path, parse_dates=['timestamp'])


    ### Generic GPX Files contain: 
    # time, latitude, longitude, elevation 
    # Only the first segment of the first track is analyzed.
    def process_gpx_file(self, gpx):
        # gpx_points = self.gpx.routes[0].points
        gpx_points = gpx.tracks[0].segments[0].points
        self.raw_df = pd.DataFrame.from_records([
                (pt.time, pt.latitude, pt.longitude) for pt in gpx_points
            ], columns=['timestamp', 'latitude', 'longitude'])


    def set_sample_rate(self, sample_rate='0.5S'):
        """
        Create an average for the given sample_rate to smooth the data and reduce the number
        of data points to be plotted on the map.
        Resampling will set timestamp as the index, so reset to restore a numeric index.
        """
        self.df = self.raw_df.resample(sample_rate, on='timestamp').mean().bfill().reset_index()

    # Calculate the distance from one lat/lon point to the next, in meters,
    # and the cumulative distance. Return calculation time.
    #
    # Drop rows where the distance step seems like a misreading
    #
    # takes ~35s for a dataframe with 50,000 rows
    def add_distances(self):
        start_time = time.time()

        self.df["distance_step"] = [0] + [distance.distance(
            (self.df.loc[i - 1].latitude, self.df.loc[i - 1].longitude),
            (self.df.loc[i].latitude, self.df.loc[i].longitude)).m
        for i in range(1, len(self.df))]

        # a distance step of 50m = almost 200kts (given avg time_diff of 0.5)
        self.df = self.df[self.df["distance_step"] < 50].reset_index(drop=True)
        
        self.df.loc[:, "distance_cumulative"] = np.cumsum(self.df["distance_step"])
        if self.debug: 
            print(f'add_distances elapsed: {time.time() - start_time} s.')


    # Calculate change in distance to the mark (provided) to work out vmg~ish
    # will include lateral motion on the course (not just upwind/down) 
    # removing that would require complex math involving wind dir, or a straight
    # line between the downwind/upwind marks -_-;


    # Add time gap between this point and previous point, and time elapsed at point
    #
    # takes ~15s for a dataframe with 50,000 rows
    def add_time_steps(self):
        start_time = time.time()

        self.df["time_diff"] = [1] + [
            (self.df.loc[i].timestamp - self.df.loc[i - 1].timestamp).total_seconds() 
        for i in range(1, len(self.df))]

        # cumulative time from first point
        self.df.loc[:, "time_elapsed_sec"] = np.cumsum(self.df["time_diff"])

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
        self.df.loc[:, "cog"] = [0.0] + [round(calculate_initial_compass_bearing(
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


    def calculate_leeway(self):
        """
        calculate the difference between heading and course over ground
        requires hdg_true to be set (from gps).
        sog, cog and hd_true are columns 4, 5 and 6 in the row
        """
        def leeway_calc(row):
            if row[3] < self.params['min_sailing_kts']:
                return 0
            diff = abs(row[4] - row[5])
            if diff > 180:
                diff = 360 - diff
            return diff

        self.df.loc[:, 'leeway'] = self.df.apply(leeway_calc, axis=1)
        


    ### set specified and computed wind directions
    # Wind direction can be set manually with a `wind_dir` session param 
    # 
    # Direction is compass heading set to `self.calculated_wind_dir` 
    def infer_wind_dir(self):
        if "wind_dir" in self.params:
            if self.debug:
                print(f"using preset wind_dir {self.params['wind_dir']}")
            self.wind_dir = self.params["wind_dir"]
        else:
            # Only points with speeds faster than min_sailing_kts will be used
            # to calculate wind_direction
            moving_locs = self.df["sog_kts"] > self.params["min_sailing_kts"]
            self.wind_dir = wind_direction(
                    self.df[moving_locs]["cog"].values, 
                    self.df[moving_locs]["sog_kts"].values
                )
            if self.debug:
                print(f"inferred wind_dir as: {self.wind_dir} degrees")

    def infer_course_axis(self):
        """
        Set the course from lwd to windward mark, based on line between two marks.
        """
        if len(self.course_points) == 2:
            pt_1, pt_2 = [tuple(pt) for pt in self.course_points]
            self.course_axis = calculate_initial_compass_bearing(pt_1, pt_2)
            print(f"inferred course axis from line between leeward & windward mark as {self.course_axis} degrees")
        else:
            print("no points to infer course axis from")

    # set VMG (based on straight upwind, straight downwind), and what tack we are on (1 for Starboard)
    # TODO: allow course axis to be different from wind_dir
    def infer_vmg_tack_direction(self):
        course_axis = self.wind_dir

        self.df.loc[:, "vmg_kts"] = np.cos((self.df.loc[:, "cog"] - course_axis) * np.pi / 180) * self.df.loc[:, "sog_kts"]
        # negative VMG indicates moving downwind
        self.df.loc[:, "upwind"] = np.sign(self.df.loc[:, "vmg_kts"])
        self.df.loc[:, "tack_raw"] = [1 if ((cog - course_axis) % 360) < 180 else -1 for cog in self.df["cog"]]

 
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

        self.df.loc[:, "is_moving"] = maneuvers["is_moving"]
        self.df.loc[:, "tack"] = maneuvers["tack"]

        self.crashes = maneuvers["crashes"]
        self.port_crashes = maneuvers["port_crashes"]
        self.starboard_crashes = maneuvers["starboard_crashes"]
        self.stopped_segments = maneuvers["stopped_segments"]
        self.moving_segments = maneuvers["moving_segments"]

        self.crashes_df = pd.DataFrame(data=maneuvers["crash_data"], 
                                       columns=["Loc", "Tack", "Upwind"])


    # Organize tracks into segments, numbered sequentially, separated by maneuvers
    # Modify upwind/downwind to set not-moving segments
    # also start new segments for roundings (from upwind -> downwind, or reverse)
    # TODO - this duplicates the maneuvers calculation
    def set_segments(self):
        df = self.df
        segment = 1

        def has_tacked(index):
            try:
                if index == 0: return False
                return df.loc[index, 'tack'] != df.loc[index - 1, 'tack']
            except KeyError:
                return False

        for index, row in df.iterrows():
            if has_tacked(index):
                segment += 1
            df.loc[index, 'segment'] = segment

    # Set transition data
    # Creates transitions_df dataframe
    # Add Roundings 
    def set_transitions(self):
        df = self.df
        transitions_port = []
        transitions_starboard = []
        transitions = []

        tack = df["tack"].iloc[0]
        upwind = df["upwind"].iloc[0]

        tacks = []
        gybes = []
        roundings = []

        for i, row in df.iterrows():
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

            # add rounding when upwind changes (either direction)
            if upwind != df["upwind"][i]:
                roundings.append(i)
                upwind = df["upwind"][i]

        self.transitions_df = pd.DataFrame(data=transitions, columns=["Loc", "Tack", "Maneuver"])
        self.tacks = tacks
        self.gybes = gybes
        self.roundings = roundings
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










