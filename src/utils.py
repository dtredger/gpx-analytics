import math
import numpy as np


### Code taken from other people/places used unmodified


# https://gist.github.com/jeromer/2005586
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
# window of time vs min_sailing_kts
# @param speed_window [Array]
# @param params [Dict]
def is_craft_moving(speed_window, params):
    # TODO (why use both params?)
    return (np.mean(speed_window) >= params["min_sailing_kts"] )



# This chooses a wind direction that is in the middle of the range of directions that are almost-never-travelled
# 
# adopted from kitefoil repo (don't remember name)
#
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




# create count of maneuvers
def parse_maneuvers(sog_kts, tack_raw, upwind, params):
    """
    "is_moving": is_moving,
    "tack": tack,
    "crashes": crashes,
    "crash_data": crash_data,
    "port_crashes": port_crashes,
    "starboard_crashes": starboard_crashes,
    "stopped_segments": stopped_segments,
    "moving_segments": moving_segments}
    """
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
    is_moving = np.zeros(len(sog_kts))
    is_moving[:params["window_size"]] = 1 if is_craft_moving(sog_kts[window], params) else 0
    # Set all points in the window to the median tack in the current window
    tack = np.zeros(len(sog_kts))
    tack[:params["window_size"]] = np.median(tack_raw[window])

    # Iterate over every speed point
    for i in range(1, len(sog_kts)):
        start = min(i, len(sog_kts) - params["window_size"])
        end = min(i + params["window_size"], len(sog_kts))
        # Adjust the window to the current point and the `window_size` points ahead
        window = range(start, end)
        # Average speed within the current window
        window_sog_kts = np.mean(sog_kts[window])
        # was the craft moving at the last recorded point?
        if is_moving[i - 1] == 1:
            is_moving[i] = 1
            if (tack[i - 1] != tack_raw[i]) and (len(set(tack_raw[window])) == 1):
                # only update tack if consistent across the window
                # if we're updating tack, scroll back that update earlier in time
                tack[max(i - params["window_size"] + 1, 0):i] = np.median(tack_raw[window])
            tack[i] = tack[i - 1]
            # craft is not below `stopped` threshold
            if window_sog_kts > params["min_sailing_kts"]:
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
            if window_sog_kts < params["min_sailing_kts"]:
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
        "moving_segments": moving_segments }







