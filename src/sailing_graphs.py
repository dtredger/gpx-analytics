import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go




def polar(session, use_calculated_wind_dir=True):
    if use_calculated_wind_dir is True:
        wind_dir = session.calculated_wind_dir
    else:
        wind_dir = session.wind_dir

    n = int(len(session.df))
    wind = list(range(wind_dir - 2, wind_dir + 3))
    ax = windrose.plot_windrose(
        session.df[session.df["is_moving"] == 1],
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


### Map the sailing session
# default trace color is stbd/port tack
# markers for tacks, gybes, and crashes
def map(session):
    fig = px.line_mapbox(
        session.df,
        lat="latitude",
        lon="longitude",
        color="tack_raw",
        # line_group="segment",
        mapbox_style="open-street-map",
        hover_data=["speed_kts", "cog", "tack_raw", "distance_cumulative", "upwind"],
        zoom=14,
        height=1000
    )
    fig.add_trace(go.Scattermapbox(
        lat=session.df["latitude"][session.tacks],
        lon=session.df["longitude"][session.tacks],
        mode="markers",
        marker=go.scattermapbox.Marker(size=10),
        name="Tack"
    ))
    fig.add_trace(go.Scattermapbox(
        lat=session.df["latitude"][session.jibes],
        lon=session.df["longitude"][session.jibes],
        mode="markers",
        marker=go.scattermapbox.Marker(size=10),
        name="Jibe"
    ))
    fig.add_trace(go.Scattermapbox(
        lat=session.df["latitude"][session.crashes],
        lon=session.df["longitude"][session.crashes],
        mode="markers",
        marker=go.scattermapbox.Marker(size=10),
        name="Crash"
    ))
    fig.show()


#  Starboard tracks are 
def add_traces(pf, twd):
    if pf.tack_raw.mean() < 0:
        colour = '#6AFF33'
        max_colour = '#46FF01'
    else:
        colour = '#C02002'
        max_colour = '#FF2800'

    return go.Scatterpolar(r = pf['top'],
                           theta = pf['bearing'],
                           mode = 'lines',
                           name = '95%',
                           line_color = colour)


# # df = session.df
# # def bins():
# # df.loc[df['bearing'].between(0, 90, 'both'), 'grade'] = 'C'
# # df.loc[df['score'].between(91, 269, 'both'), 'grade'] = 'B'
# # df.loc[df['score'].between(270, 360, 'both'), 'grade'] = 'A'

# pf = pd.DataFrame(columns=['bearing', 'top', 'mean', 'tack_raw'])

# for x in range(0, 361):
#     rows = df.loc[df["bearing"] == (x)]
#     pf.loc[x] =  { 'bearing': x, 
#                    'top': rows.speed_ms.quantile(0.9), 
#                    'mean': rows.speed_ms.quantile(0.5),
#                    'tack_raw': rows.tack_raw.mean() }



def new_polar(polar_frame, twd):
	pf = polar_frame
    fig = go.Figure()
    fig.update_layout(autosize=False, 
                      width=700, 
                      height=700, 
                      polar_angularaxis_direction='clockwise')
    stbd = pf.loc[pf["tack_raw"] > 0]
    port = pf.loc[pf["tack_raw"] < 0]

    fig.add_trace(add_traces(stbd, twd))
    fig.add_trace(add_traces(port, twd))
    
    # fig.add_trace(go.Scatterpolar(r = polar_frame['top'],
    #                               theta = polar_frame['bearing'],
    #                               mode = 'lines',
    #                               name = '95%',
    #                               line_color = 'peru'))
    # fig.add_trace(go.Scatterpolar(r = polar_frame['mean'],
    #                               theta = polar_frame['bearing'],
    #                               mode = 'lines',
    #                               name = 'Mean',
    #                               fill = 'green',
    #                               line_color = 'darkviolet'))
    fig.add_trace(go.Barpolar(
        r=[10],
        theta=[twd],
        width=[20],
        marker_color=["gray"],
        marker_line_color="black",
        marker_line_width=2,
        opacity=0.6,
        name="True Wind Direction"
    ))
    fig.update_layout(
        title = 'Speed Polar',
        showlegend = True
    ).show()


