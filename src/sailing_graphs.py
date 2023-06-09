import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



### Map the sailing session
# default trace color is stbd/port tack
# markers for tacks, gybes, and crashes
def render_chart(session, crash_enabled=False):
    """
    possible chart styles are:
        'open-street-map', 
        'white-bg', 
        'carto-positron', 
        'carto-darkmatter', 
        'stamen- terrain', 
        'stamen-toner', 
        'stamen-watercolor'.
    """
    fig = px.line_mapbox(
        session.df,
        lat="latitude",
        lon="longitude",
        color="tack",
        line_group="segment",
        mapbox_style='carto-positron', #"open-street-map",
        hover_data=["sog_kts", "cog", "distance_cumulative", "upwind"],
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
        lat=session.df["latitude"][session.gybes],
        lon=session.df["longitude"][session.gybes],
        mode="markers",
        marker=go.scattermapbox.Marker(size=10),
        name="Gybe"
    ))
    if crash_enabled:
        fig.add_trace(go.Scattermapbox(
            lat=session.df["latitude"][session.crashes],
            lon=session.df["longitude"][session.crashes],
            mode="markers",
            marker=go.scattermapbox.Marker(size=10),
            name="Crash"
        ))
    fig.show()


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

# Polar diagram showing 
def new_polar(session, wind_dir=False, bin_deg_size=3):
    if wind_dir == False:
        wind_dir = session.calculated_wind_dir
    elif wind_dir == True:
        wind_dir = session.preset_wind_dir

    pf = pd.DataFrame(columns=['cog', 'top', 'mean', 'tack_raw'])
    bin_mod = round((bin_deg_size - 1) / 2)

    for x in range(0, 361):
        rows = session.df.loc[session.df["cog"].isin([x-bin_mod, x, x+bin_mod])]
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


# new_polar(s)