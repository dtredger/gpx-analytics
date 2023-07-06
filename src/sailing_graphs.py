import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



### Map the sailing session
# default trace color is stbd/port tack
# markers for tacks, gybes, and crashes
# uses session.filtered_df for data, so modify df with widgets to use subset
# of points
def render_chart(session, crash_enabled=False, boat_enabled=False):
    filtered_df = session.filtered_df
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
def render_polar(session, bin_deg_size=3):
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


def render_speed(session):
    fig = px.line(session.df, x="timestamp", y="sog_kts", title="Speed (kts)")
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode=mode).show()