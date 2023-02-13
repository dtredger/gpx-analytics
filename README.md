# GPX Analytics

This is a jupyter notebook to analyze GPX files. In this case, it is targeted at analysis of GPS traces of sailing.
The logic is a mashup of 
-  [voila-gpx-viewer](https://github.com/jtpio/voila-gpx-viewer)


### Features

Sailing analytics relies on:
- GPS trace map (to see what's what)
- Speed 
- Polar diagram (speed vs True Wind angle)
- 



[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jtpio/voila-gpx-viewer/master?urlpath=voila%2Frender%2Fapp.ipynb)

Experimental GPX Viewer web app built with Jupyter, ipywidgets, ipyleaflet, bqplot and voila

![screencast](https://user-images.githubusercontent.com/591645/60527710-0ff1c680-9cf3-11e9-87b5-8711fd3da344.gif)

## Usage

Create the environment with the dependencies:

```bash
conda env create
conda activate voila-gpx-viewer
```

Open the app:

```bash
voila app.ipynb
```
