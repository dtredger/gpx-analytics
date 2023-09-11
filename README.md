# GPX Analytics

This is a jupyter notebook to analyze GPX files. In this case, it is targeted at analysis of GPS traces of sailing.
The logic is a mashup of [voila-gpx-viewer](https://github.com/jtpio/voila-gpx-viewer) and a kitefoil analytics (the name of which I forget).


### Features

Sailing analytics relies on:
- GPS trace map (to see what's what)
- Speed 
- Polar diagram (speed vs True Wind angle)
- 

Run it with:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dtredger/gpx-analytics/HEAD?labpath=%2Fapp.ipynb)
or 
[Google Colab:](https://colab.research.google.com/github/dtredger/gpx-analytics/blob/master/app.ipynb)


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
