# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
#%matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import datetime
from io import StringIO
import plotly.express as px
import glob
from FTpylib import XYLL
from matplotlib.patches import Polygon

# %%
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

# %%


# %%
def shipOutline(xp, yp, psi, outl):
    """
    rotate a polygon psi (radians) and translate it to the point xp,yp
    and translate a polygon
    the outlinepolygon must be defind with the bow pointing North
    and the coordinate system used is a NESW system
    returns a Polygon
    ====================================
    Lpp=1
    Beam=1
    outl=np.loadtxt(r'U:\ships\3789\3789_outl.dat')
    outl[:,0]=outl[:,0]*Beam
    outl[:,1]=outl[:,1]*Lpp
    ship_polygon = Polygon(outl, closed=True,
                      fill=False)

    """

    pos = np.array([xp, yp])

    # rotate
    rotM = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])

    a = [np.dot(rotM, x) for x in outl]
    rotship = Polygon(a, closed=True, fill=False)

    # translate

    b = [np.add(pos, x) for x in a]
    transship = Polygon(b, closed=True, fill=False)
    return transship


def distDir(xe1, ye1, xe2, ye2):
    """
    Calculates distance and bearing from point1 to point2
    in an local earth fixed frame with x-axis pointing towards north
    """
    dist = np.sqrt((xe2 - xe1) ** 2 + (ye2 - ye1) ** 2)
    direction = np.arctan2([ye2 - ye1], [xe2 - xe1])
    return dist, np.degrees(direction).item(0)


# print (distDir(0,0,0,0))
# print (distDir(0,0,0,1))
# print (distDir(0,0,-1,0))
# print (distDir(0,0,0,-1))
# print (distDir(0,0,-1,-1))


def pos2decdeg(val):
    deg = int(val / 100)
    minutes = (val / 100 - deg) * 100
    decdeg = deg + minutes / 60.0
    return decdeg


# %%
Lpp = 1
Beam = 1
outl = np.loadtxt(r"U:\ships\3789\3789_outl.dat")


# %%
# plot port and distance points
#
origo = np.array([np.radians(54.57363333), np.radians(11.92475)])  ## Gedser berth
point_G1 = np.array([-88.51367952, 62.97410853])
point_G2 = np.array([-272.31321847, 154.5955789])
point_G3 = np.array([-296.4857112, 73.57014819])
col_points = np.array([origo, point_G1, point_G2, point_G3])
point_G4 = np.array([-400.0, 300])
point_G5 = np.array([-400.0, 73.57])

point_G6 = np.array([-900.0, 600])
point_G7 = np.array([-900.0, 0])

line1 = [point_G4, point_G5]
xyll = XYLL.XYLL(origo[0], origo[1])


# green buoys
gb1latlng = (54 + 34.205999 / 60.0, 11 + 55.489999 / 60.0)
gb2latlng = (54 + 33.910748 / 60.0, 11 + 55.652174 / 60.0)
gb3latlng = (54 + 34.005850 / 60.0, 11 + 56.012984 / 60.0)
gb1 = xyll.llxy(np.radians(gb1latlng[0]), np.radians(gb1latlng[1]))
gb2 = xyll.llxy(np.radians(gb2latlng[0]), np.radians(gb2latlng[1]))
gb3 = xyll.llxy(np.radians(gb3latlng[0]), np.radians(gb3latlng[1]))
# red buoys
rb1latlng = (54 + 34.120951 / 60.0, 11 + 55.976734 / 60.0)
rb2latlng = (54 + 33.924996 / 60.0, 11 + 55.801036 / 60.0)
rb1 = xyll.llxy(np.radians(rb1latlng[0]), np.radians(rb1latlng[1]))
rb2 = xyll.llxy(np.radians(rb2latlng[0]), np.radians(rb2latlng[1]))


gedros = pd.read_csv(
    "z:/Tasks/121/121-20658/04 Technical/GedserOutlinePolygon/gedros_data.dat",
    sep="\s+|;|:",
    header=None,
    engine="python",
)


gedros["lat"] = gedros[0] + (gedros[1] + gedros[2] / 1000.0) / 60
gedros["lon"] = gedros[3] + (gedros[4] + gedros[5] / 1000.0) / 60

West_Break_Water = (54 + (34 + 257.910 / 1000.0) / 60, 11 + (55 + 553.527 / 1000.0) / 60)
East_Break_Water = (54 + (34 + 270.961 / 1000.0) / 60, 11 + (55 + 628.999 / 1000.0) / 60)
Ferry_berth_South = (54 + (34 + 370.206 / 1000.0) / 60, 11 + (55 + 543.660 / 1000.0) / 60)


ged_poly = []

for ix in range(len(gedros["lat"].values)):
    p = xyll.llxy(gedros["lat"].values[ix] * np.pi / 180.0, gedros["lon"].values[ix] * np.pi / 180.0)
    ged_poly.append(p)
ged_poly = np.array(ged_poly)

# col_points=ged_poly[[220,36,127]]

# %%
#
# dfReal=pd.read_csv()
df = pd.read_csv(r"C:\temp\rk2021\3789\data\AInn_AI.csv", header=0, delimiter=",")
# %%

fig, ax = plt.subplots(figsize=(8, 5))
# plt.plot(ged_poly[:,1],ged_poly[:,0],'*',color='k')
# plt.plot(col_points[:,1],col_points[:,0],'*',color='k')
for ix in range(len(col_points)):
    p_ix = col_points[ix]
    plt.plot(p_ix[1], p_ix[0], "*", color="k")
    ax.annotate(
        str(ix),
        xy=(p_ix[1], p_ix[0]),
        xytext=(0, 3),
        textcoords="offset points",
    )
plt.scatter([gb1[1], gb2[1], gb3[1]], [gb1[0], gb2[0], gb3[0]], color="g")
plt.scatter([rb1[1], rb2[1]], [rb1[0], rb2[0]], color="r")

plt.plot(ged_poly[:, 1], ged_poly[:, 0], "--", color="r")
# plt.plot(transit1_track[0:npt,1],transit1_track[0:npt,0],label='recorded')
plt.plot(df.y, df.x, label="GAN-sailing")

# event lines
x_vals = [point_G4[0], point_G5[0]]
y_vals = [point_G4[1], point_G5[1]]
plt.plot(y_vals, x_vals, "--", color="k")

x_vals = [point_G6[0], point_G7[0]]
y_vals = [point_G6[1], point_G7[1]]
plt.plot(y_vals, x_vals, "--", color="k")

plt.plot([gb1[1], gb2[1]], [gb1[0], gb2[0]], "--", color="g")

poly = shipOutline(origo[0], origo[1], 5.916491631335578, outl)
ax.add_patch(poly)

for index, row in df.iterrows():
    # print (row["Name"], row["Age"])
    if row["time[s]"] % 60 == 0:
        poly = shipOutline(row["y"], row["x"], row["psi"], outl)
        ax.add_patch(poly)
ax.set_aspect("equal", "datalim")
plt.title("Gedser port with distance points")
plt.legend()

# %%
head = ["distLead", "dist0", "Heading", "COG", "SOG", "STW_UW", "STW_VW", "SOG_U", "SOG_V", "Turnrate", "depth"]


state = pd.read_csv(
    r"Z:\Tasks\121\121-20658\04 Technical\GedserExpertTracksWithTurn\transit_10_22_2019_022853_state.csv",
    header=0,
    names=head,
    delimiter=",",
)


# %%
app = dash.Dash()

fig = go.Figure()

fig.add_trace(go.Scatter(x=ged_poly[:, 1], y=ged_poly[:, 0]))
fig.add_trace(go.Scatter(x=df.y, y=df.x))
for index, row in df.iterrows():
    # print (row["Name"], row["Age"])
    if row["time[s]"] % 60 == 0:

        poly = shipOutline(row["y"], row["x"], row["psi"], outl)
        fig.add_trace(go.Scatter(x=poly.get_xy()[:, 0], y=poly.get_xy()[:, 1]))


fig.update_layout(
    width=800,
    height=500,
    title="GAN sailing",
)
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)


app.layout = html.Div(children=[html.H1("Dash DK1"), dcc.Graph(id="gedser", figure=fig)])


# %%

if __name__ == "__main__":
    app.run_server()
# %%
