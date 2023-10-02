import numpy as np
from matplotlib.patches import Polygon


def shipOutline(xp,yp,psi,outl):
    '''
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

    '''

    pos=np.array([xp,yp])

    #rotate
    rotM=np.array([[np.cos(psi),np.sin(psi)],[-np.sin(psi),np.cos(psi)]])

    a=[np.dot(rotM,x) for x in outl]
    rotship=Polygon(a, closed=True,
                          fill=False)

    #translate

    b=[np.add(pos,x) for x in a]
    transship=Polygon(b, closed=True,fill=False)
    return transship