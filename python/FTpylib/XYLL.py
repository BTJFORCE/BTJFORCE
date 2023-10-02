import numpy as np
class XYLL:
    def __init__(self, lat_origo=0, lng_origo=0):
        self.set_origo(lat_origo ,lng_origo)  #        // long. [rad]
        self._fEarthRadius = 6366707.0# [m]

    def set_origo(self,lat_origo, lng_origo):
        self._lat_origo = lat_origo #        // lat.  [rad]
        self._lng_origo = lng_origo #        // long. [rad]




#! Get location where tangent plane touches the Earth
#
# @param fOrigin  Location where tangent plane touches the Earth (lat,long). [rad]
#
    def get_origin(self):
        return self._lat_origo,self._lng_origo

#/////////////////////////////////////////////////////////////////////////
#--- Projection

#! Projection from Earth to tangent plane
#*!
# * @param fLatLong  Point on Earth. (lat,long). [rad]
# * @param fXY       Point on tangent plane. [m]
 #*/
    def llxy(self,lat,lng):
        dlat = lat - self._lat_origo
        dlng = lng - self._lng_origo
        x = self._fEarthRadius * (dlat + .5 * np.sin(lat)*np.cos(lat) * dlng*dlng) 
        y = self._fEarthRadius * np.cos(lat) *dlng 
        return x,y


#! Projection from tangent plane to Earth
#/*!
# * @param fXY       Point on tangent plane. [m]
# * @param fLatLong  Point on Earth. (lat,long). [rad]
# */
    def xyll(self,x, y):
        EPS  = (.001/self._fEarthRadius) # stop within 1 mm
   
        fLat = self._lat_origo
        a = self._lat_origo + x /self._fEarthRadius
        b = 0.5 * (y/self._fEarthRadius) **2 # *(y/my_fEarthRadius);
        count = 0
        while True:

            dLat  = - (fLat + b * np.tan(fLat) - a)  \
                / (1. + b/(np.cos(fLat)*np.cos(fLat)))

            fLat = fLat + dLat
 
            count +=1

            if ( np.abs(dLat) > EPS or count > 100):
                break
        
        fLng = self._lng_origo + y / self._fEarthRadius / np.cos(fLat)
        return fLat,fLng
