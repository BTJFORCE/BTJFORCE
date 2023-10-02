import math
import copy

import numpy as np
import pandas as pd
from pydriver.dk1driver.IRL import GAIL
from pydriver.dk1driver.scenarios.ScenarioBase import ScenarioBase
from pydriver.dk1driver.generated.devices import MSM

## Sensor signals from VDR
## SOG,
##
##


class TrainAInetwork(ScenarioBase):
    """implementation VDR recordings"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.actiontags = {
            "PD_S": 0,
            "CentRPM_S_abs": 1,
            "CentRud_S": 2,
            "StbdRPM_S_abs": 3,
            "cosStbdDir_S": 4,
            "sinStbdDir_S": 5,
            "PortRPM_S_abs": 6,
            "cosPortDir_S": 7,
            "sinPortDir_S": 8,
            "Bow1RPM_C": 9,
            "Bow2RPM_C": 10,
        }
        self.max_action = np.array([100.0, 150.0, 35.0, 150, 1.0, 1.0, 150.0, 1, 1, 100, 100], dtype=float)
        self.max_state = [132.36330113, 1109.96408521, 351.12, 232.1, 8.3, 7.9, 2.8, 8.2, 1.3, 14.01]
        # ,
        # 28.3  ]
        # self.statetags=['dist0','dir0','dist1','dir1', 'dist2','dir2', 'dist3','dir3',
        #    'Heading','COG','SOG','STW_UW', 'STW_VW', 'SOG_U',
        #    'SOG_V', 'Turnrate', 'depth']
        self.statetags = [
            "distLead",
            "dist0",
            "Heading",
            "COG",
            "SOG",
            "STW_UW",
            "STW_VW",
            "SOG_U",
            "SOG_V",
            "Turnrate",
        ]

        self.stateix = {}
        for ix, val in enumerate(self.statetags):
            self.stateix[val] = ix

        self.state = np.zeros(len(self.statetags))

        self.origo = np.array([np.radians(54.57363333), np.radians(11.92475)])  ## Gedser berth
        self.G1 = np.array([-88.51367952, 62.97410853])
        self.G2 = np.array([-272.31321847, 154.5955789])
        self.G3 = np.array([-296.4857112, 73.57014819])

        # leading line gedser
        Np = (0.0, 0.0)  # origo North point
        Sp = (self.G2 + self.G3) / 2.0

        # line between points
        r = Sp
        self.endP = 1.5 * r

        # graph logging variables:
        self.epochs = []
        self.reward = 0

        # Data used for the scenario
        self._handleCommandfileName = "z:\\Tasks\\119\\119-25095\\Task1.4\\Berlin\\cmnd_interpolated\\transit_10_22_2019_085529_cmnd.dat"  # kwargs.get('VDRfile')
        self._initFilename = self._handleCommandfileName.replace("_cmnd", "_init")
        self._dfHandleCommands = pd.read_csv(self._handleCommandfileName, sep=",", header=0)
        df = self._dfHandleCommands
        if df.isnull().values.any():
            df.fillna(method="ffill", inplace=True)
            if df.isnull().values.any():
                df.fillna(method="bfill", inplace=True)

        self._index = 0  ## keep track of active index

        self._P = -0.1  ## Proportional gain
        self._D = 20.0  ## Differential gain

    def setPolicy(self, policy):
        self.policy = policy

    def getReward(self):
        return self.reward

    def distDir(self, xe1, ye1, xe2, ye2):
        """
        Calculates distance and bearing from point1 to point2
        in an local earth fixed frame with x-axis pointing towards north
        """
        dist = np.sqrt((xe2 - xe1) ** 2 + (ye2 - ye1) ** 2)
        direction = np.arctan2([ye2 - ye1], [xe2 - xe1])
        return dist, np.degrees(direction).item(0)

    def distLeadingLine(self, shipPos, p1, p2):
        d = np.cross(p2 - p1, shipPos - p1) / np.linalg.norm(p2 - p1)
        return d

    def loadShip(self, driverApp):
        """"""
        super().loadShip(driverApp)
        self._ship = driverApp.getShip()
        self._MSMIf = MSM.MSM(driverApp.sharedMemoryInterface)
        # set initial Ship State (position + velocity)
        self.init_state = {}
        with open(self._initFilename, "r") as f:
            for line in f:
                line = line.strip()
                (key, val) = line.split(",")
                self.init_state[key] = val
        linearPosition = np.array([float(self.init_state["x"]), float(self.init_state["y"]), 0.0])
        angularPosition = np.array([0.0, 0.0, float(self.init_state["psi"])])

        # set initial speed
        linearVelocity = np.array([float(self.init_state["u"]), float(self.init_state["v"]), 0.0])
        angularVelocity = np.array([0.0, 0.0, float(self.init_state["r"])])

        self._ship.reset(linearPosition, angularPosition, linearVelocity, angularVelocity)

        self.updateState()

    def preStep(self, driverApp, time):
        """"""
        # log initial conditions
        if time == 0.0:
            self.log(time)

    def timeStep(self, driverApp, time):

        state = self.getState()
        state /= self.max_state
        action = self.policy.select_action(state)
        action *= self.max_action
        """
        # no matter how we sail apply the same relative wind as measure by modifying the true wind with the ship speed
        knot2meterPerSec = 0.51444
        wspeedRelative = self._dfHandleCommands.iloc[self._index].W_Speed_R_kn * knot2meterPerSec
        wdirRelative =  np.radians(self._dfHandleCommands.iloc[self._index].W_Dir_R)
        uWindGBArelative = wspeedRelative * np.cos(wdirRelative+np.pi)
        vWindGBArelative = wspeedRelative * np.sin(wdirRelative+np.pi)

        velocityTrueGBA = self._ship.getVelocityOverBottom()
        uwTrueGBA =   velocityTrueGBA[0] - uWindGBArelative
        vwTrueGBA =   velocityTrueGBA[1] - vWindGBArelative

        position = self._ship.getPosition()
        headingShip = position[5]
        windDirTrue = np.arctan2(vwTrueGBA, uwTrueGBA) + headingShip

        if windDirTrue < 0.0:
            windDirTrue += 2 * np.pi
        if windDirTrue > 2 * np.pi :
             windDirTrue -= 2 * np.pi

        # Set Scenario specific environment
        self._environment.wind.speed = self._dfHandleCommands.iloc[self._index].Speed_MS_T #np.sqrt(uwTrueGBA**2 + vwTrueGBA**2)
        self._environment.wind.direction = np.radians(self._dfHandleCommands.iloc[self._index].W_Dir_T) #windDirTrue

        """
        # should actually look depth and wind
        self._environment.depth = 6.3  # self._dfHandleCommands.iloc[self._index].depth + self._ship.getDraught()
        self._ship.environment = self._environment
        # Extract time dependent command data to be forwarded to devices
        # commands= self._dfHandleCommands.iloc[self._index]

        # Conversion factor
        RPM2RPS = 1.0 / 60.0
        PERCENT2FRACTION = 1.0 / 100.0

        # Set Azimuth Propeller Revolutions [1/sec]
        REVS_S = action[self.actiontags["StbdRPM_S_abs"]] * RPM2RPS
        REVS_P = action[self.actiontags["PortRPM_S_abs"]] * RPM2RPS
        if REVS_S < 100 * RPM2RPS:
            REVS_S = 0
        if REVS_P < 100 * RPM2RPS:
            REVS_P = 0
        self._ship.setAzimuthPropellerRevolutionsPerSec(REVS_S, 12801)
        self._ship.setAzimuthPropellerRevolutionsPerSec(REVS_P, 12802)

        # Set Azimuth Angle [rad]
        angleS = np.arctan2(action[self.actiontags["sinStbdDir_S"]], action[self.actiontags["cosStbdDir_S"]])
        angleP = np.arctan2(action[self.actiontags["sinPortDir_S"]], action[self.actiontags["cosPortDir_S"]])
        self._ship.setAzimuthPropellerAngle(angleS, 12801)
        self._ship.setAzimuthPropellerAngle(angleP, 12802)
        # self._ship.setAzimuthPropellerAngle(np.radians(action[self.actiontags['StbdDir_C']]),12801)
        # self._ship.setAzimuthPropellerAngle(np.radians(action[self.actiontags['PortDir_C']]),12802)

        velocityTrueGBA = self._ship.getVelocityOverBottom()
        ubottom = velocityTrueGBA[0]
        yawVelocity = velocityTrueGBA[5]

        # set rudder
        self._ship.setRudderAngle(
            -np.radians(action[self.actiontags["CentRud_S"]]), 10101
        )  # it seems our sign convention is opposite

        # Set Propeller Revolution [RPS]
        self._ship.setPropellerRevolutionsPerSec(action[self.actiontags["CentRPM_S_abs"]] * RPM2RPS, 10101)

        # Set Propeller Pitch [-]
        self._ship.setPropellerPitch(action[self.actiontags["PD_S"]] * PERCENT2FRACTION, 10101)

        # Set Thruster Fraction [-]
        self._ship.setThrusterFractionCommand(action[self.actiontags["Bow1RPM_C"]] * PERCENT2FRACTION, 10401)
        self._ship.setThrusterFractionCommand(action[self.actiontags["Bow2RPM_C"]] * PERCENT2FRACTION, 10402)

        # set time and send command
        self._ship.executeCommand(time)

    def updateState(self):

        # self.state[self.stateix['xe']]=self._MSMIf.getTPOEA()[0]
        # self.state[self.stateix['ye']]=self._MSMIf.getTPOEA()[1]
        xe = self._MSMIf.getTPOEA()[0]
        ye = self._MSMIf.getTPOEA()[1]
        """
        self.state[self.stateix['dist0']],self.state[self.stateix['dir0']] = self.distDir(xe,ye,0.0,0.0)
        self.state[self.stateix['dist1']],self.state[self.stateix['dir1']] = self.distDir(xe,ye,self.G1[0],self.G1[1])
        self.state[self.stateix['dist2']],self.state[self.stateix['dir2']] = self.distDir(xe,ye,self.G2[0],self.G2[1])
        self.state[self.stateix['dist3']],self.state[self.stateix['dir3']] = self.distDir(xe,ye,self.G3[0],self.G3[1])
        self.state[self.stateix['Heading']] =  self._MSMIf.getTPOEA()[5]
        self.state[self.stateix['COG']]  = self._MSMIf.getCPOE()[5]
        self.state[self.stateix['SOG']]  = np.sqrt(self._MSMIf.getTUBGB()[0]**2 + self._MSMIf.getTUBGB()[1]**2)
        self.state[self.stateix['STW_UW']] = self._MSMIf.getUWGB()[0]
        self.state[self.stateix['STW_VW']] = self._MSMIf.getUWGB()[1]
        self.state[self.stateix['SOG_U'] ]= self._MSMIf.getTUBGB()[0]
        self.state[self.stateix['SOG_V'] ]= self._MSMIf.getTUBGB()[1]
        self.state[self.stateix['Turnrate']] = self._MSMIf.getTUBGB()[5]
        """
        # ['distLead','dist0','Heading','COG,SOG','STW_UW','STW_VW','SOG_U','SOG_V','Turnrate','depth']
        self.state[self.stateix["distLead"]] = self.distLeadingLine((xe, ye), self.origo, self.endP)
        self.state[self.stateix["dist0"]], _ = self.distDir(xe, ye, 0.0, 0.0)
        self.state[self.stateix["Heading"]] = self._MSMIf.getTPOEA()[5]
        self.state[self.stateix["COG"]] = self._MSMIf.getCPOE()[5]
        self.state[self.stateix["SOG"]] = np.sqrt(self._MSMIf.getTUBGB()[0] ** 2 + self._MSMIf.getTUBGB()[1] ** 2)
        self.state[self.stateix["STW_UW"]] = self._MSMIf.getUWGB()[0]
        self.state[self.stateix["STW_VW"]] = self._MSMIf.getUWGB()[1]
        self.state[self.stateix["SOG_U"]] = self._MSMIf.getTUBGB()[0]
        self.state[self.stateix["SOG_V"]] = self._MSMIf.getTUBGB()[1]
        self.state[self.stateix["Turnrate"]] = self._MSMIf.getTUBGB()[5]
        ##self.state[self.stateix['depth']] = 1000

        depth = 0
        for ix in range(4):
            depth += self._MSMIf.getWDPS()[ix] / 4

    #        self.state[self.stateix['depth']] = depth

    def getState(self):
        return self.state

    def postStep(self, driverApp, time):
        """"""
        # log current solution. (byt skipping initial condition at time = 0.0)
        if time > 0.0:
            self.log(time)
            self.updateState()

    def isFinished(self, driverApp, time):
        """"""
        """
        dist2,dir2 =self.distDir(self._MSMIf.getTPOEA()[0],self._MSMIf.getTPOEA()[1],self.G2[0],self.G2[1])
        if dir2 < 90:
            self.reward += 1000 #-self._MSMIf.FUEUS[0]
            return True
        """
        # for now leaving Gedser scenario
        if np.degrees(self._MSMIf.getTPOEA()[5]) < 175.0 and time > 700:
            self.reward += 5000
            return True
        if time > 1000:
            self.reward -= 1000  # punish hard if too slow
            return True
        self.reward += -1  # self._MSMIf.FUEUS[0]
        return False
