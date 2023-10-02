import math
import copy

import numpy as np
import pandas as pd
from pydriver.dk1driver.scenarios.ScenarioBase import ScenarioBase
from pydriver.dk1driver.dk1Commands import shipStateResetCommand, shipCommandCommand, shipExtraCommand
from pydriver.dk1driver.generated.devices import MSM
from pydriver.dk1driver.generated.devices import RudderPropeller, AzimuthPropeller

## Sensor signals from VDR
## SOG,
##
##
def filter_stw(df_, th_acc=0.5, th_len=50):

    df = df_.copy()

    # numpy-array to modify
    new_stw_u = df["STW_UW"].values.copy()
    new_stw_v = df["STW_VW"].values.copy()

    # Acceleration estimate, discrete derivative with period length of 20 (a(t) = v(t) - v(t-20))
    diff_index = df["SOG_U"].diff(20).fillna(th_acc + 0.1).abs().values > th_acc

    # Find periods with acceleration
    acc_phases = []

    up = True
    len_count = 0

    start_index = 0
    prev_val = diff_index[0]

    for i, val in enumerate(diff_index):
        if val == False and up == True:
            len_count += 1
        if len_count == th_len:
            acc_phases.append((start_index, i))
            up = False
            len_count = 0
        if val == True and prev_val == False and up == False:
            up = True
            start_index = i

        prev_val = val

    # For phases with acceleration, set STW = SOG (+ bias) for 0 current
    for idx in acc_phases:
        new_stw_u[idx[0] : idx[1]] = df["SOG_U"].iloc[idx[0] : idx[1]].values + 1.4
        new_stw_v[idx[0] : idx[1]] = df["SOG_V"].iloc[idx[0] : idx[1]].values

    df["STW_UW"] = new_stw_u
    df["STW_VW"] = new_stw_v

    stw_u_filt = df["STW_UW"].rolling(window=50, center=True).mean().values
    stw_v_filt = df["STW_VW"].rolling(window=50, center=True).mean().values

    stw_u_filt[:25] = df["SOG_U"].iloc[:25].values + 1.4
    stw_u_filt[-24:] = df["SOG_U"].iloc[-24:].values + 1.4

    stw_v_filt[:25] = df["SOG_V"].iloc[:25].values
    stw_v_filt[-24:] = df["SOG_V"].iloc[-24:].values

    stw_u_filt -= 1.4

    return stw_u_filt, stw_v_filt


def mapAngle2pi(ang):
    if ang < 0.0:
        ang += 2 * np.pi
    if ang > 2 * np.pi:
        ang -= 2 * np.pi
    return ang


class VDR(ScenarioBase):
    """implementation VDR recordings"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._handleCommandfileName = kwargs.get("VDRfile")
        self._initFilename = self._handleCommandfileName.replace("_cmnd", "_init")
        self._dfHandleCommands = pd.read_csv(self._handleCommandfileName, sep=",", header=0)
        df = self._dfHandleCommands  # just for ease of manimuplating as below

        # df['STW_UW_filt'], df['STW_VW_filt'] = filter_stw(df)

        # df['cur_u']=df['SOG_U'] - df['STW_UW_filt']    ## Mesuremnst are in knots
        # df['cur_v']=df['SOG_V'] - df['STW_VW_filt']
        # df['cur_speed_ms']=np.sqrt(df.cur_u**2 + df.cur_v**2)*.5144   ## convert to m/s
        # df['cudir_body']=np.arctan2(df.cur_v,df.cur_u)
        # df['cudir2pi'] =   df.apply(lambda row:mapAngle2pi(row.cudir_body),axis=1)
        # df['cur_dir'] =df.cudir2pi + np.radians(df.Heading)
        # df['cur_dir_cor'] = df.apply(lambda row:mapAngle2pi(row.cur_dir),axis=1)

        if df.isnull().values.any():
            df.fillna(method="ffill", inplace=True)
            if df.isnull().values.any():
                df.fillna(method="bfill", inplace=True)
        self._index = 0  ## keep track of active index

        self._P = -0.1  ## Proportional gain
        self._D = 20.0  ## Differential gain
        # self._dfHandleCommands.set_index(['time'],inplace=True)

        # static data

        pass

    def loadShip(self, driverApp):
        """"""
        super().loadShip(driverApp)

        self._ship = driverApp.getShip()
        self._MSMIf = self._MSMIf = MSM.MSM(driverApp.sharedMemoryInterface)
        self._rudderPropeller = RudderPropeller.RudderPropeller(driverApp.sharedMemoryInterface)
        self._CenterRatio = self._rudderPropeller.getGearRatio(10101)[0]
        self._aziPOD = AzimuthPropeller.AzimuthPropeller(driverApp.sharedMemoryInterface)
        self._gearRatio = self._aziPOD.getGearRatio(12801)[0]
        # set initial position
        init_state = {}
        with open(self._initFilename, "r") as f:
            for line in f:
                line = line.strip()
                (key, val) = line.split(",")
                init_state[key] = val
        linearPosition = np.array([float(init_state["x"]), float(init_state["y"]), 0.0])
        angularPosition = np.array([0.0, 0.0, float(init_state["psi"])])

        # set initial speed
        linearVelocity = np.array([float(init_state["u"]) * 0.5144, float(init_state["v"]), 0.0])
        angularVelocity = np.array([0.0, 0.0, float(init_state["r"])])

        self._ship.reset(linearPosition, angularPosition, linearVelocity, angularVelocity)

    def preStep(self, driverApp, time):
        """"""
        # log initial conditions
        if time == 0.0:
            self.log(time)

    def timeStep(self, driverApp, time):
        """"""
        if self._index < len(self._dfHandleCommands) - 1:
            if time > self._dfHandleCommands.iloc[self._index + 1].datetime:
                self._index += 1

        # no matter how we sail apply the same relative wind as measure by modifying the true wind with the ship speed
        wspeedRelative = self._dfHandleCommands.iloc[self._index].W_Speed_R_kn * 0.5144
        wdirRelative = np.radians(self._dfHandleCommands.iloc[self._index].W_Dir_R)
        uWindGBArelative = wspeedRelative * np.cos(wdirRelative + np.pi)
        vWindGBArelative = wspeedRelative * np.sin(wdirRelative + np.pi)

        uwTrueGBA = self._MSMIf.getUBGB()[0] - uWindGBArelative
        vwTrueGBA = self._MSMIf.getUBGB()[1] - vWindGBArelative
        windDirTrue = np.arctan2(vwTrueGBA, uwTrueGBA) + self._MSMIf.getTPOEA()[5]

        xe = self._MSMIf.getTPOEA()[0]
        ye = self._MSMIf.getTPOEA()[1]
        distFromOrigo = np.sqrt(xe ** 2 + ye ** 2)

        if windDirTrue < 0.0:
            windDirTrue += 2 * np.pi
        if windDirTrue > 2 * np.pi:
            windDirTrue -= 2 * np.pi

        self._environment.wind.speed = self._dfHandleCommands.iloc[
            self._index
        ].Speed_MS_T  # np.sqrt(uwTrueGBA**2 + vwTrueGBA**2)
        self._environment.wind.direction = np.radians(self._dfHandleCommands.iloc[self._index].W_Dir_T)  # windDirTrue

        self._environment.depth = self._dfHandleCommands.iloc[self._index].depth + self._MSMIf.getBDM()

        self._environment.current.speed = self._dfHandleCommands.iloc[self._index].cur_speed_ms
        self._environment.current.direction = self._dfHandleCommands.iloc[self._index].cur_dir_cor

        shipExtraCmd = shipExtraCommand()

        shipCommand = shipCommandCommand(self._environment)
        # dk1 need time in ms
        time_ms = time * 1000.0
        shipCommand.setTime(time_ms)

        commands = self._dfHandleCommands.iloc[self._index]

        # shipExtraCmd.setAzipodNotch(commands['punot_s'],0)
        # shipExtraCmd.setAzipodNotch(commands['punot_p'],1)
        aziPropellerRevolutionSTBD = self._aziPOD.getDSSP(12801)
        aziPropellerRevolutionSTBD[0] = commands["StbdRPM_S_abs"] * 2 * np.pi / (60.0 * self._gearRatio)

        aziPropellerRevolutionPORT = self._aziPOD.getDSSP(12802)
        aziPropellerRevolutionPORT[0] = commands["PortRPM_S_abs"] * 2 * np.pi / (60.0 * self._gearRatio)

        shipExtraCmd.setAzipodDirection(np.radians(commands["StbdDir_C"]), 0)
        shipExtraCmd.setAzipodDirection(np.radians(commands["PortDir_C"]), 1)

        # self._ship.setMachineryTelegraph(commands['hand'])

        ubottom = self._MSMIf.getUBGB()[0]
        if ubottom > 5.0:
            # self._ship.setAutoPilotMode(1)
            # self._ship.setAutoPilotCourse(np.radians(commands['Heading']))
            ## twin controller soft lock to the heading
            headingError = np.radians(commands["Heading"]) - self._MSMIf.getTPOEA()[5]
            if headingError > np.pi:
                headingError -= 2 * np.pi
            elif headingError < -np.pi:
                headingError += 2 * np.pi

            rudderCommand = self._P * headingError + self._D * self._MSMIf.getTUBGB()[5]
            self._ship.setAutoPilotMode(0)
            self._ship.setRudderAngle(rudderCommand, 0)
        else:
            self._ship.setAutoPilotMode(0)
            self._ship.setRudderAngle(
                -np.radians(commands["CentRud_S"]), 0
            )  ## it seems our sign convention is opposite

        propellerRevolution = self._rudderPropeller.getDSSP(10101)
        propellerRevolution[0] = commands["CentRPM_S_abs"] * 2 * np.pi / (60.0 * self._CenterRatio)  # /60.

        propellerPitch = self._rudderPropeller.getPDCMD(10101)
        propellerPitch[0] = commands["PD_S"] / 100.0

        self._ship.setThrusterFractionCommand(commands["Bow1RPM_C"], 10401)
        self._ship.setThrusterFractionCommand(commands["Bow2RPM_C"], 10402)

        # send command and wait for response
        driverApp.runCommand(shipExtraCmd, waitResponse=False)  ## send msg250 no response to wait for
        # driverApp.runCommand(shipCommand)   ## send msg200
        self._ship.executeCommand(time)
        # _ = shipCommand.getResponse()

    def postStep(self, driverApp, time):
        """"""
        # log current solution. (byt skipping initial condition at time = 0.0)
        if time > 0.0:
            self.log(time)

    def isFinished(self, driverApp, time):
        """"""
        # we just run all the time steps
        if self._index + 1 == len(self._dfHandleCommands):
            return True

        return False
