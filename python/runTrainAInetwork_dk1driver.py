from pydriver.dk1driver.scenarios.TrainAInetwork import TrainAInetwork
import sys, os
import math
import logging
import jsons
import pathlib
from io import StringIO
from pydriver.dk1driver.IRL import GAIL
import numpy as np
import torch

# add parent path to sys path
import pathlib

sys.path.append(os.path.abspath(os.path.join(pathlib.Path(__file__).parent.absolute(), "..")))

# get our dk1 launcher
from pydriver.dk1driver.ScenarioRunner import (
    ScenarioRunner,
    openAndValidateRunFile,
    RunConfigurationReader,
)
from pydriver.dk1driver.Environment import Environment, WaveTypes


# turn on logging of errors
logging.basicConfig(format="%(asctime)s:%(threadName)s:%(levelname)s:%(message)s", level=logging.DEBUG)
logging.logThreads = 1


shallowWater = Environment()
shallowWater.depthDraughRatio = 500.0

deepWater = Environment()
deepWater.depthDraughRatio = math.inf

currentPath = pathlib.Path(__file__).parent.absolute()

whatToRun = 1
configFilePath = ""
outputDir = ""

random_seed = 0
max_timesteps = 1400  # max time steps in one episode
n_eval_episodes = 20  # evaluate average reward over n episrodes
lr = 0.0002  # learing rate
betas = (0.5, 0.999)  # betas for adam optimizer
n_epochs = 4  # 00  # number of epochs
n_iter = 100  # updates per epoch
batch_size = 100  # num of transitions sampled from expert
env_name = "Gedser"
directory = "z:/Tasks/121/121-20658/04 Technical/TrainedNetwork/{}".format(env_name)  # save trained models
filename = "GAIL_{}_{}".format(env_name, random_seed)

state_dim = 17
action_dim = 9

max_action = np.array([100, 150, 35, 150, 180, 150, 180, 100, 100])

max_state = [
    299.37420654296875,
    180.0,
    193.76779174804688,
    144.89976501464844,
    315.1005554199219,
    150.5034637451172,
    307.4475402832031,
    179.985595703125,
    347.6549987792969,
    165.76666259765625,
    7.150000095367432,
    0.30000001192092896,
    2.700000047683716,
    0.699999988079071,
    1.149999976158142,
    13.489999771118164,
    28.299999237060547,
]

# pol='GAIL'
pol = "BTJnn"

if "GAIL" in pol:
    policy = GAIL.GAIL(env_name, state_dim, action_dim, lr, betas, max_state, max_action, loadActor=True)
else:
    policy = GAIL.Actor(state_dim, action_dim)
    policy.load_state_dict(torch.load(r"Z:\Tasks\121\121-20658\04 Technical\btjNotes\btjnn.pth"))

# graph logging variables:
epochs = []
rewards = []


if whatToRun == 1:
    configFilePath = os.path.join(currentPath, "runs", r"TrainAInetworkScenario.json")
    outputDir = r"C:\temp\RK2021\Train_AI"


# modelIdList = [3708]
modelIdList = [3789]  # [3292]   #[3501]
for modelId in modelIdList:

    # serialize Environment objects to json objects
    shallowWaterConfig = jsons.dump(shallowWater)
    deepWaterConfig = jsons.dump(deepWater)

    variableDict = {
        "shipId": int(modelId),
        "shallowWater": shallowWaterConfig,
        "deepWater": deepWaterConfig,
        "outputDir": outputDir,
    }

    # open and parse a configuration file
    configReader = RunConfigurationReader(variableDict)
    configuration = configReader.readAndParse(configFilePath)

    # create a launcher
    launcher = ScenarioRunner(configuration, buildDocumentation=False)

    # launch dk1
    launcher.launch()

    # connect dk1driver to dk1
    launcher.connect()

    scenario = TrainAInetwork()

    scenario.setPolicy(policy)

    # launcher.driverApp.sharedMemoryInterface.save("c:/temp/sharedMemory_3501.bin")
    # training procedure
    for epoch in range(1, n_epochs + 1):
        # update policy n_iter times
        if pol == "GAIL":
            policy.update(n_iter, batch_size)
        # run the scenarios
        # evaluate in environment
        total_reward = 0
        for episode in range(n_eval_episodes):
            launcher.runScenario(scenario, 5000)
            # state = env.reset()
            # for t in range(max_timesteps):
            #    action = policy.select_action(state)
            #    state, reward, done, _ = env.step(action)
            #    total_reward += reward
            #    if done:
            #        break
        total_reward = scenario.getReward()
        avg_reward = int(total_reward / n_eval_episodes)
        print("Epoch: {}\tAvg Reward: {}".format(epoch, avg_reward))

        # add data for graph
        epochs.append(epoch)
        rewards.append(avg_reward)

        if pol == "GAIL":  # avg_reward > solved_reward:
            print("########### Solved! ###########")
            policy.save(directory, filename)
        # break


print("done")
launcher.disconnect()
