import gym
import neat
import sys
from neat import Checkpointer
from neat.reporting import BaseReporter


class generationData():
    generation = 0
    framesForGen = 50

    # advance metadata by one generation
    def nextGen(self):
        self.generation += 1

        if self.generation % 10 == 0 and self.framesForGen < 1000:
            self.framesForGen += 50

    # get generation number
    def getGen(self):
        return self.generation

    # set initial frames
    def setFrames(self, frames):
        self.framesForGen = frames

    # set framesForGen to full
    def finalize(self):
        self.framesForGen = 1000


env = gym.make('CarRacing-v0')
genData = generationData()


# class that logs information
class afterGenerationReporter(BaseReporter):

    def __init__(self, genData):
        self.generation = genData

    # after all the genomes in a generation have been evaluated
    def post_evaluate(self, config, population, species, best_genome):
        print("----- Post Evaluate -----")
        print("completed generation " + str(genData.generation))
        print("best genome fitness " + str(best_genome.fitness))

    # after the generation has completed
    def end_generation(self, config, population, species_set):
        # for genome in population.values():
        #     print(genome.fitness)

        self.generation.nextGen()


def convert_pixel_to_input(p):
    # black (0,0,0) red (255,0,0),(204,0,0) blue (51, 0, 255), (0,0,255)
    if (p[0] == 0 and p[1] == 0 and p[2] == 0) or \
            (p[0] == 255 and p[1] == p[2] == 0) or \
            (p[0] == 204 and p[1] == p[2] == 0) or \
            (p[0] == 51 and p[1] == 0 and p[2] == 255) or \
            (p[0] == 0 and p[1] == 0 and p[2] == 255):
        return 0

    # green (102,204,102), (102, 229, 102), (0, 255, 0)
    elif (p[0] == 102 and p[1] == 204 and p[2] == 102) or \
            (p[0] == 102 and p[1] == 229 and p[2] == 102) or \
            (p[0] == 0 and p[1] == 255 and p[2] == 0):
        return -1

    # grey (107,107,107), (105,105,105), (42,42,42), (31,31,31), (4,4,4), (173,173,173), (102,102,102)
    #      (177,177,177) (190,190,190), (57,57,57)
    elif (p[0] == p[1] == p[2] == 107) or \
            p[0] == p[1] == p[2] == 105 or \
            p[0] == p[1] == p[2] == 42 or \
            p[0] == p[1] == p[2] == 31 or \
            p[0] == p[1] == p[2] == 4 or \
            p[0] == p[1] == p[2] == 173 or \
            p[0] == p[1] == p[2] == 177 or \
            p[0] == p[1] == p[2] == 190 or \
            p[0] == p[1] == p[2] == 57 or \
            p[0] == p[1] == p[2] == 244 or \
            p[0] == p[1] == p[2] and 102:
        return 1

    else:
        raise EnvironmentError(str(p) + " not found")


# convert observation from gym to
def convert_observation_to_inputs(observation):
    newInputs = list()
    for x in range(95):
        for y in range(95):
            newInputs.append(convert_pixel_to_input(observation[x, y]))
    return newInputs


# # get mean of all elements in the 4 x 4 chunk of values
# def get_mean(list):
#     if len(list) != 16:
#         raise EnvironmentError("number of inputs in list is invalid")
#
#     total = 0
#     items = 0
#     for i in list:
#         items += 1
#         total += i
#     return total / items
#
#
# # converts observation from gym to 576 inputs for 4 x 4 chunks of pixels
# def convert_observation_to_inputs(observation):
#     finalInputs = list()
#     tempInputs = list()
#     for x in range(24):
#         for y in range(24):
#             for xr in range(4):
#                 for yr in range(4):
#                     tempInputs.append(convert_pixel_to_input(observation[4 * x + xr, 4 * y + yr]))
#             finalInputs.append(get_mean(tempInputs))
#             tempInputs = list()
#     return finalInputs


# alters outputs from neural network
def prepare_outputs(outputs):
    processedOutputs = list()

    processedOutputs.append(outputs[0] - .5)
    processedOutputs.append(outputs[1])
    processedOutputs.append(outputs[2])

    return processedOutputs


# function that is used to evaluate each individual genome in a generation
def eval_genomes(genomes, config):
    for _, genome in genomes:
        observation = env.reset()
        totalReward = 0  # cumulative reward given by gym
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # instantiate new NN
        for _ in range(genData.framesForGen):
            env.render()  # comment out for better performance but no visual
            inputs = convert_observation_to_inputs(observation)
            inputs.append(1)  # add bias node
            output = net.activate(inputs)
            observation, reward, done, info = env.step(prepare_outputs(output))  # advance to next frame

            totalReward += reward

            if done:
                break

        env.close()
        genome.fitness = totalReward


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# Create the population
if len(sys.argv) == 1:
    p = neat.Population(config)
else:
    p = Checkpointer.restore_checkpoint(sys.argv[1])
    genData.setFrames(200)  # initial number of frames when loading population
    print("loaded population from " + sys.argv[1])

# add checkpoints and logger
p.add_reporter(afterGenerationReporter(genData))
p.add_reporter(Checkpointer(10, None, "gen-"))  # name of file that will be generated with suffix of generation number

# run until fitness of 750 is met
winner = p.run(eval_genomes)
