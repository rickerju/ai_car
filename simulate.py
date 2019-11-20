from datetime import datetime
import gym
import neat

env = gym.make('CarRacing-v0')

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
            p[0] == p[1] == p[2] == 177 or\
            p[0] == p[1] == p[2] == 190 or\
            p[0] == p[1] == p[2] == 57 or \
            p[0] == p[1] == p[2] == 244 or \
            p[0] == p[1] == p[2] and 102:
        return 1

    else:
        raise EnvironmentError(str(p) + " not found")

def convert_observation_to_inputs(observation):
    newInputs = list()
    for x in range(95):
        for y in range(95):
            newInputs.append(convert_pixel_to_input(observation[x, y]))
    return newInputs

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        observation = env.reset()
        frames = 0
        totalReward = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for index in range(1000):
            env.render()
            frames += 1
            inputs = convert_observation_to_inputs(observation)
            inputs.append(1)
            observation, reward, done, info = env.step(net.activate(inputs))

            if(index % 50 == 0):
                print(str(datetime.now()) + " frame " + str(index) + "\ncurrent reward=" + str(totalReward))

            totalReward += reward

            if done:
                break
        env.close()
        reward = 1000 - .1 * frames
        genome.reward = reward

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.
# print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
# print('\nOutput:')
# winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
# for xi, xo in zip(xor_inputs, xor_outputs):
#     output = winner_net.activate(xi)
#     print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))