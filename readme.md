## CIS 365 Project 3

## Group Members

* Justin Rickert
* Jose Garcia Reyes
* Rose Ault

## Necessary Packages

* gym
* neat
* sys

## Setup Commands

* `pip install gym`
* `pip install gym[box2d]`
* `pip install box2d`
* `pip install neat`

## Run Program

Program automatically saves the population to a new file every 10 generations. `config` file must be at root level

* To start a new population from scratch starting at 100 frames per generation and increment frames every 10 generations 
    * `python simulate.py`
* To load a population from file starting at 1000 frames per generation
    * `python simulate.py {relativeFilePath}`