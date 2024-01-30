# Basic RL from scratch
 
Goal : code a golf game using reinforcement learning, without using any ML library and game environment.
We restrict ourselves to the case of a grid representing a golf course of size n × n. Each game ends when 
the golf ball is on the flag square, or after T = 3n time steps.
 
The state of the environment is defined by the position (random at time t=0) of the golf ball 
on the grid, and by the position (also random at time t=0) of the flag. The golfer can tap 
the ball and thus move it in one of the eight neighbouring squares.

If the ball falls out of the grid at time t, it will take the previous place at time t+1, the golfer has wasting time looking for it. The reward is zero at each time step, and 1 only the ball falls into the 
the flag hole before the game's end.