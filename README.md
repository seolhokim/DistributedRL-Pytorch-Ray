# DistributedRL-Pytorch-Ray

## Algorithm
* A3C
* DPPO
* Ape-X
  * (Discrete version)
## Tested Environment
### Continuous
* MountainCarContinuous-v0
* Mujoco Benchmarks(Hopper,... etc)
### Discrete
* CartPole-v1
* LunarLander-v2
## TODO
### Fix
* ReplayBuffer copy computation
  * When remote function is executed, numpy array becomes immutable. so I used copy function to cover that.
### Add
* add impala
* add LASER
* add R2D2
* add NGU
* add Agent57
* test more environments
