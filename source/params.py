# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-05 15:19:03
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-05 15:44:12

import torch

gamma = 0.99
min_value = -10
max_value = 10
soft_tau = 1e-3
policy_lr = 1e-5 
value_lr = 1e-5 
actor_init_weight = 5e-2
critic_init_weight = 6e-1 
