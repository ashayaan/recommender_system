# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-05 15:19:03
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-13 11:48:28


'''
parameters that are used in 
training the learning algorithm
'''

params = {
    'gamma'      : 0.99,
    'min_value'  : -10,
    'max_value'  : 10,
    'policy_step': 10,
    'soft_tau'   : 0.001,
    'policy_lr'  : 1e-5,
    'value_lr'   : 1e-5,
    'actor_weight_init': 5e-2,
    'critic_weight_init': 6e-1,
}