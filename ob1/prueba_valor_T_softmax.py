#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:28:50 2020

@author: camilo
"""

import numpy as np
import matplotlib.pyplot as plt

x = 5
y = 5

T = np.linspace(0.01, 5, 100)
f = lambda x, y: np.exp(x) / (np.exp(x) + np.exp(y))

y = f(x/T, y/T)
plt.figure()
plt.plot(T, y)
plt.xlabel('T')
plt.ylabel('Valor')
