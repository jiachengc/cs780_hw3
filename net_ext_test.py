#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import string
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))
from net_ext import dot_product_vectors

a = np.array([ [ 1, 2 ] ])
b = np.array([ [ 1, 2, 3 ], [ 1, 2, 3 ] ])
c = dot_product_vectors(a,b)
print("c:", c)
print("type(c):", type(c))



