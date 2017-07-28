
# coding: utf-8

# # Vectors

# ### Adding

# In[1]:

def vector_add(vec1, vec2):
    return [v1i+ v2i
           for v1i, v2i in zip(vec1, vec2)]


# In[2]:

v1 = range(10,20)
v1


# In[3]:

v2 = range(60,70)
v2


# In[4]:

v_add = vector_add(v1,v2)
v_add


# ### Subtracting

# In[5]:

def vector_subtract(vec1, vec2):
    return [v1i -  v2i
           for v1i, v2i in zip(vec1, vec2)]


# In[6]:

v_sub = vector_subtract(v2, v1)
v_sub


# In[7]:

v_sub = vector_subtract(v1, v2)
v_sub


# ### Sum list of vectors

# In[8]:

from functools import reduce

def vector_sum(vectors):
    return reduce(vector_add, vectors)


# In[9]:

from numpy import random as rd
n = rd.randint(1000)
v3 = range(n,(n+10))
vectors = [v1, v2,v3]
vectors


# In[10]:

v_sum = vector_sum(vectors)
v_sum


# ### Multiplication Scalar

# In[11]:

def scalar_mult(scalar, vector):
    return [ scalar * vi for vi in vector]


# In[12]:

for i,vec in enumerate(vectors):
    sm = scalar_mult(i, vec)
    print('scalar: ', i, ' mult: ', sm)


# ### Dot Product /Produto Scalar
# O produto escalar de dois vetores eh a soma de seus produtos item a item. O produto escalar mede a distancia a qual o vetor vec1 se estende na direcao de vec2, ou seja, eh o tamanho do vetor projetado no plano cartesiano

# In[13]:

def dot_product(vec1, vec2):
    return sum(v1i * v2i
              for v1i , v2i in zip(vec1, vec2))


# In[14]:

dot_vec = dot_product(v1, v2)
dot_vec


# ### Sum of Squares

# In[15]:

def sum_of_squares(vec):
    return dot_product(vec, vec)


# In[16]:

for v in vectors:
    sos = sum_of_squares(v)
    print(sos)


# ### Magnitude (Size)

# In[17]:

import math

def magnitude(vec):
    return math.sqrt(sum_of_squares(vec))


# In[18]:

for v in vectors:
    m = magnitude(v)
    print('Magnitude of: ', v , ' is: ', m)


# ### Distancia entre dois vetores

# In[19]:

def distance(vec1, vec2):
    return magnitude(vector_subtract(vec1, vec2))


# In[20]:

for v1 in vectors:
    for v2 in vectors:
        d = distance(v1, v2)
        print('Distance between: ', v1, ' and ', v2, ' is: ', d)


# ## Parallel Orthogonol

# In[21]:

def is_orthogonol(vec1, vec2, tolerance=1e-10):
    return abs(dot_product(vec1, vec2)) < tolerance;


# In[22]:

zeros = [0,0,0,0,0]
ones = [1,1,1,1,1]
seq = [1,2,3,4,5]


# In[23]:

is_orthogonol(zeros,ones)


# In[24]:

is_orthogonol(ones, seq)


# In[25]:

def is_zero(vec, tolerance=1e-10):
    return magnitude(vec) < tolerance


# In[26]:

from math import pi

def angle(vec1, vec2, in_degrees = False):
    return 0;


# In[27]:

def is_parallel(vec1, vec2):
    return (is_zero(vec1) or
           is_zero(vec2) or
           angle(vec1,vec2) == 0 or
           angle(vec1,vec2) == pi)


# In[28]:

is_parallel(seq,ones)


# In[ ]:



