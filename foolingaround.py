import numpy as np 
import matplotlib.pyplot as plt

import matplotlib.animation as animation

#===============Constants================# 
J=10 #choosing this J results in about 30% probability of a flip for \sum_{<0i,j>}=-1 for 100 degrees kelvin
k_B=1
T=1
dimen=100
np.random.seed(1)
data = np.random.randint(0,2,size=(dimen,dimen))
data[data==0]=-1





def calculateEnergy(data, i):
    """
    Takes the input of a square matrix and coordinates in the format [y,x] in relation to the visualisation.
    Gives back the energy change caused by flipping the selected spin.
    """
    [top, bottom, left, right] = [False]*4
    if i[1]!=0:
        left = data[i[0],i[1]-1]
    if i[1]!=dimen - 1:
        right = data[i[0],i[1]+1]
    if i[0]!=0:
        top = data[i[0]-1,i[1]]
    if i[0]!=dimen - 1:
        bottom = data[i[0]+1,i[1]]
    neighbour_sum = sum([top, bottom, left, right])
    return(-1 * J * data[i[0], i[1]] * neighbour_sum)

def flip(data):
    picker=np.random.randint(0,dimen,size=2)
    energy=-2*calculateEnergy(data, picker)
    if energy <= 0 or np.random.rand() < np.exp(-energy / (k_B * T)):
        data[picker[0],picker[1]] *= -1
    return(data)

fig, ax = plt.subplots()
image = ax.imshow(data, cmap=plt.cm.berlin, vmin=-1, vmax=1)
ax.set_xticks([])
ax.set_yticks([])


def update(i):
    for _ in range(50):
        flip(data)
    image.set_data(data)
    ax.set_title(f'Spin configuration')
    return (image,)


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=10)
plt.show()