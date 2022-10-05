import math

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import sin, cos, pi

M=5
m=1
L=1
a=0.25
I1=0.01
I2=(M*L**2)/12
F0=70
g=9.82
x=0.3*L
#I=I2+M*((L/2-a)**2)+I1+m*(y[0]**2)

xpos = []
ypos = []

y0 = [x,-pi/32,0 ,0]

tend = 10
frames = 4000

def animate_events(i,xpos,ypos):
    redDot.set_data(xpos[i],ypos[i])
    return redDot,


def f(t, y):
    #(abs(cos(y[1])) < 0.01)
    if (abs(y[0]) < 0.001):
        return [y[2],
                y[3],
                ((-sin(y[1]) * (F0 * a - M * g * (L / 2 - a) * cos(y[1]))) / (y[0]+0.002) / m
                 + 2 * y[2] * y[3] * sin(y[1]) +
                 (y[0] * sin(y[1]) * (F0 * a - M * g * cos(y[1]) * (L / 2 - a) - m * g * cos(y[1]) * y[0]) / (
                         I2 + M * ((L / 2 - a) ** 2) + I1 + m * (y[0] ** 2)))
                 + y[0] * (y[3] ** 2) * cos(y[1])) / cos(y[1]),(F0*a-(M*g*cos(y[1])*(L/2-a))-m*g*cos(y[1])*y[0])/(I2+M*((L/2-a)**2)+I1+m*(y[0]**2))]
    elif (abs(cos(y[1])) < 0.001):
        return [0,0,0,0]
    else:
        return [y[2],
                y[3],
                ((-sin(y[1]) * (F0 * a - M * g * (L / 2 - a) * cos(y[1]))) / y[0] / m
                 + 2 * y[2] * y[3] * sin(y[1]) +
                 (y[0] * sin(y[1]) * (F0 * a - M * g * cos(y[1]) * (L / 2 - a) - m * g * cos(y[1]) * y[0]) / (
                         I2 + M * ((L / 2 - a) ** 2) + I1 + m * (y[0] ** 2)))
                 + y[0] * (y[3] ** 2) * cos(y[1])) / cos(y[1]),(F0*a-(M*g*cos(y[1])*(L/2-a))-m*g*cos(y[1])*y[0])/(I2+M*((L/2-a)**2)+I1+m*(y[0]**2))]


        #[y[2],
               # y[3],
                #(sin(y[1])/cos(y[1]))*((-(F0*a -M*g*(L/2 -a)*cos(y[1]))/y[0])/m +2*y[2]*y[3] +y[0]*((F0*a-M*g*(L/2-a)*cos(y[1])-m*g*y[0]*cos(y[1]))/(I2 + M*(L/2-a)**2 +I1 +m*(y[0])**2)) +y[2]*(y[3])**2),
                #(F0*a -M*g*(L/2-a)*cos(y[1]) -m*g*y[0]*cos(y[1]))/(I2 + M*(L/2-a)**2 +I1 +m*(y[0])**2)]

        #[y[2],
                #y[3],
               # ((-sin(y[1]) * (F0 * a - M * g * (L / 2 - a) * cos(y[1])) / x) / m
               #  + 2 * y[2] * y[3] * sin(y[1]) +
                # (y[0] * sin(y[1]) * (F0 * a - M * g * cos(y[1]) * (L / 2 - a) - m * g * cos(y[1]) * y[0]) / (
                 #            I2 + M * ((L / 2 - a) ** 2) + I1 + m * (y[0] ** 2)))
               #  + y[0] * (y[3] ** 2) * cos(y[1])) / cos(y[1]),(F0*a-(M*g*cos(y[1])*(L/2-a))-m*g*cos(y[1])*y[0])/(I2+M*((L/2-a)**2)+I1+m*(y[0]**


sol = solve_ivp(f, [0, tend], y0, method='Radau', t_eval=np.linspace(0, tend, frames))


for i in range(400):
    radius = (sol.y[0][i])
    angle = (sol.y[1][i])

    y = radius*sin(angle)

    x = radius*cos(angle)
    xpos.append(x)
    ypos.append(y)
    #print((sol.y[0][i]))

fig, ax = plt.subplots(figsize=(5,5))
plt.plot(xpos, ypos)
plt.xlabel('x')
plt.ylabel('y')

redDot, = plt.plot(xpos[0], ypos[0], 'ro')
anim = animation.FuncAnimation(fig, animate_events, frames=400, fargs=(xpos, ypos),
                               interval=10, blit=True, repeat= True)
plt.show()


