import math

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import sin, cos, pi
import pygame

M=5
m=1
L=1
a=0.25
Iv=0.01
Ip=(M*L**2)/12
F0=200
g=9.82
x=0.3*L
#I=I2+M*((L/2-a)**2)+I1+m*(y[0]**2)

xpos = []
ypos = []

y0 = [x,pi/32,0 ,0]

tend = 10
frames = 4000

def animate_events(i,xpos,ypos):
    redDot.set_data(xpos[i],ypos[i])
    return redDot,


def f(t, y):
    return [y[2],
            y[3],
            y[2]*y[3]^2 -(M*g/m)*sin(y[1]),
            (F*a -M*g*cos(y[1])*(L/2 -a) + (m*y[0]*(y[2]*y[3]^2 - (M*g/m)*sin(y[1]))*cos(y[1]))/sin(y[1]) + m*y[0]*y[2]*y[3]^2 *(-2 -cos(y[1])/sin(y[1])))/(Ip + Iv + m*y[0]^2)
            ]
    #(abs(cos(y[1])) < 0.01)

    """
    if (abs(y[0]) < 0.001):
        return [y[2],
                y[3],
                y[2]*y[3]^2 -(M*g/m)*sin(y[1]),
                (F*a -M*g*cos(y[1])*(L/2 -a) + (m*y[0]*(y[2]*y[3]^2 - (M*g/m)*sin(y[1]))*cos(y[1]))/sin(y[1]) + m*y[0]*y[2]*y[3]^2 *(-2 -cos(y[1])/sin(y[1])))/(Ip + Iv + m*y[0]^2)
                ]
    elif (abs(cos(y[1])) < 0.001):
        return [0,0,0,0]
    else:
        return [y[2],
                y[3],
                (sin(y[1])/cos(y[1]))*((-(F0*a -M*g*(L/2 -a)*cos(y[1]))/y[0])/m +2*y[2]*y[3] +y[0]*((F0*a-M*g*(L/2-a)*cos(y[1])-m*g*y[0]*cos(y[1]))/(I2+ I1) +y[2]*(y[3])**2)),
                 (F0*a -M*g*(L/2-a)*cos(y[1]) -m*g*y[0]*cos(y[1]))/(I2 +I1)]
    """

        #[y[2],
               # y[3],
                #(sin(y[1])/cos(y[1]))*((-(F0*a -M*g*(L/2 -a)*cos(y[1]))/y[0])/m +2*y[2]*y[3] +y[0]*((F0*a-M*g*(L/2-a)*cos(y[1])-m*g*y[0]*cos(y[1]))/(I2+ I1) +y[2]*(y[3])**2),
                #(F0*a -M*g*(L/2-a)*cos(y[1]) -m*g*y[0]*cos(y[1]))/(I2 +I1)]

        #[y[2],
                #y[3],
               # ((-sin(y[1]) * (F0 * a - M * g * (L / 2 - a) * cos(y[1])) / x) / m
               #  + 2 * y[2] * y[3] * sin(y[1]) +
                # (y[0] * sin(y[1]) * (F0 * a - M * g * cos(y[1]) * (L / 2 - a) - m * g * cos(y[1]) * y[0]) / (
                 #            I2 + M * ((L / 2 - a) ** 2) + I1 + m * (y[0] ** 2)))
               #  + y[0] * (y[3] ** 2) * cos(y[1])) / cos(y[1]),(F0*a-(M*g*cos(y[1])*(L/2-a))-m*g*cos(y[1])*y[0])/(I2+M*((L/2-a)**2)+I1+m*(y[0]**


sol = solve_ivp(f, [0, tend], y0, method='Radau', t_eval=np.linspace(0, tend, 4000))

i=0
for i in range(400):
#while abs(sol.y[0][i])<L:
    radius = (sol.y[0][i])
    angle = (sol.y[1][i])

    y = radius*sin(angle)

    x = radius*cos(angle)
    xpos.append(x)
    ypos.append(y)
    

    #i=i+1
    

fig, ax = plt.subplots(figsize=(5,5))
plt.plot(xpos, ypos)
plt.xlabel('x')
plt.ylabel('y')


redDot, = plt.plot(xpos[0], ypos[0], 'ro')
anim = animation.FuncAnimation(fig, animate_events, frames=i, fargs=(xpos, ypos),
                               interval=10, blit=True, repeat= True)
plt.show()


pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([800, 800])

t = 0
clock = pygame.time.Clock()
# Run until the user asks to quit
running = True
while running:
    clock.tick(30)

    
    def converter(xcoord, ycoord):
        var1,var2 =(400 + xcoord*400), (400 - ycoord*400)
        return(var1,var2)

    x_coord,y_coord = converter(xpos[t],ypos[t])
    
    vinkel = math.atan((ypos[t]*400 -10)/(xpos[t]*400))


    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen, (0, 0, 255), (x_coord,y_coord), 10)

    pygame.draw.circle(screen, (0, 0, 0), (400,400), 2)

    pygame.draw.line(screen, (0,0,0), (400,400),(converter(cos(vinkel),sin(vinkel))))

    pygame.draw.line(screen, (0,0,0), (400,400),(converter(-cos(vinkel)*0.25,-sin(vinkel)*0.25)))

    # Flip the display
    pygame.display.flip()

    t= t+1
    if t>=i:
        t=0
# Done! Time to quit.
pygame.quit()