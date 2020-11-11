import math
import scipy.io as sio
import numpy as np

def lax_wendroff(dx, dy, dt, g, u, v, h, u_tendency, v_tendency):
    uh = np.multiply(u, h)
    vh = np.multiply(v, h)
    NumR, NumC = h.shape
    Numr, Numc = uh.shape
    h_mid_xt = 0.5*(h[1:NumC-1][:]+h[0:NumC-2][:]) - (0.5*dt/dx)*(\
                   uh[1:u-1][:]-uh[0:u-2][:])
    h_mid_yt = 0.5*(h[:][1:NumR-1]+h[:][0:NumR-2]) - (0.5*dt/dy)*(\
                   vh[:][1:h-1]-vh[:][0:h-2])
    
    Ux = np.multiply(uh, u)+0.5*g*np.multiply(h, h)
    Uy = np.multiply(uh, v)
    uh_mid_xt = 0.5*(uh[1:Numc-1][:] + uh[0:Numc-2][:]) - (0.5*dt/dx) \
                *(Ux[1:Numc-1][:] - Ux[0:Numc-2][:])
    uh_mid_yt = 0.5*(uh[:][1:Numr-1] + uh[:][0:Numr-2]) - (0.5*dt/dy) \
                *(Uy[:][1:Numr-1] - Uy[:][0:Numr-2])
    
    Vx = Uy
    Vy =np.multiply(vh, v) + 0.5*g*h**2
    vh_mid_xt = 0.5*(vh[1:Numc-1][:] + vh[0:Numc-2][:]) - (0.5*dt/dx) \
                *(Vx[1:Numc-1][:] - Vx[0:Numc-2][:])
    vh_mid_yt = 0.5*(vh[:][1:Numr-1] + vh[:][0:Numr-2]) - (0.5*dt/dy) \
                *(Vy[:][1:Numr-1] - Vy[:][0:Numr-2])
    
    
    h_new = h[1:NumC-2][1:NumR-2]  \
            - (dt/dx)*(uh_mid_xt[1:Numc-1][1:Numr-2]- uh_mid_xt[0:Numc-2][1:Numr-2]) \
            - (dt/dy)*(vh_mid_xt[1:Numc-2][1:Numr-1] - vh_mid_xt[1:Numc-2][0:Numr-2])
    
    Ux_mid_xt = np.multiply(uh_mid_xt, uh_mid_xt)/h_mid_xt + 0.5*g*h_mid_xt**2       
    Uy_mid_yt = np.multiply(uh_mid_yt, vh_mid_yt)/h_mid_yt
    uh_new = uh[1:Numc-2][1:Numr-2] \
             - (dt/dx)*(Ux_mid_xt[1:Numc-1][1:Numr-2] - Ux_mid_xt[0:Numc-2][1:Numr-2]) \
             - (dt/dy)*(Uy_mid_yt[1:Numc-2][1:Numr-1] - Uy_mid_yt[1:Numc-2][0:Numr-2]) \
             + np.multiply((dt*u_tendency), 0.5*(h[1:NumC-2][1:NumR-2] + h_new)) 
             
    Vx_mid_xt = np.multiply(uh_mid_xt, vh_mid_xt)/h_mid_xt       
    Vy_mid_yt = np.multiply(vh_mid_yt, vh_mid_yt)/h_mid_yt + 0.5*g*h_mid_yt**2
    vh_new = vh[1:Numc-2][1:Numr-2] \
             - (dt/dx)*(Vx_mid_xt[1:Numc-1][1:Numr-2] - Vx_mid_xt[0:Numc-2][1:Numr-2]) \
             - (dt/dy)*(Vy_mid_yt[1:Numc-2][1:Numr-1] - Vy_mid_yt[1:Numc-2][0:Numr-2]) \
             + np.multiply((dt*v_tendency), 0.5*(h[1:NumC-2][1:NumR-2] + h_new))
    u_new = uh_new/h_new
    v_new = vh_new/h_new
    
    return u_new, v_new, h_new
#Possible initial conditions of the height field
UNIFORM_WESTERLY = 1
ZONAL_jet = 2
REANALYSIS = 3
GAUSSIAN_BLOB = 4
STEP = 5
CYCLONE_IN_WESTERLY = 6
SHARP_SHEAR = 7
EQUTAORIAL_EASTAERLY = 8
SINUSOIDAL = 9

#Possible orographies
FLOAT = 0
SLOPE = 1
GAUSSIAN_MOUNTAIN = 2
EARTH_OROGRAPHY = 3
SEA_MOUNT = 4

#Section 1: Configuration
g = 9.81
f = 1e-4
beta = 1.6e-11

dt_mins = 1
output_interval_mins = 60
forecast_length_days = 4

orography = 'GAUSSIAN_MOUNTAIN'
initial_conditions = 'UNIFORM_WESTERLY'
initially_geostrophic = True
add_random_height_noise = False

nx = 254
ny = 50

dx = 100.0e3
dy = dx

#Specify the range of heights to plot in metres
plot_height_range = [9500, 10500]

#SECTION 2: Act on the configuration information
dt = dt_mins*60.0
output_interval = output_interval_mins*60.0
forecast_length = forecast_length_days*24.0*3600.0
nt = np.fix(forecast_length/dt)+1
timesteps_between_outputs = np.fix(output_interval/dt)
noutput = np.ceil(nt/timesteps_between_outputs)

x = np.multiply(np.array(list(range(nx))), dx)
y = np.multiply(np.array(list(range(ny))), dy)
X, Y = np.meshgrid(x, y)

#Create the orography field "H"
if orography == 'FLAT':
    
    H = np.zeros((nx, ny), dtype = np.float)
    
elif orography == 'SLOPE':
    
    H = 9000*2*abs((np.mean(x)-X)/max(x))
    
elif orography == 'GAUSSIAN_MOUNTAIN':
    
    std_mountain_x = 5*dx
    std_mountain_y = 5*dy
    H = 4000*math.exp(-0.5*((X-np.mean(x))/std_mountain_x)**2\
                      -0.5*((Y-np.mean(y))/std_mountain_y)**2)
    
elif orography == 'SEA_MOUNT':
    
    std_mountain = 40*dy
    H = 9250*math.exp(-((X-np.mean(x))**2+(Y-0.5*np.mean(y))**\
                        2)/(2*std_mountain**2))
    
elif orography == 'EARTH_OROGRAPHY':
    
    data = sio.loadmat('digital_elevation_map.mat')
    H = data['elevation']
    NumR, NumC = H.shape
    H[0][:] = H[NumR-2][:]
    H[NumR-1][:] = H[1][:]
    
else:
    
    print("Don't know what to do with orography = ",str(orography))


if initial_conditions == 'UNIFORM_WESTERLY':
    
    mean_wind_speed = 20
    height = 10000-(mean_wind_speed*f/g)*(Y-np.mean(y))
#    height = 10000-np.multiply((mean_wind_speed*f/g), np.array(Y-np.mean(y)))
elif initial_conditions == 'SINUSOIDAL':
    
    height = 10000-np.multiply(350, math.cos \
             (np.multiply(np.multiply(Y/max(y), 4))), math.pi)
elif initial_conditions == 'EQUATORIAL_EASTERLY':
    height = 10000-50*math.cos((Y-np.mean(y))*4*math.pi/max(y))
    
elif initial_conditions == 'ZONAL_JET':
    
    height = 10000-math.tanh(20.0*((Y-np.mean(y))/max(y)))*400
    
elif initial_conditions == 'REANALYSIS':
    
    data = sio.loadmat('reanalysis.mat')
    height = 0.99*data['pressure']/g
    
elif initial_conditions == 'GAUSSIAN_BLOB':
    
    std_blob = 8.0*dy
    height = 9750+1000*math.exp(-((X-0.25*np.mean(x))**2 \
             +(Y-np.mean(y))**2)/(2*std_blob**2))
    
elif initial_conditions == 'STEP':
    
    height = 9750*np.ones((nx, ny), dtype = np.float)
    NumR, NumC = height.shape
    
    for i in range(NumC):
        for j in range(NumR):
            if X<(max(x)/5) and Y>(max(y)/10) and Y<(max(y)*0.9):
                height[i][j] = 10500
    
elif initial_conditions == 'CYCLONE_INWESTERLY':
    
    mean_wind_speed = 20
    sed_blob = 7.0*dy
    height = 10000-(mean_wind_speed*f/g)*(Y-np.mean(y))- \
             500*math.exp(-((X-0.5*np.mean(x))**2+(Y-np.mean(y))**2)\
                          /(2*std_blob**2))
    max_wind_speed = 20
    height = 10250-(max_wind_speed*f/g)*(Y-np.mean(y))**2/max(y) \
             -1000*math.exp(-(0.25*(X-1.5*np.mean(x))**2+ \
                              (Y-0.5*np.mean(y))**2)/(2*std_blob**2))
             
elif initial_conditions == 'SHARP_SHEAR':
    
    mean_wind_speed = 50
    height = (mean_wind_speed*f/g)*abs(Y-np.mean(y))
    NumR, NumC = height.shape
    height = 10000+height-np.mean(height.reshape((NumR*NumC), order='F'))
    
else:
    print("Don't know what to do with initial_conditions = ", \
          str(initial_conditions))

#Coriolis parameter as a matrix of values varying in y only
F = f+beta*(Y-np.mean(y))

#Initialize the wind to rest
u = np.zeros((nx, ny), dtype = np.float)
v = np.zeros((nx, ny), dtype = np.float)

if add_random_height_noise:
    
    NumpR, NumC = height.shape
    height = height+np.multiply(1.0*np.random.randn(NumC, NumR)*(dx/1.0e5), \
                                abs(F)/1e-4)
    
if initially_geostrophic:
    
    NumR, NumC = F.shape
    Numr, Numc = height.shape
    u[:][1:ny-2] = -0.5*g/(F[:][1:NumR-2]*dx)
    v[1:nx-2][:] = np.multiply((0.5*g/(F[1:NumC-2][:]*dx)), height[2:Numc-1] \
                     [:]-height[0:Numc-3][:])
    u[0][:] = u[1][:]
    u[nx-1][:] = u[nx-2][:]
    v[:][0] = 0
    v[:][ny-1] = 0
    
    max_wind = 200
    
    for i in range(nx):
        for j in range(ny):
            if u[i][j]>max_wind:
                u[i][j] = max_wind
            if u[i][j]<-max_wind:
                u[i][j] = -max_wind
            if v[i][j]>max_wind:
                v[i][j] = max_wind
            if v[i][j]<-max_wind:
                v[i][j] = -max_wind

h = height - H

u_save = np.zeros(nx, ny, noutput)
v_save = np.zeros(nx, ny, noutput)
h_save = np.zeros(nx, ny, noutput)
t_save = np.zeros(1, noutput)

i_save = 1
#SECTION 3: Main loop
for i in range(nt):
    if np.mod(i-1, timesteps_between_outputs)==0:
        max_u = math.sqrt(max(np.multiply(u.reshape((nx*ny), order='F'), \
                                          u.reshape((nx*ny), order='F'))+ \
        np.multiply(v.reshape(nx*ny, order='F'), v.reshape(nx*ny, order='F'))))
        time = i*dt/3600
        max_hour = forecast_length_days*24
        print('Time = '+str(i)+'hours (max'+str(max_hour)+');max(|u|) = '+\
              str(max_u))
        u_save[:][:][i_save-1] = u
        v_save[:][:][i_save-1] = v
        h_save[:][:][i_save-1] = h
        t_save[i_save-1] = i*dt
        i_save += 1
        
    NumR, NumC = F.shape
    Numr, Numc = H.shape
    u_accel = np.multiply(F[1:NumC-2][1:NumR-2], v[1:nx-2][1:ny-2]) - \
          (g/(2*dx))*(H[2:Numc-1][1:Numr-2]-H[0:Numc-3][1:Numr-2])
    v_accel = -np.multiply(F[1:NumC-2][1:NumR-2], u[1:nx-2][1:ny-2]) - \
          (g/(2*dy))*(H[1:Numc-2][2:Numr-1]-H[1:Numc-2][0:Numr-3])

    unew, vnew, h_new = lax_wendroff(dx, dy, dt, g, u, v, h, \
                                 u_accel, v_accel)
    
    NumR, NumC = unew.shap     
    u[1:nx-2][1:ny-2] = unew
    u[0][0] = unew[NumC-1][0]
    u[0][ny-1] = unew[NumC-1][NumR-1]
    u[nx-1][ny-1] = unew[0][NumR-1]
    u[nx-1][0] = unew[0][0]
    u[0][1:ny-2] = unew[NumC-1][:]
    u[nx-1][1:ny-2] = unew[0][:]
    u[1:nx-2][0]= unew[:][0] 
    u[1:nx-2][ny-1] = unew[:][NumR-1]
    
    NumR, NumC = vnew.shap     
    v[1:nx-2][1:ny-2] = vnew
    v[0][0] = vnew[NumC-1][0]
    v[0][ny-1] = vnew[NumC-1][NumR-1]
    v[nx-1][ny-1] = vnew[0][NumR-1]
    v[nx-1][0] = vnew[0][0]
    v[0][1:ny-2] = vnew[NumC-1][:]
    v[nx-1][1:ny-2] = vnew[0][:]
    v[1:nx-2][0]= vnew[:][0] 
    v[1:nx-2][ny-1] = vnew[:][NumR-1]
    
    v[:][0] = 0
    v[:][ny-1] = 0
    
    NumR, NumC = h_new.shape
    Numr, Numc = h.shape
    h[1:Numc-2][1:Numr-2] = h_new
    h[0][1:Numr-2] = h_new[Numc][:]
    h[Numc-1][1:Numr-2] = h_new[0][:]
       