import numpy as np
import matplotlib.pyplot as plt 


# -------------------------------------------------------------
# -------------------- TX Functions ---------------------------
# -------------------------------------------------------------

def source(N, p):
       return np.where(np.random.rand(1,N) > p, 1, 0)

def mapper(bits, const, mapping):
    # we suppose the input bit sequence is matched to the constellation size

    s = []

'''
    for i in range(0 
        match bits:
            case [0,0,0,0]:
                s.append(-3-3j)
            case [0,0,0,1]:
                s.append(-3-3j)
            case [0,0,1,0]:
                s.append(-3-3j)
            case [0,0,1,1]:
                s.append(-3-3j)
            case [0,1,0,0]:
                s.append(-3-3j)
            case [0,1,0,1]:
                s.append(-3-3j)
            case [0,1,1,0]:
                s.append(-3-3j)
            case [0,1,1,1]:
                s.append(-3-3j)
            case [1,0,0,0]:
                s.append(-3-1j)
            case [1,0,0,1]:
                s.append(-1-3j)
            case [1,0,1,0]:
                s.append(-3-3j)
            case [1,0,1,1]:
                s.append(-1-3j)
            case [1,1,0,0]:
                s.append(1-1j)
            case [1,1,0,1]:
                s.append(3-1j)
            case [1,1,1,0]:
                s.append(1-3j)
            case [1,1,1,1]:
                s.append(3-3j)

'''

def constellation(M, Es):
    dist_matrix_real = np.tile(np.arange(-(np.log2(M)-1), np.log2(M)-1, 2), (1, int(np.log2(M))))
    print(dist_matrix_real)
    dist_matrix_img = dist_matrix_real.transpose()*1j

    return (dist_matrix_real + dist_matrix_img).flatten()

def modul(t, s, B):
    Ns = len(s)
    
    N = len(t) # size of the simulation time

    qt = np.zeros(N)

    for i in range(int (-np.floor(Ns/2)), int (np.ceil(Ns/2))):
        print(i)
        qt = qt + s[i+(int (np.floor(Ns/2)))]*np.sinc(B*t-i)

    return qt

# a demodulator simply does a projection over the basis (inner prod -> integral of the product of those signals)
def demod(x, dt, B, Ns, t):
    
    s = []
    
    # check lower limit of range -> may be buggy
    for i in range(int (-np.floor(Ns/2)), int (np.ceil(Ns/2))):
        print(i)
        s.append(B*sum(x*np.sinc(B*t - i))*dt)

    return s

# -------------------------------------------------------------
# ---------------- Channel Functions --------------------------
# -------------------------------------------------------------

# takes q(t,z') and outputs q(t,z'+z)
def lin_step(qt, f, beta2, z):
    w = np.fft.fftshift(2*np.pi*f)
    return np.fft.ifft(np.fft.fft(qt)*np.exp(1j*beta2*(w**2)*z/2))

# takes q(t,z') and outputs q(t,z'+z)
def non_lin_step(qt, gamma, z):
    return qt*np.exp(1j*gamma*z*(np.abs(qt)**2))

# idea for the future: encapsulate channel in a class with all channel params
def channel(L, Nsteps, gamma, beta2, qt, t, f):
    dz = L/Nsteps

    for i in range(0, Nsteps):
        qt = lin_step(qt, f, beta2, dz)
        qt = non_lin_step(qt, gamma, dz)

    return qt

# -------------------------------------------------------------
# ------------------ Plot Functions ---------------------------
# -------------------------------------------------------------

# create a plot function that plots on time and frequency
def plot_sig(sig, t, f):
    plt.plot(t, np.abs(sig)**2)
    plt.show()
    plt.plot(f, np.abs(sig_spectrum(sig))**2)
    plt.show()

# -------------------------------------------------------------
# --------------- Miscellaneous Functions ---------------------
# -------------------------------------------------------------

# create a fft function that does fft + fftshift in one go
def sig_spectrum(sig):
    return np.fft.fftshift(np.fft.fft(sig))


# create function to compare whether symbols are correctly demod (sum the absolute squared error + remember to normalize by vector s norm)
def symbol_error(s, s_hat):
    return (2*(np.abs(s-s_hat)**2)/((np.abs(s)**2)+np.abs(s_hat)**2))


# -------------------------------------------------------------
# -------------------- Main Function --------------------------
# -------------------------------------------------------------
    
def main():

    # --------------- Simulation parameters -----------------

    T = 20
    N = 2**12
    dt = T/N
    t = np.linspace(-T/2, T/2, N)

    B = 3
    
    A = 2

    beta2 = -2
    gamma = 2
    L = 1
    Nsteps = 5000

    F = 1/dt
    df = 1/T
    f = np.arange(-F/2, F/2, df)

    # --------------- System Parameters ---------------------

    Ns = 2;

    s = np.array([1+1j, 2-0.5j])

    # --------------- Simulation ----------------------------

    #x = 1*np.exp(-(t**2)/2*0.1)

    # breather waves - A = 2.5/3.5

    x = A/np.cosh(A*t)

    x_out = channel(L, Nsteps, gamma, beta2, x, t, f)

    #x_l = lin_step(x, f, beta2, z)

    #x_nl = non_lin_step(x, gamma, z)
    
    plt.plot(t, np.abs(x))
    plt.plot(t, np.abs(x_out))
    plt.grid(True)
    plt.suptitle('Time domain signal')
    plt.show()

    plt.plot(f, np.abs(sig_spectrum(x)))
    plt.plot(f, np.abs(sig_spectrum(x_out)))
    plt.grid(True)
    plt.suptitle('Frequency domain signal')
    ax = plt.gca()
    ax.set_xlim([-4, 4])
    plt.show()

    #print("Before energy: {}".format(sum((np.abs(x)**2)*df)))
    #print("After energy: {}".format(sum((np.abs(x_nl)**2)*df)))

if __name__ == "__main__":
    main()  
