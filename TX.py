import numpy as np
import matplotlib.pyplot as plt 


def source(N, p):
       return np.where(np.random.rand(1,N) > p, 1, 0)

def mapper(bits, const, mapping):
    # we suppose the input bit sequence is matched to the constellation size
    match bits:
        case [0,0,0,0]:
            return -3-3j
        case [0,0,0,1]:
            return -3-3j
    
        case [0,0,1,0]:
            return -3-3j

        case [0,0,1,1]:
            return -3-3j

        case [0,1,0,0]:
            return -3-3j
            
        case [0,1,0,1]:
            return -3-3j

        case [0,1,1,0]:
            return -3-3j

        case [0,1,1,1]:
            return -3-3j

        case [1,0,0,0]:
            return -3-1j

        case [1,0,0,1]:
            return -1-3j
    
        case [1,0,1,0]:
            return -3-3j

        case [1,0,1,1]:
            return -1-3j

        case [1,1,0,0]:
            return 1-1j
            
        case [1,1,0,1]:
            return 3-1j

        case [1,1,1,0]:
            return 1-3j

        case [1,1,1,1]:
            return 3-3j

def constellation(M, Es):
    dist_matrix_real = np.tile(np.arange(-(np.log2(M)-1), np.log2(M)-1, 2), (1, int(np.log2(M))))
    print(dist_matrix_real)
    dist_matrix_img = dist_matrix_real.transpose()*1j

    return (dist_matrix_real + dist_matrix_img).flatten()

def modulator(t, symb, BW):
    
    

    return qt
    
def main():

    T = 100
    N = 2**11
    dt = T/N
    t = np.linspace(-T/2, T/2, N)

    F = 1/dt
    df = 1/T
    f = np.linspace(-F/2, F/2, N)

    sinc_t = np.sinc(t)

    sinc_f = np.fft.fftshift(np.fft.fft(np.sinc(t)))


    plt.plot(t, sinc_t)
    plt.show()
    plt.plot(f, np.abs(sinc_f)**2)
    plt.show()

if __name__ == "__main__":
    main()  
