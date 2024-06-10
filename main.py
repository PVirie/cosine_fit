import numpy as np

def generate_data(b, n = 100):
    T = np.random.rand(b, n)*10*np.pi
    
    # random amplitude
    a = np.random.rand(b, 1)
    # random frequench
    f = np.random.rand(b, 1)
    # random start phase
    phi = np.random.rand(b, 1)
    
    # Y has shape (b, n)
    Y = a*np.cos(2*np.pi*f*T + phi)

    return T, Y, (a, f, phi)


def solve_A_and_Phase(Y, T, F):
    # find amplitude 
    C = np.cos(2*np.pi*F*T)
    S = np.sin(2*np.pi*F*T)
    M = np.stack((C, -S), axis = 2)
    # M has shape (b, n, 2), X has shape (b, 2, 1), then Y = M@X
    # solve with pseudo inverse
    Y = Y[:, :, np.newaxis]
    X = np.linalg.pinv(M)@Y
    # the amplitude is then sqrt(X[0]**2 + X[1]**2)
    A = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

    # find phase
    # the phase is np.arctan(X[1]/X[0])
    phi = np.arctan(X[:, 1]/X[:, 0])

    return A, phi


def get_F(T, Y):
    # computing various manual fourier response
    responses = []
    for i in np.arange(0, 1, 0.01):
        responses.append(np.sum(Y*np.cos(2*np.pi*i*T), axis = 1))

    responses = np.array(responses)
    # find the max response
    F = np.argmax(responses, axis = 0) * 0.01
    # return F of dim (b, 1)
    return F[:, np.newaxis]


def extract_param(T, Y):
    # T has shape (b, n)
    # Y has shape (b, n)
    # Y = Acos(2*pi*F*T + phi)
    # Acos(2*pi*F*T)cos(phi) - Asin(2*pi*F*T)sin(phi) = Y

    F_ = get_F(T, Y)
    A, phi = solve_A_and_Phase(Y, T, F_)

    # now fine tune with gradient descend
    for i in range(100):
        grad = (Y - A*np.cos(2*np.pi*F_*T + phi))*A*np.sin(2*np.pi*F_*T)*T*2*np.pi
        G = np.sum(grad, axis = 1, keepdims = True)
        F_ = F_ - 0.00001*G
        A, phi = solve_A_and_Phase(Y, T, F_)

    return F_, A



if __name__ == "__main__":
    # numpy 2 decimal print
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    T, Y, (a, f, phi) = generate_data(10)

    F, A = extract_param(T, Y)

    # print zip F and f
    print("===== F delta ======")
    print(np.concatenate((F, f), axis = 1))

    # print zip A and A
    print("===== A delta ======")
    print(np.concatenate((A, a), axis = 1))