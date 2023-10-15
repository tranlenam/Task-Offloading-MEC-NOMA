# A Max-Min Task Offloading Algorithm for Mobile Edge Computing Using Non-Orthogonal Multiple Access
# Authors: (*)Vaibhav Kumar, (^)Muhammad Fainan Hanif, ($)Markku Juntti, and (*)Le-Nam Tran
# DOI: 10.1109/TVT.2023.3263791
# Journal: IEEE Transactions on Vehicular Technology
# (*): School of Electrical and Electronic Engineering, University College Dublin, D04 V1W8 Dublin, Ireland
# (^): Institute of Electrical, Electronics and Computer Engineering, University of the Punjab, Lahore 54590, Pakistan
# ($): Centre for Wireless Communications, University of Oulu, 90014 Oulu, Finland
# email: vaibhav.kumar@ieee.org

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(0)


# function for dB to power conversion
def db2pow(x):
    return 10 ** (0.1 * x)


# function to generate channel gain
def chanGen(distance):
    distanceKm = distance * 1e-3  # distance in Km
    PL = db2pow(128.1 + 37.6 * np.log10(distanceKm))  # path loss
    y = np.sqrt(0.5 / PL) * (np.random.randn(1, ) + 1j * np.random.randn(1, ))[0]
    return abs(y) ** 2


# function to model the optimization problem
def model():
    # defining the constraints                
    constraints = []
    # ---------- constraint (2c)
    [constraints.append(cp.sum(Dbar[0:m + 1]) <= D[m]) for m in np.arange(1, M, 1)]
    # ---------- constraint (2d)
    [constraints.append(P[m, j] <= Pt) for j in range(M) for m in np.arange(j, M, 1)]
    # ---------- constraint in (6)
    q = 0
    for m in range(M):
        fSum = 0
        for j in range(m + 1):
            I = 1 + cp.sum([g[i] * P[i, j] for i in np.arange(j, m, 1)])  # see above (5)
            fSum = (fSum + aPara[m, j] + bPara[m, j] * Dbar[j] - cPara[m, j] * I
                    - dPara[m, j] * P[m, j] - new1Para[m, j] * cp.quad_over_lin((Dbar[j] + I), 4)
                    - ePara[m, j] * cp.quad_over_lin((Dbar[j] + 1), (g[m] * P[m, j] + I)))
            q = (q + cp.quad_over_lin(Dbar[j] + P[m, j], 4) + 0.25 * new2Para[m, j]  # see (7)
                 - 0.5 * new3Para[m, j] * (Dbar[j] - P[m, j]))  # see (8)
        constraints.append(fSum >= l)  # see (6)
    constraints.append(q <= Eth)  # constraint in (9)
    constraints.append(Dbar[0] == D[0])  # see footnote 1
    prob = cp.Problem(cp.Maximize(l), constraints)  # modeling the problem
    return prob


# function to solve the optimization problem
def findSolution(probModel):
    # memory allocation
    ITemp = np.tril(np.ones((M, M), dtype=float))  # memory allocation
    aTemp = np.zeros(np.shape(ITemp), dtype=float)
    bTemp = np.zeros(np.shape(ITemp), dtype=float)
    cTemp = np.zeros(np.shape(ITemp), dtype=float)
    dTemp = np.zeros(np.shape(ITemp), dtype=float)
    eTemp = np.zeros(np.shape(ITemp), dtype=float)
    new1Temp = np.zeros(np.shape(ITemp), dtype=float)
    new3Temp = np.zeros(np.shape(ITemp), dtype=float)
    # computing the constants
    for m in range(M):
        for j in range(m + 1):
            for i in np.arange(j, m, 1):
                ITemp[m, j] = ITemp[m, j] + g[i] * PCurrent[i, j]  # see below (6)
            aTemp[m, j] = (0.5 * (1 - DbarCurrent[j]) - (1 / (4 * ITemp[m, j]))
                           * ((DbarCurrent[j] - ITemp[m, j]) ** 2))  # see below (6)
            bTemp[m, j] = (2 + np.log(1 + g[m] * PCurrent[m, j] / ITemp[m, j]) + 0.5 * (DbarCurrent[j] - 1)
                           + 0.5 * ((DbarCurrent[j] / ITemp[m, j]) - 1))  # see below (6)
            cTemp[m, j] = ((((DbarCurrent[j] - 1) ** 2) / (4 * (g[m] * PCurrent[m, j] + ITemp[m, j])))
                           + 0.5 * ((DbarCurrent[j] / ITemp[m, j]) - 1))  # see below (6)
            dTemp[m, j] = (((DbarCurrent[j] - 1) ** 2) /
                           (4 * (g[m] * PCurrent[m, j] + ITemp[m, j]))) * g[m]  # see below (6)
            eTemp[m, j] = 0.25 * (g[m] * PCurrent[m, j] + ITemp[m, j])  # see below (6)
            new1Temp[m, j] = 1 / (ITemp[m, j])  # to be used in (7)
            new3Temp[m, j] = DbarCurrent[j] - PCurrent[m, j]  # to be used in (8)
    new2Temp = np.square(new3Temp)  # to be used in (8)

    # evaluate the parameters
    PCurrentPara.value = PCurrent
    DbarCurrentPara.value = DbarCurrent
    ICurrentPara.value = ITemp
    aPara.value = aTemp
    bPara.value = bTemp
    cPara.value = cTemp
    dPara.value = dTemp
    ePara.value = eTemp
    new1Para.value = new1Temp
    new2Para.value = new2Temp
    new3Para.value = new3Temp

    # solving the problem
    result = probModel.solve(verbose=False, solver=cp.MOSEK)  # solving the problem
    return l.value, P.value, Dbar.value


# system parameters
M = 4  # number of mobile users
Eth = 5e-3  # total energy budget in Joules
Pt = db2pow(4 - 30)  # total transmit power in each time slot
NoisePower = db2pow(-92 - 30)  # noise power

# user locations
start_dist = 200  # distance of the nearest user from the MEC in m
dist_gap = 50  # distance between the neighboring users
end_dist = start_dist + M * dist_gap  # distance of the M+1 th user
distance = np.arange(start_dist, end_dist, dist_gap)

# channel modeling and generation
Gamma = db2pow(8)  # reference SNR
gamma = 4  # path loss exponent
d0 = 1  # reference distance
g = (1 / NoisePower) * np.array([chanGen(distance[m]) for m in range(M)])


# deadline specification
start_deadline = 0.5  # U1 deadline in second
deadline_gap = 0.05  # deadline difference in second
end_deadline = start_deadline + M * deadline_gap  # U_{M+1} deadline in second
D = np.arange(start_deadline, end_deadline, deadline_gap, dtype=float)  # user deadlines in second
DbarCurrent = np.insert(np.zeros((M - 1,), dtype=float), 0, D[0], axis=0)  # initialization

# cvxpy variables
l = cp.Variable((1,), nonneg=True)
P = cp.Variable((M, M), nonneg=True)
Dbar = cp.Variable((M,), nonneg=True)

# cvxpy parameters
ICurrentPara = cp.Parameter((M, M))
aPara = cp.Parameter((M, M))
bPara = cp.Parameter((M, M))
cPara = cp.Parameter((M, M))
dPara = cp.Parameter((M, M))
ePara = cp.Parameter((M, M), nonneg=True)
PCurrentPara = cp.Parameter((M, M))
DbarCurrentPara = cp.Parameter((M,))

new1Para = cp.Parameter((M, M), nonneg=True)
new2Para = cp.Parameter((M, M))
new3Para = cp.Parameter((M, M))

# memory allocation
PCurrent = np.zeros((M, M), dtype=float)

# implementing the algorithm
RelChange = 1e3  # arbitrary large value
epsilon = 1e-5  # tolerance
lSeq = np.array([])
iIter = 0

# modeling the optimization problem
prob = model()  # see the paramModel function

while RelChange >= epsilon:
    lCurrent, PCurrent, DbarCurrent = findSolution(prob)
    lSeq = np.append(lSeq, 20e3 * np.log2(np.exp(1)) * lCurrent)
    if iIter > 2:
        RelChange = (lSeq[iIter] - lSeq[iIter - 1]) / lSeq[iIter - 1]
    iIter = iIter + 1
    
print('The convergence sequence: ')
print(lSeq)
plt.plot(np.arange(iIter), lSeq)
plt.xlabel('Iteration number', fontsize=15)
plt.ylabel('Minimum # offloaded bits', fontsize=15)
plt.savefig("../results/Convergence.pdf", format="pdf", bbox_inches="tight")
