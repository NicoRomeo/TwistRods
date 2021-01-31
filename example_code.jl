include("submit.jl")

"""
Example code to run simulation
"""

n = 41 #vertices

#initialize parameters: # vertices, voronoi length, external forces @ ends of rod
p = (n,1.,[1.,0.,0.],[-1.,0.,0.])

#initialize rod configuration
q0 = Array{Float64}(undef,4*n + 3)
q0[1:3] = [0. 1. 0.]
for i = 1:n
    q0[(3*(i - 1)) + 4:(3*(i - 1)) + 6] = [i*1. 0. 0.]
end
j = 3
q0[(3*(j - 1)) + 4:(3*(j - 1)) + 6] += [0.,0.,10^(-4)]
tot_thet = 20 * pi
q0[end-(n-1):end] = 0. : tot_thet/(n-1) : tot_thet

#init. fine and coarse solvers
algF = DP5()
algG = RK4()
dtF = 10^(-5)
dtC = 10^(-4)

#init. ODE problem
tspan  = (0.,5.)
tarr = 0.:0.0625:5. #1 time-interval per processor
tcount = Int(round(tarr[2]/dtF)) + 1
tlen = length(tarr)
prob = ODEProblem(gTw!,q0,tspan,p)

#choosing diff. eq.
f! = gTw!

#init. # of parareal iterations
kcount = 5

#initial estimate of U
solG_init = solve(prob,algG;dt = dtC,adaptive=false,saveat=tarr)

#initialize arrays to hold final solutions
Ukfin_WRAPPER = Array{Float64}(undef,(tlen,n * 4 + 3))
finefinal_WRAPPER = Array{Float64}(undef,(tcount,n * 4 + 3, tlen - 1))
finalt_WRAPPER = 0.: dtF : tarr[end]
U_k = Array{Float64}(undef,(tlen,4*n + 3))
for j = 1:tlen
    U_k[j,:] = solG_init[j]
end #loop

#run simulation
int_sol = net_sim(solF_true,U_k,tarr,p,kcount,algF,algG,dtF,dtC,tcount)
Ukfin_WRAPPER[:],finefinal_WRAPPER[:,:,:] = int_sol

"""
Saving parareal data for later plotting
"""

currFine = finefinal_WRAPPER
net_rows = (tlen - 1) * (tcount - 1)
true_tc = tcount - 1
net_frame = Array{Float64}(undef,(net_rows,n * switch + fr))
for i = 1:tlen - 1
    net_frame[(((i - 1) * true_tc) + 1): i * true_tc ,:] = currFine[1:end - 1,:,i]
end #loop

#net_frame contains all plotting data for parareal soln.
