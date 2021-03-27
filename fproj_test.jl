include("/Users/cathji/Documents/GitHub/TwistRods/new_F_projection.jl")
include("/Users/cathji/Documents/GitHub/TwistRods/submit.jl")
include("/Users/cathji/Documents/GitHub/TwistRods/rod_configs.jl")

"""
SIMULATION CONDITIONS
    simulation conditions / affect funcs here, to be checked @ each timestep
    - short(): checks if rod contracted certain length
"""

function short(u,t;ends=([0.,0.0,0.0],[20.,0.0,0.0]),limit=2.0,n=21)
    bdiff = u[4:6] - ends[1]
    ediff = u[3*(n-1)+4:3*(n-1)+6] - ends[2]
    sqrt_sum = (sqrt(dot(bdiff,bdiff)) + sqrt(dot(ediff,ediff)))
    bool = limit > sqrt_sum
    return bool
end #function

"""
INITIALIZING SYSTEM
    initialize q, mass matrix (WIP)
"""
n = 21 #vertices
#initialize parameters: # vertices, voronoi length, external forces @ ends of rod

#initialize rod configuration
q0 = Array{Float64}(undef,4*n + 3)
qint,vor = twist_rod(n,20.)
q0[4:(3 * n) + 3] = qint
q0[10:12] += [0., 0., 10^(-3)]
q0[1:3] = [0. 0. 1.]

p = (n,vor,[1.,0.,0.],[-1.,0.,0.])

#initializing twist
tot_thet = 2*pi
q0[end-(n-1):end] = 0. : tot_thet/(n-1) : tot_thet

"""
INITIALIZING SOLVER
    - choose solver, timestep, tspan, etc.
"""

algF = RK4()
algG = RK4()
dtF = 10^(-4)
dtC = 10^(-5)

println("this is q0: $q0")
#choosing diff. eq.
f! = gTw!

#init. ODE problem
tspan  = (0.,1.0)
tarr = 0.:1.0:2.0 #1 time-interval per processor
tcount = Int(round(tarr[2]/dtF)) + 1
tlen = length(tarr)

prob = ODEProblem(f!,q0,tspan,p)
integrator = init(prob,algG;dt=dtC,adaptive=false)

#this is totally broken
"""
SOLVING SYSTEM
"""

fends_bool = true
fends = [0.,0.,0.,tot_thet]
etol = 10^(-3)

dt = dtC
@time while short(integrator.u,integrator.t) == true && integrator.t < 0.98*tspan[2]
    step!(integrator,dtC)
    fp_WRAP!(integrator.u,p[1],p[2],constraint_ends_fixed,fends_bool,fends,etol,dt)
    println("t: ",integrator.t)
end #loop

buckle_sol = integrator.sol
soln_plot = buckle_sol
#init. # of parareal iterations
# kcount = 5

#initial estimate of U

#NOTE: anim WIP
# @time solG_init = solve(prob,algG,callback=cb;dt = dtC,adaptive=false)
# solve(prob,algG;dt=dtC,adaptive=false)
# t_saved = @elapsed solve(prob,algG;dt = dtC,adaptive=false)
#
# soln_plot  = solG_init
# anim_xyzG = @animate for k ∈ 1:length(soln_plot)
#     temp = soln_plot.u[k]
#     temp_p = reshape(temp[1 + 3:3*n + 3],(3,n))
#     plt1 = plot(temp_p[1,:],temp_p[2,:],temp_p[3,:],legend=false,linewidth = 3)
#     title!("Twisted rod")
#     xlims!(-1.,11.)
#     ylims!(-2.,2.)
#     zlims!(-2.,2.)
#     plot(plt1)
# end every 100
#
# gif(anim_xyzG,"mygifXYZ_COARSE.gif", fps = 20)

#saving plotting data
"""
postsim, SAVING PLOTTING DATA
"""

using DelimitedFiles
using DataFrames, CSV

outWRAPPER = Array{Float64}(undef,(length(soln_plot),length(soln_plot[1])))
for i = 1:length(soln_plot)
    outWRAPPER[i,:] = soln_plot[i][:]
end #loop

outWRAPPER = Tables.table(outWRAPPER)
CSV.write("out.csv",outWRAPPER)
