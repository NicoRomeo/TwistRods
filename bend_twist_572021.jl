include("/Users/cathji/Documents/GitHub/TwistRods/new_F_projection.jl")
include("/Users/cathji/Documents/GitHub/TwistRods/submit_serial_f.jl")
include("/Users/cathji/Documents/GitHub/TwistRods/rod_configs.jl")

words = readlines("/Users/cathji/Documents/GitHub/TwistRods/sim_file_final.txt")
"""
INITIALIZING SYSTEM
    initialize q, mass matrix (WIP)
"""

time_scale = 10.0 # s
length_scale = 10 # cm
mass_scale = 0.1 # kg

n = 75 #vertices
vtot = 10.
limit = 1.
#initialize rod configuration
q0 = zeros(4*n + 3)
# qint,vor = bent_rod(n,vtot)
qint,vor = bent_rod(n,vtot)
q0[4:(3 * n) + 3] = qint
q0[1:3] = [0. 0. 1.]
tlim = 54*pi

pFac = 0.01
cFac = 0.01

etol = 10^(-3)
dtC = 0.00001

q0[13:15] += [0., 0., 0.]

f = 0.
tq = 1000*pi
p = (n,vor,[0,0.,0.],[-0,0.,0.],tq)

#initializing twist
tot_thet = 0.
if tot_thet > 0
    q0[end-(n-1):end] = 0. : tot_thet/(n-1) : tot_thet
end #cond
tdens = tot_thet/n

"""
INITIALIZING SOLVER
    - choose solver, timestep, tspan, etc.
"""

algF = RK4()
algG = RK4()
dtF = 0.1
dtC = dtC

println("this is q0: $q0")
#choosing diff. eq.
f! = gTwQ_quas!

#init. ODE problem
tspan  = (0.,100.)
tarr = 0.:1.0:2.0 #1 time-interval per processor
tcount = Int(round(tarr[2]/dtF)) + 1
tlen = length(tarr)

prob = ODEProblem(f!,q0,tspan,p)
integrator = init(prob,algG;dt=dtC,adaptive=false)

"""
SOLVING SYSTEM
- pushing ends together
"""

etol = etol

dt = dtC

"""
SIMULATION CONDITIONS
    simulation conditions / affect funcs here, to be checked @ each timestep
    - short(u): checks if rod contracted certain length
    - tw(u):

    - main_push(): pushing two ends of clamped rod together
    - main_twist(): twisting one end of clamped rod
    -
"""

function short(u,t;ends = ([0.,0.0,0.0],[vtot,0.0,0.0]),limit=limit,n=n)
    bdiff = u[4:6] - ends[1]
    ediff = u[3*(n-1)+4:3*(n-1)+6] - ends[2]
    sqrt_sum = (sqrt(dot(bdiff,bdiff)) + sqrt(dot(ediff,ediff)))
    bool = limit > sqrt_sum
    return bool
end #function

function tw(u;tlim = tlim)
    bool = (u[end] < tlim)
    return bool
end #function

function main_push()
    while short(integrator.u,integrator.t) == true && integrator.t < 0.98*tspan[2]
        step!(integrator,dtC)
        fp_WRAP!(integrator.u,p[1],p[2],constraint_ends_fixed,etol,dt)
        println("t: ",integrator.t)
    end #loop
end #loop

function int_plot(ui)
    plt = plot()
    xire = reshape(ui[4:end-n],(3,n))
    # # xire[2,:]
    plot!(xire[1,:],xire[2,:],xire[3,:])
    # plot!(xire[1,:],xire[2,:],ylim = (-10.,10.))
end #function

function int_xy(ui)
    plt = plot()
    xire = reshape(ui[4:end-n],(3,n))
    # # xire[2,:]
    plot!(xire[1,:],xire[2,:])
    # plot!(xire[1,:],xire[2,:],ylim = (-10.,10.))
end #function

function main_twist(int,a)
    while tw(int.u) == true && int.t < 0.98*tspan[2]

        pos_t, theta_t = unroll_pos(int.u,p[1])
        x_arr = pos_t
        fixed_pos = (theta_t[1],theta_t[end])
        first_theta = copy(int.u[3n+4:end])
        println("this is theta init: ", first_theta[end-3:end])
        theta_evolve!(int.u,p[1],x_arr,fixed_pos)
        # println("this is theta_init after: ",int.u[3n+4:end])
        # println("this is THETA DIFF: ", int.u[3n+4:end] - first_theta)
        step!(int,dtC)
        fp_WRAP!(int.u,p[1],p[2],constraint_ends_fixed,etol,dt)
        # push!(a[1],nn_eq(n,l0,int.u))
        # push!(a[2],nbend(n,l0,int.u))
        # push!(a[3],ntwist(n,l0,int.u))
        #

        if isapprox(mod(int.t,dtC*100),0,;atol = dtC*1.1)
            println("t: ", int.t)
            display(int_plot(integrator.u))
        end #cond
    end #loop
end #func

#init energy plot
# ETwplot = plot(title="ETwplot")
# Eplot = plot("Eplot")

using Plots
using NLsolve

b = [[],[],[]]
timeTwist = main_twist(integrator, b)
int_plot(integrator.u)
int_xy(integrator.u)
# time1000 = @elapsed main_push()
soln_push = integrator.sol
soln_t = integrator.t

"""
SOLVING SYSTEM:
- TWISTING END
"""

algF = RK4()
algG = RK4()

n,l0,f1,f2,t2 = p

t2 = 27 * 2 * pi
f! = gTwQ!
p = (n,vor,[0.,0.,0.],[-0.,0.,0.],t2)
q1 = integrator.u
tspan2 = (0.,1.0)
prob = ODEProblem(f!,q1,tspan2,p)
integrator_twist = init(prob,algG;dt=dtC,adaptive=false)

time_tw = @elapsed main_twist(integrator_twist)
soln_twist = integrator_twist.sol

"""
SOLVING SYSTEM:
- FIXING ENDS
"""

algF = RK4()
algG = RK4()
# dtF = 10^(-4)
# dtC = 10^(-4)

f! = gTwFIX!
p = (n,vor,[0.,0.,0.],[-0.,0.,0.])
q1 = integrator.u
tspan2 = (0.,100.)
prob = ODEProblem(f!,q1,tspan2,p)
dtC = 0.001
integrator_fixed = init(prob,algG;dt=dtC,adaptive=false)

while integrator_fixed.t < 0.98*tspan2[2]
    step!(integrator_fixed,dtC)
    fp_WRAP!(integrator_fixed.u,p[1],p[2],constraint_ends_fixed,fends_bool,fends,etol,dt)
    println("t: ",integrator_fixed.t)
end #loop

buckle_sol = integrator_fixed.sol
soln_plot = buckle_sol

"""
postsim, SAVING PLOTTING DATA
"""

using DelimitedFiles
using DataFrames, CSV

function outWRAP(soln_arr)
    tl = 0
    finsol = soln_arr[1]
    for i = 2:length(soln_arr)
        finsol = hcat(Array(finsol),Array(soln_arr[i]))
    end #loop
    return finsol
end #func

tl = length(soln_push)+length(soln_plot)
outWRAPPER = Array{Float64}(undef,(tl,length(soln_plot[1])))
for i = 1:length(soln_push)
    outWRAPPER[i,:] = soln_push[i][:]
end #loop

for i = 1:length(soln_plot)
    outWRAPPER[i+length(soln_push),:] = soln_plot[i][:]
end #loop

outWRAPPER = Tables.table(outWRAPPER)
CSV.write("out.csv",outWRAPPER)

tl = length(soln_push)
outWRAPPER = Array{Float64}(undef,(tl,length(soln_push[1])))
for i = 1:length(soln_push)
    outWRAPPER[i,:] = soln_push[i][:]
end #loop

soln_arr1 = [soln_push,buckle_sol]
outWRAPPER5 = Tables.table(outWRAP(soln_arr1))
CSV.write("/Users/cathji/Documents/GitHub/TwistRods/out.csv",outWRAPPER5)

q = integrator.u
l0 = vor
bf = Force(n,l0,q;fxn = nn_eq)
println(bf)

temp = integrator.u

tl = length(soln_push)
outWRAPPER = Array{Float64}(undef,(tl,length(soln_push[1])))
for i = 1:length(soln_push)
    outWRAPPER[i,:] = soln_push[i][:]
end #loop

outWRAPPER = Tables.table(outWRAPPER)
CSV.write("/Users/cathji/Documents/GitHub/TwistRods/out.csv",outWRAPPER)
