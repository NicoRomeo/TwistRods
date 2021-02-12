include("/Users/cathji/Documents/GitHub/TwistRods/submit.jl")

"""
RUNNING SIMULATION
"""

n = 21 #vertices

#initialize parameters: # vertices, voronoi length, external forces @ ends of rod
p = (n,1.,[3.,0.,0.],[-3.,0.,0.])

#initialize rod configuration
q0 = Array{Float64}(undef,4*n + 3)
q0[1:3] = [0. 1. 0.]
for i = 1:n
    q0[(3*(i - 1)) + 4:(3*(i - 1)) + 6] = [i*1. 0. 0.]
end #loop
j = 3
q0[(3*(j - 1)) + 4:(3*(j - 1)) + 6] += [0.,0.,0.1]
tot_thet = 5 * pi
q0[end-(n-1):end] = 0. : tot_thet/(n-1) : tot_thet

#init. fine and coarse solvers
algF = RK4()
algG = RK4()
dtF = 10^(-4)
dtC = 10^(-3)

#choosing diff. eq.
# f! = gTw!
f! = gTwF!

#init. ODE problem
tspan  = (0.,10.)
tarr = 0.:0.1:10. #1 time-interval per processor
tcount = Int(round(tarr[2]/dtF)) + 1
tlen = length(tarr)
prob = ODEProblem(f!,q0,tspan,p)

#init. # of parareal iterations
kcount = 5

#initial estimate of U
solG_init = solve(prob,algG;dt = dtC,adaptive=false)

soln_plot  = solG_init
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
SAVING PLOTTING DATA
"""

using DelimitedFiles
using DataFrames, CSV

outWRAPPER = Array{Float64}(undef,(length(soln_plot),length(soln_plot[1])))
for i = 1:length(soln_plot)
    outWRAPPER[i,:] = soln_plot[i][:]
end #loop

outWRAPPER = Tables.table(outWRAPPER)
CSV.write("out.csv",outWRAPPER)
