using Distributed
rmprocs(workers())
addprocs(6)

@everywhere begin
    using DifferentialEquations, LinearAlgebra
    using ForwardDiff, ReverseDiff
    using DistributedArrays,SharedArrays
    using Plots, ColorSchemes, DelimitedFiles, BenchmarkTools, DataFrames
end

println("things are staeting to happen")

"A cluster spanning machines using the --machine-file option. This uses a passwordless ssh login to start Julia worker processes (from the same path as the current host) on the specified machines."

"""
BASIC HELPER FUNCTIONS
"""

# MPI.Init()
#
# comm = MPI.COMM_WORLD
# rank = MPI.Comm_rank(comm)
# size = MPI.Comm_size(comm)
@everywhere begin
    normd(vec) = 1/sqrt(dot(vec,vec)) .* vec
    dotv(vec) = sqrt(dot(vec,vec))
end

"""
ENERGY FUNCTION
    - b_st: return E_bend + E_stretch
    - nn_eq: return E_bend + E_twist + E_stretch
"""

@everywhere function b_st(n,l0,q;mfac = 6,β =1., α=1.)
    # edges, tangent, kb, phi
    x = q[1:3*n]
    #init Ebend, stretch
    eL = q[4:6] - q[1:3]
    tL = normd(eL)
    Ebend = 0.0
    s = dotv(tL) .- l0
    Estretch = s^2 #Estretch is edge value
    for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)] #current edge
        tC = normd(eC)
        kb = 2 .* cross(tL, tC) / (1 + tL'tC)
        ell = 0.5 * (dotv(eL) + dotv(eC))
        Ebend +=  α * dot(kb,kb)/ ell
        s = dotv(eC) .- l0
        Estretch += s^2
        # update for next vertex
        eL,tL = eC,tC
    end #loop
    ell = 0.5 * dotv(eL) #edge case
    Ebend = 0.5 * Ebend * mfac
    Estretch = 0.5 * Estretch
    return Ebend + Estretch
end #function

@everywhere function nn_eq(n,l0,uq;B = [1 0;0 1],mfac = 6)
    # edges, tangent, kb, phi
    q = uq[4:end]
    x = q[1:3*n]
    theta = q[3*n + 1:end]
    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    #init Ebend, twist, stretch
    eL = q[4:6] - q[1:3]
    tL = normd(eL)

    mat_f = uq[1:3]
    u,v = mat_f,cross(tL,mat_f)
    m1 = cos(theta[1]) * u + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u + cos(theta[1]) * v
    Ebend = 0.0
    Etwist = 0.0
    s = sqrt(eL'eL) .- l0
    Estretch = s * s #Estretch is edge value

    for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 .* cross(tL, tC) / (1 + tL'tC)
        kbn = sqrt(kb'kb)
        ell = 0.5 * (sqrt(eL'eL) + sqrt(eC'eC))

        if !isapprox(kbn, 0.0)
            phi = 2 * atan(dot(kb, kb / kbn) / 2)
            ax = kb / kbn
            u =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end #conditional, RRF
        # u,v = mat_f[:,1],mat_f[:,2]
        m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
        k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]

        Ebend += k' * B * k / ell
        Etwist += m[i] .^ 2 / ell
        s = sqrt(eC'eC) .- l0
        Estretch += s * s

        # update for next vertex
        eL,tL,m1,m2 = eC,tC,m1_1,m2_1
    end #loop

    ell = 0.5 * sqrt(eL'eL) #edge case
    Etwist += m[end] .^ 2 / ell
    Ebend = 0.5 * Ebend * mfac
    Estretch = 0.5 * Estretch
    Etwist = 0.5 * Etwist
    return Ebend + Etwist + Estretch
end #function

"""
SETTING UP DIFFEQ
"""

@everywhere function Force(n,l0,q)
    E = q -> b_st(n,l0,q)
    return ForwardDiff.gradient(E,q)
end #function

@everywhere function g!(dq,q,p,t) #diffeq
    n,l0 = p
    dq[:] = -Force(n,l0,q)
end #function

@everywhere function ForceTw(n,l0,q)
    E = q -> nn_eq(n,l0,q)
    return ForwardDiff.gradient(E,q)
end #function

@everywhere function gTw!(duq,uq,p,t) #diffeq
    n,l0 = p
    u = @view uq[1:3]
    q = @view uq[4:end]
    du = @view duq[1:3]
    dq = @view duq[4:end]

    t = normd(q[4:6] - q[1:3])
    dq[:] = -ForceTw(n,l0,q)
    dt = dq[4:6] - dq[1:3]
    du[:] = shit #TODO: fix this

    n,l0 = Int(p[1]),p[2]
    dq = FF(n,l0,vcat(vec(u[2,:,1:n]),u[3,1,1:n]),u[1,1:3,1:3])
    du[2,:,1:n] = reshape(dq[1:3n],3,n)
    du[3,1,1:n] = dq[3n+1:end]
    t = normd(u[2,:,2] - u[2,:,1])
    du[1,1:3,1] = cross(cross(t,du[2,:,2] - du[2,:,1]),du[1,1:3,1])
    du[1,1:3,2] = cross(cross(t,du[2,:,2] - du[2,:,1]),du[1,1:3,2])
    du[1,1:3,3] = cross(cross(t,du[2,:,2] - du[2,:,1]),du[1,1:3,3])
end #function

"""
PARAREAL ALG
    - prrl_C and prrl_F return integrators

TODO:
    - 1. fix solution handling for integrator
    - 2. test
    - 3. better integrate w/ integrator interface
"""
function prrl_C(prob,Δt,algC,dts)
    integrator = init(prob,algC;dt = dts,adaptive=false)
    step!(integrator,Δt,true)
    return integrator.t,integrator.u
end #function

@everywhere function prrl_Final(prob,Δt,algF,dts)
    integrator = init(prob,algF;dt=dts,adaptive=false,save_everystep=true)
    step!(integrator,Δt,true)
    # println("integrator.sol specs: ",typeof(integrator.sol.u),size(integrator.sol.u))
    # println("integrator.u specs: ",integrator.t)
    # # println(hcat(integrator.sol.u...)')
    # println("not broken")
    cat = hcat(integrator.sol.u...)'
    return integrator.t,integrator.u,cat,integrator.sol.t
end #function

@everywhere function prrl_F(prob,Δt,algF,dtF)
    integrator = init(prob,algF;dt = dtF,adaptive=false)
    step!(integrator,Δt,true)
    return integrator.t,integrator.u
end #function

function Final_int(U_k,t_arr,p,algF,dtF,tcount;F = prrl_Final,g! = g!) #NOTE: insert array dim
    tlen = length(t_arr)
    F_up = SharedArray{Float64}(size(U_k))
    fine_sol = SharedArray{Float64,3}(tcount,size(U_k)[2],tlen-1)
    fine_sol_t = SharedArray{Float64,2}(tcount,tlen-1)
    Δt = t_arr[2]
    tspan = (0.,Δt)
    # F_up = Array{Float64}(undef,tlen,size(U_k)[1],size(U_k)[2])
    @sync @distributed for i = 1:tlen - 1
        # println("is something happening")
        q0 = U_k[i,:]
        prob = ODEProblem(g!,q0,tspan,p) #how efficient is this?
        # println(size(fine_sol))
        # println(size(F(prob,Δt,algF,dtF)[3]))
        # println("this is fine_sol slice size: ",size(fine_sol[:,:,i]))
        temp,F_up[i+1,:],fine_sol[:,:,i],fine_sol_t[:,i] = F(prob,Δt,algF,dtF)
    end # loop #that's hot
    return F_up,fine_sol,fine_sol_t
end #prrl F integrator

function F_int(U_k,t_arr,p,algF,dtF;F = prrl_F,g! = g!) #NOTE: insert array dim
    tlen = length(t_arr)
    F_up = SharedArray{Float64}(size(U_k))
    Δt = t_arr[2]
    tspan = (0.,Δt)
    # F_up = Array{Float64}(undef,tlen,size(U_k)[1],size(U_k)[2])
    @sync @distributed for i = 1:tlen - 1
        q0 = U_k[i,:]
        prob = ODEProblem(g!,q0,tspan,p) #how efficient is this?
        F_up[i+1,:] = F(prob,Δt,algF,dtF)[2]
    end # loop #that's hot
    return F_up
end #prrl F integrator

function G_int(U_k,t_arr,p,algC,dtC;G = prrl_C,g! = g!) #NOTE: insert array dim
    tlen = length(t_arr)
    G_up = copy(U_k)
    Δt = t_arr[2]
    tspan = (0.,Δt)
    # F_up = Array{Float64}(undef,tlen,size(U_k)[1],size(U_k)[2])
    for i = 1:tlen - 1
        q0 = U_k[i,:]
        prob = ODEProblem(g!,q0,tspan,p) #how efficient is this?
        G_up[i+1,:] = G(prob,Δt,algC,dtC)[2]
    end # loop #that's hot
    return G_up
end #prrl F integrator

function F_intTw(U_k,t_arr,p;F = prrl_F,gTw! = gTw!) #NOTE: insert array dim
    tlen = length(t_arr)
    F_up = SharedArray{Float64}(size(U_k))
    Δt = t_arr[2]
    tspan = (0.,Δt)
    # F_up = Array{Float64}(undef,tlen,size(U_k)[1],size(U_k)[2])
    @sync @distributed for i = 1:tlen - 1
        # println("is something happening")
        q0 = U_k[i,:]
        prob = ODEProblem(gTw!,q0,tspan,p) #how efficient is this?
        F_up[i+1,:] = F(prob,Δt)[2]
    end # loop #that's hot
    return F_up
end #prrl F integrator

function G_intTw(U_k,t_arr,p;G = prrl_C,gTw! = gTw!) #NOTE: insert array dim
    tlen = length(t_arr)
    G_up = copy(U_k)
    Δt = t_arr[2]
    tspan = (0.,Δt)
    # F_up = Array{Float64}(undef,tlen,size(U_k)[1],size(U_k)[2])
    for i = 1:tlen - 1
        q0 = U_k[i,:]
        prob = ODEProblem(gTw!,q0,tspan,p) #how efficient is this?
        G_up[i+1,:] = G(prob,Δt)[2]
    end # loop #that's hot
    return G_up
end #prrl F integrator

"""
Testing F_int
    - Comparable times to prrl_F (overhead is negligible)
    - can it be sped up w/ integrator interface?
"""

function G1(yj,tarr,p,algG,dta;G = prrl_C,g! = g!)
    q0 = yj
    Δt = tarr[2]
    prob = ODEProblem(g!,q0,(0.,tarr[2]),p)
    integrator = init(prob,algG;dt=dta,adaptive=false)
    step!(integrator,Δt,true)
    return integrator.u
end #function

function G1Tw(yj,tarr,p;algG = Euler(),dta = 10^(-4),G = prrl_C,gTw! = gTw!)
    q0 = yj
    Δt = tarr[2]
    prob = ODEProblem(gTw!,q0,(0.,tarr[2]),p)
    integrator = init(prob,algG;dt=dta,adaptive=false)
    step!(integrator,Δt,true)
    return integrator.u
end #function

function c_crrt(U_k,t_arr,p,algF,algC,dtF,dtC;G = prrl_C, F = prrl_F)
    U_kn = SharedArray{Float64}(size(U_k))
    U_kn[1,:] = U_k[1,:]
    # G_pr = init #NOTE: fix
    tlen = length(t_arr)
    # println("%%%()*%%%%%%%)(*#@)(*@#)(*@#)(&@#)(*@#)(*)%%%%%%%%%%%%")
    # println("this is U_k: ", U_k)
    F_net = F_int(U_k,t_arr,p,algF,dtF)
    G_net_p = G_int(U_k,t_arr,p,algC,dtC)
    # println("this is F_net", F_net[:,end])
    # println("this is G_net", G_net_p[:,end])
    Δt = t_arr[2]
    for i = 2:tlen
        G_nk = G1(U_kn[i-1,:],t_arr,p,algC,dtC) #oh god
        # println("%%%%%%%%%%%%%%%%%%%%%%")
        # println("this is U_kn: ", U_kn[i-1,:])
        # println("this is G_nk: ", G_nk)
        # println("***")
        # println("this is fine: ", F_net[i,:])
        # println("***")
        # println("this is coarse: ",G_net_p[i,:])
        # println(G_nk,G_net_p[i,:])
        U_kn[i,:] = G_nk + F_net[i,:] - G_net_p[i,:]
    end #loop
    return U_kn,U_kn - U_k
end #function

function c_crrtFinal(U_k,t_arr,p,algF,algC,dtF,dtC,tcount;G = prrl_C, F = prrl_F)
    U_kn = SharedArray{Float64}(size(U_k))
    U_kn[1,:] = U_k[1,:]
    # G_pr = init #NOTE: fix
    tlen = length(t_arr)
    # println("%%%()*%%%%%%%)(*#@)(*@#)(*@#)(&@#)(*@#)(*)%%%%%%%%%%%%")
    # println("this is U_k: ", U_k)
    F_net,fine_sol,fine_sol_t = Final_int(U_k,t_arr,p,algF,dtF,tcount)
    G_net_p = G_int(U_k,t_arr,p,algC,dtC)
    # println("this is F_net", F_net[:,end])
    # println("this is G_net", G_net_p[:,end])
    Δt = t_arr[2]
    for i = 2:tlen
        G_nk = G1(U_kn[i-1,:],t_arr,p,algC,dtC) #oh god
        # println("%%%%%%%%%%%%%%%%%%%%%%")
        # println("this is U_kn: ", U_kn[i-1,:])
        # println("this is G_nk: ", G_nk)
        # println("***")
        # println("this is fine: ", F_net[i,:])
        # println("***")
        # println("this is coarse: ",G_net_p[i,:])
        # println(G_nk,G_net_p[i,:])
        U_kn[i,:] = G_nk + F_net[i,:] - G_net_p[i,:]
    end #loop
    return U_kn,U_kn - U_k,fine_sol,fine_sol_t
end #function

function c_crrtTw(U_k,t_arr,p;G = prrl_C, F = prrl_F)
    U_kn = SharedArray{Float64}(size(U_k))
    U_kn[1,:] = U_k[1,:]
    # G_pr = init #NOTE: fix
    tlen = length(t_arr)
    # println("%%%()*%%%%%%%)(*#@)(*@#)(*@#)(&@#)(*@#)(*)%%%%%%%%%%%%")
    # println("this is U_k: ", U_k)
    F_net = F_intTw(U_k,t_arr,p)
    G_net_p = G_intTw(U_k,t_arr,p)
    # println("this is F_net", F_net[:,end])
    # println("this is G_net", G_net_p[:,end])
    Δt = t_arr[2]
    for i = 2:tlen
        G_nk = G1Tw(U_kn[i-1,:],t_arr,p) #oh god
        # println("%%%%%%%%%%%%%%%%%%%%%%")
        # println("this is U_kn: ", U_kn[i-1,:])
        # println("this is G_nk: ", G_nk)
        # println("***")
        # println("this is fine: ", F_net[i,:])
        # println("***")
        # println("this is coarse: ",G_net_p[i,:])
        # println(G_nk,G_net_p[i,:])
        U_kn[i,:] = G_nk + F_net[i,:] - G_net_p[i,:]
    end #loop
    return U_kn,U_kn - U_k
end #function

function net_sim(U_k,t_arr,p,k_count,algFi,algCo,dtF,dtC,tcount;crrt = c_crrt)
    diff = Array{Float64}(undef,(size(U_k)[1],size(U_k)[2],k_count))
    tlen = length(t_arr)
    final_fine = Array{Any}(undef,tlen-1)
    for k = 1:k_count-1
        U_k,diff[:,:,k] = c_crrt(U_k,t_arr,p,algFi,algCo,dtF,dtC)
    end #loop
    U_k,diff[:,:,k_count],final_fine,final_fine_t = c_crrtFinal(U_k,t_arr,p,algFi,algCo,dtF,dtC,tcount)
    return U_k,diff,final_fine,final_fine_t
end #function

function net_simTw(U_k,t_arr,p,k_count;c_crrtTw = c_crrtTw)
    diff = Array{Float64}(undef,(size(U_k)[1],size(U_k)[2],k_count))
    for k = 1:k_count
        U_k,diff[:,:,k] = c_crrtTw(U_k,t_arr,p)
    end #loop
    return U_k,diff
end #function

using Statistics

"""
RUNNING SIM
"""

# algF = RK4()
# algG = Euler()
# tarr = 0.:1.:10.0
# tspan = (0.,tarr[end])
# tlen = length(tarr)
# n,l0 = 20,1.
# p = (n,l0)
# q0 = rand(Float64,3*n)
# prob = ODEProblem(g!,q0,tspan,p)
# solG_init = solve(prob,algG;dt = 10^(-4),adaptive=false,saveat=tarr)
# solG_init.u
# U_k = SharedArray{Float64}((tlen,3*n))
#
# for i = 1:tlen
#     U_k[i,:] = solG_init[i]
# end #loop
#
# k = 4
# @time U_kfin,nnet = net_sim(U_k,tarr,p,k)

function rodTest!(inputArr,
                tArr_PRRL,
                tArr_CON,
                algF,
                algG,
                dtF,
                dtC,
                tarr,
                tcount,
                Ukfin_WRAPPER,
                nnet_WRAPPER,
                finefinal_WRAPPER,
                finalt_WRAPPER
)
    l0 = 1.
    for i = 1:length(inputArr)
        n = inputArr[i]
        tspan = (0.,tarr[end])
        tlen = length(tarr)
        p = (n,l0)
        q0 = rand(Float64,3*n)
        prob = ODEProblem(g!,q0,tspan,p)
        solG_init = solve(prob,algG;dt = dtC,adaptive=false,saveat=tarr)
        U_k = SharedArray{Float64}((tlen,3*n))
        for j = 1:tlen
            U_k[j,:] = solG_init[j]
        end #loop
        k = 4
        #warmup
        int_sol = net_sim(U_k,tarr,p,k,algF,algG,dtF,dtC,tcount)
        Ukfin_WRAPPER[i],nnet_WRAPPER[i],finefinal_WRAPPER[i],finalt_WRAPPER[i] = int_sol
        println("hey")
        println("this is nnet: ",nnet_WRAPPER[i])
        tArr_PRRL[i] = @elapsed net_sim(U_k,tarr,p,k,algF,algG,dtF,dtC,tcount)
        tArr_CON[i] = @elapsed solve(prob,algF;dt = dtF,adaptive=false,saveat=tarr)
        println("done with $i")
    end #loop
end #function

inputArr = [5,5,5,5,5]
# inputArr = [25,30,35,40]
tArr_PRRL = Array{Float64}(undef,length(inputArr))
tArr_CON = Array{Float64}(undef,length(inputArr))

algF = RK4()
algG = Euler()
dtF = 10^(-5)
dtC = 10^(-4)
tarr = 0.:1.:10.
tlen = length(tarr)
in_len = length(inputArr)
tcount = Int(round(1/dtF)) + 2 #NOTE: FIX

Ukfin_WRAPPER = Array{Any}(undef,in_len)
nnet_WRAPPER = Array{Any}(undef,in_len)
finefinal_WRAPPER = Array{Any}(undef,in_len)
finalt_WRAPPER = Array{Any}(undef,in_len)

rodTest!(inputArr,tArr_PRRL,tArr_CON,algF,algG,dtF,dtC,tarr,tcount,Ukfin_WRAPPER,nnet_WRAPPER,finefinal_WRAPPER,finalt_WRAPPER)
Ukfin_WRAPPER
nnet_WRAPPER
finefinal_WRAPPER
finalt_WRAPPER
tArr_CON
tArr_PRRL

"""
PLOTS
"""
pl1 = plot()
for i = 1:
end #loop
display(pl1)

"""
SIMULATION 1, 12/27 12 AM
    - using 6 cores

    algF = RK4()
    algG = Euler()
    dtF = 10^(-5)
    dtC = 10^(-4)
    tarr = 0.:1.:10.

    SOLVING DIRECTLY:
    5: 28.178100216 s
    10: 110.452675808 s
    15: 323.369510414 s
    20: 588.289440384 s
    25: 942.404383938 s
    30: 1217.519318214 s
    35: 1626.384560602 s
    40: 2180.000558705 s

    SOLVING USING PRRL, 6 core computer:
    5: 44.447938538 s
    10: 163.32808088 s
    15: 399.334093219 s
    20: 711.000077901 s
    25: 1215.970474558 s
    30: 1611.954713998 s
    35: 2111.612593471 s
    40: 2873.794242304 s
"""

"""
SIMULATION 2, 12/27 3 LM

TODO BEFORE SUPERCOMPUTER ACCESS:
1. Parse in parallel fine solution, integrator interface? Not sure...
2. Write as much of paper as possible.

For 12/27:
1. Parse in fine solution (1-2 hrs)
2. Write up rod introduction, introduce parareal alg / read up (1-2 hrs)

For 12/28:
1. Code up visualization *sob* (1-2 hrs)
2. Continue reading / witing up on parareal alg (1-2 hrs)

"""
# """
# PLOTTING
# """
#
# @time solF = solve(prob,DP5(),dt = 10^(-5),adaptive=false,saveat=tarr)
# pl1 = plot(solF); display(pl1)
#
# @time solCoarse = solve(prob,algG,dt = 10^(-4),adaptive=false,saveat=tarr)
# plot!(solCoarse); title!("Coarse vs. fine"); display(pl1)
#
# pl3 = plot(solF)
# for i = 1:size(U_kfin)[2]
#     plot!(tarr,U_kfin[:,i])
# end #loop
# title!("Prrl vs. fine"); display(pl3)
#
# pl2 = plot(solF);title!("Fine solution")
#
# pl4 = plot()
# for i = 1:size(U_kfin)[2]
#     plot!(tarr,U_kfin[:,i])
# end #loop
# title!("parareal only"); plot(pl1,pl2,pl3,pl4,legend=false)
#
# # function true_plots()
# # end #function
# #
# # function mean_plots(nloss_net,mean_arr,tarr,k_count)
# #     tlen = length(tarr)
# #     plt1 = plot()
# #     for k = 1:k_count
# #         mean_arr[k,:] = mean(nloss_net[k,:,:],dims=1)
# #         scatter!(k*ones(15),mean_arr[k,:])
# #     end #loop
# #     display(plt1)
# # end #function
# #
# # using Plots
# # gr()
# # k = 5
# # mean_arr = Array{Float64}(undef,(k,15))
# # mean_plots(nloss_net,mean_arr,tarr,k)
# #
# # """
# # vs.
# # ODE SOLVE
# #     -@btime 3.912 s
# #
# # @btime bench_solF = solve(prob,RK4();dt=10^(-5), adaptive = false)
# # """
# # n,l0 = 5,1.
# # q0 = rand(Float64,15)
# # tspan = (0.,1.0)
# # prob = ODEProblem(g!,q0,tspan,(n,l0))
# #
# # @time solve(prob,RK4();dt=10^(-5), adaptive = false)
#
# # #for iteration k (series)
# # G(y) = coarse
# # F(y) = fine
# #
# # #over j = 1:P, in parallel using MPI, or somethiing of sort
# # y_{j+1} = G(y^{k+1}_{j}) + F(y_{j}) - G(y^{k}_{j})
