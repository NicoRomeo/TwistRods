"""
Packages
"""

using Distributed
using Plots; pyplot()
using ColorSchemes, BenchmarkTools, DataFrames
procs = 10
addprocs(procs)

@everywhere begin
    using DifferentialEquations, LinearAlgebra, Quaternions
    using ForwardDiff
    using SharedArrays
end

"""
Basic helper functions
"""

@everywhere begin
    normd(vec) = 1/sqrt(dot(vec,vec)) .* vec
    dotv(vec) = sqrt(dot(vec,vec))
end

"""
Energy functions
    - b_st: isotropic case w/ bend only
        - returns E_bend + E_stretch
        - note current code is incompatible w/ b_st-- requires adjusting
    - nn_eq: isotropic/anisotropic case w/ bend & twist
        - returns E_bend + E_twist + E_stretch
"""

@everywhere function b_st(n,l0,q;sfac = 1.,mfac = 1,β =1., α=1.)
    eL = q[4:6] - q[1:3]
    tL = normd(eL)
    Ebend = 0.0
    Estretch = (dotv(eL) - l0)^2
    @inbounds for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 * cross(tL, tC) / (1 + tL'tC)
        Ebend +=  α * dot(kb,kb)
        Estretch += (dotv(eC) - l0)^2
        eL,tL = eC,tC
    end #loop
    Ebend = 0.5 * Ebend * mfac * (1/l0)
    Estretch = 0.5 * Estretch * sfac * (1/l0)
    return Ebend + Estretch
end #function

@everywhere function nn_eq(n,l0,uq;tfac = 1.,sfac = 100.,B = [1 0;0 1],mfac = 10.)
    # edges, tangent, kb, phi
    q = uq[4:end]
    mat_f = uq[1:3]
    theta = q[3*n + 1:end]

    m = diff(theta, dims = 1)
    eL = q[4:6] - q[1:3]
    tL = normd(eL)
    u,v = mat_f,cross(tL,mat_f)
    m1 = cos(theta[1]) * u + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u + cos(theta[1]) * v

    Ebend,Etwist = 0.0,0.0
    Estretch = (dotv(eL) - l0)^2

    @inbounds for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 * cross(tL, tC) / (1 + tL'tC)
        kbn = dotv(kb)
        if !isapprox(kbn, 0.0)
            phi = 2 * atan(dot(kb, kb / kbn) / 2)
            ax = kb / kbn
            u =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end #cond
        m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
        k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]
        Ebend += k' * B * k
        Etwist += m[i]^ 2
        Estretch += (dotv(eC) - l0)^2
        eL,tL,m1,m2 = eC,tC,m1_1,m2_1
    end #loop
    Etwist += m[end]^2
    Ebend = 0.5 * Ebend * mfac * (1/l0)
    Estretch = 0.5 * Estretch * sfac * (1/l0)
    Etwist = 0.5 * Etwist * (1/l0) * tfac
    return Ebend + Etwist + Estretch
end #function

"""
Setting up Diff. eq
    - Force: returns force given energy function
    - g!: diff. eq. for isotropic rod w/ bend only
    - gTw!: diff. eq. for clamped isotropic/anisotropic rod w/ bend and twist
    - gTwF!: diff. eq. for free-ended isotropic/anisotropic rod w/ bend and twist
"""

@everywhere function Force(n,l0,q;fxn = nn_eq)
    E = q -> fxn(n,l0,q)
    return ForwardDiff.gradient(E,q)
end #function

@everywhere function g!(dq,q,p,t)
    n,l0 = p
    dq[:] = -Force(n,l0,q,b_st)
end #function

@everywhere function gTw!(duq,uq,p,t;nn_eq = nn_eq)
    n,l0,f1,f2 = p
    u = @view uq[1:3]
    q_thet = @view uq[end-(n-1):end]
    q_int = @view uq[7:end-(n-1)-4]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3:end-(n-1)-1]

    du = @view duq[1:3]
    dq_thet = @view duq[end-(n-1):end]
    dq_int = @view duq[7:end-(n-1)-4]
    dq_1st = @view duq[4:6]
    dq_2nd = @view duq[end-(n-1)-3:end-(n-1)-1]

    filler = -Force(n,l0,uq)
    #adding external force, clamping ends of rod
    dq_thet[1] = 0.
    dq_thet[end] = 0.
    dq_int[:] = filler[7:end-(n-1)-4]
    dq_1st[:] = [(f1 + filler[4:6])[1],0.,0.]
    dq_2nd[:] = [(f2 + filler[end-(n-1)-3:end-(n-1)-1])[1],0.,0.]
    dq_int[2:3] = [0.,0.]
    dq_int[end-1:end] = [0.,0.]
    du[:] .= 0.
end #function

@everywhere function gTwF!(duq,uq,p,t;nn_eq = nn_eq)
    n,l0,f1,f2 = p
    u = @view uq[1:3]
    q_thet = @view uq[end-(n-1):end]
    q_int = @view uq[7:end-(n-1)-4]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3:end-(n-1)-1]

    du = @view duq[1:3]
    dq_thet = @view duq[end-(n-1):end]
    dq_int = @view duq[7:end-(n-1)-4]
    dq_1st = @view duq[4:6]
    dq_2nd = @view duq[end-(n-1)-3:end-(n-1)-1]

    e_1st = q_int[1:3] - q_1st
    t_1st = normd(e_1st)
    # println("why do i have to do this")
    # println("before: $u")
    # u[:] -= t_1st * dot(t_1st,u)
    # u[:] = normd(u[:])
    # println("after: $u")

    filler = -Force(n,l0,uq)
    #adding external force, clamping ends of rod
    dq_thet[:] = filler[end-(n-1):end]
    dq_int[:] = filler[7:end-(n-1)-4]
    dq_1st[:] = filler[4:6]
    dq_2nd[:] = filler[end-(n-1)-3:end-(n-1)-1]

    dq_Δ = dq_int[1:3] - dq_1st
    γ = 2 * dot(e_1st,dq_Δ)
    ρ = sqrt(dot(e_1st,e_1st))
    dt = ((ρ * (dq_Δ)) - (e_1st * 0.5 * (1/(sqrt(dot(e_1st,e_1st)))) * γ)) / ρ^2
    println("this is the diff: ", cross(cross(t_1st,dt),u))

    du[:] = cross(cross(t_1st,dt),u)
end #function


"""
Components of parareal algorithm
    - prrl_C: initializes and steps coarse integrator
    - prrl_F: initializes and steps fine integrator
    - prrl_Final: same as prrl_F, except returns intermediate fine solution

    - F_int: parallel intermediate fine solver (multiple shooting)
    - Final_int: same as F_int, except returns intermediate fine solution

    - G_int: intermediate coarse solver
"""

function prrl_C(prob,Δt,algC,dts)
    integrator = init(prob,algC;dt = dts,adaptive=false)
    step!(integrator,Δt)
    return integrator.t,integrator.u
end #function

@everywhere function prrl_Final(prob,Δt,algF,dts)
    integrator = init(prob,algF;dt=dts,adaptive=false,save_everystep=true)
    step!(integrator,Δt)
    cat = hcat(integrator.sol.u...)'
    return integrator.t,integrator.u,cat
end #function

@everywhere function prrl_F(prob,Δt,algF,dtF)
    integrator = init(prob,algF;dt = dtF,adaptive=false)
    step!(integrator,Δt)
    return integrator.t,integrator.u
end #function

function Final_int(U_k,t_arr,p,algF,dtF,tcount;F = prrl_Final,f! = gTw!) #NOTE: insert array dim
    tlen = length(t_arr)
    F_up = SharedArray{Float64}(size(U_k))
    fine_sol = SharedArray{Float64,3}(tcount,size(U_k)[2],tlen-1)
    Δt = t_arr[2]
    tspan = (0.,Δt)
    @sync @distributed for i = 1:tlen - 1
        q0 = U_k[i,:]
        prob = ODEProblem(f!,q0,tspan,p)
        temp,F_up[i+1,:],fine_sol[:,:,i] = F(prob,Δt,algF,dtF)
    end # loop
    return F_up,fine_sol
end #prrl F integrator

function F_int(U_k,t_arr,p,algF,dtF;F = prrl_F,f! = gTw!) #NOTE: insert array dim
    tlen = length(t_arr)
    F_up = SharedArray{Float64}(size(U_k))
    Δt = t_arr[2]
    tspan = (0.,Δt)
    @sync @distributed for i = 1:tlen - 1
        q0 = U_k[i,:]
        prob = ODEProblem(f!,q0,tspan,p)
        F_up[i+1,:] = F(prob,Δt,algF,dtF)[2]
    end # loop
    return F_up
end #prrl F integrator

function G_int(U_k,t_arr,p,algC,dtC;G = prrl_C,f! = gTw!) #NOTE: insert array dim
    tlen = length(t_arr)
    G_up = copy(U_k)
    Δt = t_arr[2]
    tspan = (0.,Δt)
    @inbounds for i = 1:tlen - 1
        q0 = U_k[i,:]
        prob = ODEProblem(f!,q0,tspan,p)
        G_up[i+1,:] = G(prob,Δt,algC,dtC)[2]
    end # loop
    return G_up
end #function

"""
Building parareal algorithm simulation
    - G1: functionally similar to prrl_C
    - c_crrt!: in-place implementation of 1 iteration of parareal algorithm
    - c_crrtFinal: same as c_crrt!,
      except returns additional intermediate fine solution
    - net_sim: parareal algorithm w/ multiple iterations
"""

function G1(yj,tarr,p,algG,dta;G = prrl_C,f! = gTw!)
    q0 = yj
    Δt = tarr[2]
    prob = ODEProblem(f!,q0,(0.,tarr[2]),p)
    integrator = init(prob,algG;dt=dta,adaptive=false)
    step!(integrator,Δt,true)
    return integrator.u
end #function

function c_crrt!(U_k,t_arr,p,algF,algC,dtF,dtC;G = prrl_C, F = prrl_F)
    tlen = length(t_arr)
    F_net = F_int(U_k,t_arr,p,algF,dtF)
    G_net_p = G_int(U_k,t_arr,p,algC,dtC)
    Δt = t_arr[2]
    @inbounds for i = 2:tlen
        G_nk = G1(U_k[i-1,:],t_arr,p,algC,dtC)
        U_k[i,:] = G_nk + F_net[i,:] - G_net_p[i,:]
    end #loop
end #function

function c_crrtFinal(U_k,t_arr,p,algF,algC,dtF,dtC,tcount;G = prrl_C, F = prrl_F)
    tlen = length(t_arr)
    F_net,fine_sol = Final_int(U_k,t_arr,p,algF,dtF,tcount)
    G_net_p = G_int(U_k,t_arr,p,algC,dtC)
    Δt = t_arr[2]
    @inbounds for i = 2:tlen
        G_nk = G1(U_k[i-1,:],t_arr,p,algC,dtC)
        U_k[i,:] = G_nk + F_net[i,:] - G_net_p[i,:]
    end #loop
    return U_k,fine_sol
end #function

function net_sim(U_k,t_arr,p,k_count,algFi,algCo,dtF,dtC,tcount)
    tlen = length(t_arr)
    @inbounds for k = 1:k_count-1
        c_crrt!(U_k,t_arr,p,algFi,algCo,dtF,dtC)
    end #loop
    U_k,final_fine = c_crrtFinal(U_k,t_arr,p,algFi,algCo,dtF,dtC,tcount)
    return U_k,final_fine
end #function
