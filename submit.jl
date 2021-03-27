"""
Packages
"""

using Distributed
using Plots; pyplot()
using ColorSchemes, BenchmarkTools, DataFrames
procs = 4
addprocs(procs)

@everywhere begin
    using DifferentialEquations, LinearAlgebra, Quaternions
    using ForwardDiff,ReverseDiff
    using SharedArrays
    using IterativeSolvers
end

@everywhere begin
    using ReverseDiff
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

@everywhere function nn_eq(n,l0,uq;tfac = 0.789,sfac = 1.,B = [1 0;0 1],mfac = 1.345)
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

# @everywhere function nn_eq(n,l0,uq;tfac = 0.789,sfac = 300.,B = [1 0;0 1],mfac = 1.345)
#     # edges, tangent, kb, phi
#     q = uq[4:end]
#     mat_f = uq[1:3]
#     theta = q[3*n + 1:end]
#
#     m = diff(theta, dims = 1)
#     eL = q[4:6] - q[1:3]
#     tL = normd(eL)
#     u,v = mat_f,cross(tL,mat_f)
#     m1 = cos(theta[1]) * u + sin(theta[1]) * v
#     m2 = -sin(theta[1]) * u + cos(theta[1]) * v
#
#     Ebend,Etwist = 0.0,0.0
#     Estretch = (dotv(eL) - l0)^2
#
#     @inbounds for i = 1:n-2
#         eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
#         tC = normd(eC)
#         kb = 2 * cross(tL, tC) / (1 + tL'tC)
#         kbn = dotv(kb)
#         if !isapprox(kbn, 0.0)
#             phi = 2 * atan(dot(kb, kb / kbn) / 2)
#             ax = kb / kbn
#             u =
#                 dot(ax, u) * ax +
#                 cos(phi) * cross(cross(ax, u), ax) +
#                 sin(phi) * cross(ax, u)
#         end #cond
#         m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
#         m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
#         k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]
#         Ebend += k' * B * k
#         Etwist += m[i]^ 2
#         Estretch += (dotv(eC) - l0)^2
#         eL,tL,m1,m2 = eC,tC,m1_1,m2_1
#     end #loop
#     Etwist += m[end]^2
#     Ebend = 0.5 * Ebend * mfac * (1/l0)
#     Estretch = 0.5 * Estretch * sfac * (1/l0)
#     Etwist = 0.5 * Etwist * (1/l0) * tfac
#     return Ebend + Etwist + Estretch
# end #function

@everywhere function dEdm(G,A,a,b,m,l0;m̄ = 0)
    return (G * A * (a^2 + b^2)/(4 * l0)) * (m - m̄)
end #fxn

@everywhere function dEde(a,b;k̄ = 0)

end #fxn

@everywhere function dkde(a,b,eN,eC;k̄ = 0)
    1/(sqrt(dot(eC,eC))) * ((-k ) + ())
end #fxn

"""
WIP
forceHand()
    - energy input
    - returns hand-calculated force
"""

# @everywhere function forceHand()
#
#     e_vec = q[4:end] - q[1:end-3]
#     e0 = e_vec[1:3]
#     t0 = normd(e0)
#
#     eE = e_vec[end-2:end]
#     tE = normd(eE)
#
#     eP = e_vec[end-5:end-3]
#     tP = normd(eP)
#
#     #stretch
#     F_s1 = @view F_s[1:3]
#     F_s2 = @view F_s[end-2:end]
#     F_si = @view F_s[4:end-3]
#
#     F_s1[:] = E * A * ((sqrt(dot(e0,e0))/l0) - 1) * t0
#     F_s2[:] = -E * A * ((sqrt(dot(eE,eE))/l0) - 1) * tE
#
#     for i = 1:n-2
#         eC = e_vec[(3*(i-1))+4:(3*(i-1))+6]
#         eN = e_vec[(3*(i))+4:(3*(i))+6]
#         tC = normd(eC)
#         tN = normd(eN)
#
#         F_si[(3*(i-1))+1:(3*(i-1))+3] = (-E * A * ((sqrt(dot(eC,eC))/l0) - 1) * tC) +
#                                         (E * A * ((sqrt(dot(eN,eN))/l0) - 1) * tN)
#     end #loop
#
#     #twist, moment
#     F_t1 = @view F_t[1:3]
#     F_t2 = @view F_t[end-2:end]
#     F_ti = @view F_t[4:end-3]
#
#     M_t1 = @view M_t[1]
#     M_t2 = @view M_t[end]
#     M_ti = @view M_t[2:end-1]
#
#     kb = 2 * cross(tL, tC) / (1 + tL'tC)
#     kbE = 2 * cross(tE,tP) / (1 + tE'tP)
#     kbP = 2 * cross() / ()
#     m = theta[1]
#     F_t1[:] = dEdm(G,A,a,b,theta[1],l0) * (1/(2 * l0)) * kb
#     F_t2[:] = (dEdm(G,A,a,b,theta[end],l0) *
#             (1/(sqrt(dot(eL,eL))) - 1/(sqrt(dot(eP,eP)))) * kbE) -
#             (dEdm(G,A,a,b,theta[end-1],l0) * () * ()) #handle edge case
#     M_t1[:] = -dEdm(G,A,a,b,theta[1],l0) + dEdm(G,A,a,b,theta[2],l0)
#     M_t2[:] = -dEdm(G,A,a,b,theta[end-1],l0)
#
#     for i = 1:n-2
#         eC = e_vec[(3*(i-1))+4:(3*(i-1))+6]
#         eN = e_vec[(3*(i))+4:(3*(i))+6]
#         tC = normd(eC)
#         tN = normd(eN)
#
#         F_ti[(3*(i-1))+1:(3*(i-1))+3] = (dEdm(G,A,a,b,theta[i],l0) *
#                 (1/(2 * sqrt(dot(eN,eN))) - 1/(2 * sqrt(dot(eC,eC)))) * kbN) -
#
#                 (dEdm(G,A,a,b,theta[i-1],l0) *
#                 (1/(2 * sqrt(dot(eC,eC)))) * (1/(2 * sqrt(dot(eN,eN)))) * kbC) +
#
#                 (dEdm(G,A,a,b,theta[i+1],l0) *
#                 (1/(2 * sqrt(dot(eN,eN)))) * (1/(2 * sqrt(dot(eF,eF)))) * kbF)
#
#         M_t[i] = -dEdm(G,A,a,b,theta[i],l0) + dEdm(G,A,a,b,theta[i+1],l0)
#     end #loop
#
#     #bend
#     F_b1 = @view F_b[1:3]
#     F_b2 = @view F_b[end-2:end]
#     F_bi = @view F_b[4:end-3]
#
#     F_s =
#     F_t =
#     F_b =
# end #function

"""
Setting up Diff. eq
    - Force: returns force given energy function
    - g!: diff. eq. for isotropic rod w/ bend only
    - gTw!: diff. eq. for clamped isotropic/anisotropic rod w/ bend and twist
    - gTwF!: diff. eq. for free-ended isotropic/anisotropic rod w/ bend and twist
    - gTwINEXT!: inext. case. CURRENTLY BROKEN, newton does not converge
"""

@everywhere function Force(n,l0,q;fxn = nn_eq)
    E = q -> fxn(n,l0,q)
    # println(ReverseDiff.gradient(E,q))
    # println("This is reverse: ", ReverseDiff.gradient(E,q))
    # println("This is forward: ", ForwardDiff.gradient(E,q))
    return ForwardDiff.gradient(E,q)
end #function

"""
interlude:
brief grad tests (rev vs. fdiff)
"""
# qint,vor = twist_rod(n)
# q0[4:(3 * n) + 3] = qint
# q0[10:12] += [0., 0., 10^(-4)]
# q0[1:3] = [0. 0. 1.]
#
# p = (n,vor,[1.,0.,0.],[-1.,0.,0.])
# for i = 1:n
#     q0[(3*(i - 1)) + 4:(3*(i - 1)) + 6] = [0.]
# end #loop
#initializing twist
# tot_thet = 27 * 2 * pi
# q0[end-(n-1):end] = 0. : tot_thet/(n-1) : tot_thet
#
# n = 101
# l0 = vor
# E = q -> nn_eq(n,l0,q)
# # println(ReverseDiff.gradient(E,q))
# Force(q) = ReverseDiff.gradient(E,q)
# ForceF(q) = ForwardDiff.gradient(E,q)
# new_F = -Force(q0)
# println(new_F)

@everywhere function g!(dq,q,p,t)
    n,l0 = p
    dq[:] = -Force(n,l0,q,b_st)
end #function


@everywhere function gTwINEXT!(out,duq,uq,p,t;nn_eq = nn_eq)
    n,l0,f1,f2 = p
    u = @view uq[1:3]
    q_inner = @view uq[4:end-(n-1)-1-n]
    e_inner = q_inner[4:end] - q_inner[1:end-3]

    q_thet = @view uq[end-(n-1)-n:end-n]
    q_int = @view uq[7:end-(n-1)-4-n]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3-n:end-(n-1)-1-n]

    du = @view duq[1:3]
    dq_thet = @view duq[end-(n-1)-n:end-n]
    dq_int = @view duq[7:end-(n-1)-4-n]
    dq_1st = @view duq[4:6]
    dq_2nd = @view duq[end-(n-1)-3-n:end-(n-1)-1-n]

    out_du = @view out[1:3]
    out_dq_thet = @view out[end-(n-1)-n:end-n]
    out_dq_int = @view out[7:end-(n-1)-4-n]
    out_dq_1st = @view out[4:6]
    out_dq_2nd = @view out[end-(n-1)-3-n:end-(n-1)-1-n]
    out_inext = @view out[end-(n-1):end]

    filler = -Force(n,l0,uq)
    #adding external force, clamping ends of rod
    out_dq_thet[1] = -dq_thet[1]
    out_dq_thet[end] = -dq_thet[end]
    out_dq_int[:] = filler[7:end-(n-1)-4-n] - dq_int[:]
    out_dq_1st[:] = [(f1 + filler[4:6])[1],0.,0.] - dq_1st[:]
    out_dq_2nd[:] = [(f2 + filler[end-(n-1)-3-n:end-(n-1)-1-n])[1],0.,0.] - dq_2nd[:]
    out_dq_int[2:3] = [0.,0.] - dq_int[2:3]
    out_dq_int[end-1:end] = [0.,0.] - dq_int[end-1:end]
    out_du[:] = - du[:]

    for i = 1:n-1
        ed = e_inner[3*(i-1)+1:3*(i-1)+3]
        out_inext[i] = sqrt(dot(ed,ed)) - 1.0
    end #loop
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
    ref = @view uq[1:3]
    q_thet = @view uq[end-(n-1):end]
    q_int = @view uq[7:end-(n-1)-4]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3:end-(n-1)-1]

    dref = @view duq[1:3]
    dq_thet = @view duq[end-(n-1):end]
    dq_int = @view duq[7:end-(n-1)-4]
    dq_1st = @view duq[4:6]
    dq_2nd = @view duq[end-(n-1)-3:end-(n-1)-1]

    e_1st = q_int[1:3] - q_1st
    t_1st = normd(e_1st)
    # println("why do i have to do this")
    # println("before: $u")
    ref[:] -= t_1st * dot(t_1st,ref)
    ref[:] = normd(ref[:])

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

    dref[:] = cross(cross(t_1st,dt),ref)
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

"""
net simulation
    - runs simulation given approp. input
"""

function net_sim(U_k,t_arr,p,k_count,algFi,algCo,dtF,dtC,tcount)
    tlen = length(t_arr)
    @inbounds for k = 1:k_count-1
        c_crrt!(U_k,t_arr,p,algFi,algCo,dtF,dtC)
    end #loop
    U_k,final_fine = c_crrtFinal(U_k,t_arr,p,algFi,algCo,dtF,dtC,tcount)
    return U_k,final_fine
end #function
