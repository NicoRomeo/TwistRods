"""
submit_serial_f.jl

Core (serial) code to run simulation. Includes energy functions and various
differential equations to evolve integrator for rod.

IMPORTANT NOTE ON HOW TO RUN PARAREAL CODE:
To run parareal code, create submit_parareal.jl file that is identical to this file,
but remove "Packages" section, include @everywhere in front of every function below.

Then, uncomment include fxn in parareal_rod.jl file, line 19.
"""

"""
Packages
"""

using ColorSchemes, BenchmarkTools, DataFrames

begin
    using DifferentialEquations, LinearAlgebra
    using ForwardDiff,ReverseDiff
    using SharedArrays
    using IterativeSolvers
end

begin
    using ReverseDiff
end

"""
Basic helper functions

"""

begin
    normd(vec) = 1/sqrt(dot(vec,vec)) .* vec
    dotv(vec) = sqrt(dot(vec,vec))
end

tfac = 2.0
sfac = 1.0
mfac = 4.1
B = [1 0;0 1]

"""
Energy functions
    - b_st: isotropic case w/ bend only
        - returns E_bend + E_stretch
        - note current code is incompatible w/ b_st-- requires adjusting
    - nn_eq: isotropic/anisotropic case w/ bend & twist
        - returns E_bend + E_twist + E_stretch
    - nbend: returns Ebend only
    - ntwist: returned Etwist only
"""

function b_st(n,l0,q;sfac = 1.,mfac = 1,β =1., α=1.)
    eL = q[4:6] - q[1:3]
    tL = normd(eL)
    Ebend = 0.0
    Estretch = (dotv(eL) - l0)^2
    for i = 1:n-2
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

function nn_eq(n,l0,uq)
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

    for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 * cross(tL, tC) / (1 + tL'tC)
        kbn = dotv(kb)

        if !isapprox(kbn, 0.0;atol = 10^(-5))
            ax = kb / kbn
            phi = 2 * atan(dot(kb, ax) / 2)
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

function nbend(n,l0,uq)
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

    Ebend = 0.0

    for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 * cross(tL, tC) / (1 + tL'tC)
        kbn = dotv(kb)
        if !isapprox(kbn, 0.0;atol = 10^(-5))
            ax = kb / kbn
            phi = 2 * atan(dot(kb, ax) / 2)
            u =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end #cond
        m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
        k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]
        Ebend += k' * B * k
        eL,tL,m1,m2 = eC,tC,m1_1,m2_1
    end #loop
    Ebend = 0.5 * Ebend * mfac * (1/l0)
    return Ebend
end #function


function ntwist(n,l0,uq)
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

    Etwist = 0.0

    for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 * cross(tL, tC) / (1 + tL'tC)
        kbn = dotv(kb)

        if !isapprox(kbn, 0.0;atol = 10^(-5))
            ax = kb / kbn
            phi = 2 * atan(dot(kb, ax) / 2)
            u =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end #cond

        m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
        k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]
        Etwist += m[i]^ 2
        eL,tL,m1,m2 = eC,tC,m1_1,m2_1
    end #loop
    Etwist += m[end]^2
    Etwist = 0.5 * Etwist * (1/l0) * tfac
    return Etwist
end #function

"""
Contact handling
    - rf(v_i, v_j): repelling function. returns force given 2 vertices
    - repf(u):
        - currently O(n^2). returns forces for all vertices.
"""

function rf(i,j,d;alph = 5,ne = 2,de = 0.09803921568627451)
    ca = [0. 0. 0.]
    temp = sqrt(dot(d,d))
    if abs(i - j) > ne && temp < de
        return (1/((temp)^alph)) * normd(d)
    end #cond
    return ca
end

function repf(n,l0,uq;B = [1 0;0 1],mfac = 40.0,sc = 10^(-2))
    #pseudo
    pa = reshape(uq[4:end-n],(3,n))
    cl_array = zeros(3,n,n) #O(n^2)

    for i = 1:n
        for j = i+1:n
            d = pa[:,i] - pa[:,j]
            cl_array[:,i,j] = rf(i,j,d)
        end #loop
    end #loop
    sum_array = zeros(3,n)
    for i=1:n
        sum_array[:,i] = sum(cl_array[:,i,:],dims=2) - sum(cl_array[:,:,i],dims=2)
                                                - cl_array[:,i,i]
    end #loop
    return sc .* vec(sum_array)
end #fxn

"""
WIP
forceHand()
    - energy input
    - returns hand-calculated force
"""

function dEdm(G,A,a,b,m,l0;m̄ = 0)
    return (G * A * (a^2 + b^2)/(4 * l0)) * (m - m̄)
end #fxn

function dEde(a,b;k̄ = 0)
end #fxn

function dkde(a,b,eN,eC;k̄ = 0)
    1/(sqrt(dot(eC,eC))) * ((-k ) + ())
end #fxn

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
NOTE: CURRENTLY ONLY gTwQ! HAS CONTACT HANDLING.
Add "contact = repf(n,l0,uq)
dq_int[:] += contact[4:end-3]" right after filler = -Force computation to
include contact handling

    - Force: returns force given energy function
    - g!: diff. eq. for isotropic rod w/ bend only
    - gTw!: diff. eq. for clamped isotropic/anisotropic rod w/ bend and twist
    - gTwF!: diff. eq. for free-ended isotropic/anisotropic rod w/ bend and twist
        - WIP: needs testing
        - test w/ parareal
    - gTwINEXT!: inext. case. CURRENTLY BROKEN, newton does not converge
    - gTwQ!: for clamped rod with torque on one end
        -has contact handling

    - gNet(!)?: WIP
        - hopefully can return any combo of "effects" on rod
        - e.g. stochastic forces, contact handling, clamping, etc.
"""

"""
1. quasistatic update
2.
"""

function theta_evolve!(uq,num,qpos,bound)
    function Force_theta_interior(
        q_theta_int::Vector;
        num = n,
        vor = vor,
        fxn = nn_eq_theta_int,
        q_pos = qpos,
        boundary = bound
    )
        E = x -> fxn(num, vor, x, q_pos, boundary)
        return ForwardDiff.gradient(E,q_theta_int)
    end #func

    function Jacob_theta_interior(q_theta_int::Vector)
        E = x -> Force_theta_interior(x)
        return ForwardDiff.jacobian(E,q_theta_int)
    end #func

    println("... evolving ...")
    uq[3num + 5:end - 1] = nlsolve(Force_theta_interior, Jacob_theta_interior, uq[3num + 5:end-1]).zero[:]
end #evolve theta

# function Jacob_theta_interior(q_theta_int::Vector)
#     E = x -> Force_theta_interior(x)
#     return ForwardDiff.jacobian(E,q_theta_int)
# end #func

function Force_theta(q_theta; num = n, vor = vor, fxn = nn_eq_theta, q_pos = x_arr)
    E = x -> fxn(num, vor, x, q_pos)
    return ForwardDiff.gradient(E,q_theta)
end #func

# function Force_theta_interior(
#     q_theta_int::Vector;
#     num = n,
#     vor = vor,
#     fxn = nn_eq_theta_int,
#     q_pos = x_arr,
#     boundary = fixed_pos
# )
#     println("this is boundary: ", boundary)
#     E = x -> fxn(num, vor, x, q_pos, boundary)
#     return ForwardDiff.gradient(E,q_theta_int)
# end #func

function roll_pos(pos, theta; n = n)
    uq = zeros(length(theta) + length(pos))
    uq = vcat(pos, theta)
    return uq
end #func

function unroll_pos(q, n)
    pos = zeros(3*n + 3)
    theta = zeros(n)
    pos = q[1:3*n + 3]
    theta = q[3*n + 4:end]
    return pos, theta
end #func

function nn_eq_theta_int(
    n,
    l0,
    q_theta_int,
    q_pos,
    boundary
)
    # edges, tangent, kb, phi
    q_theta = vcat(boundary[1],q_theta_int,boundary[end])
    uq = roll_pos(q_pos, q_theta)

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

    for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 * cross(tL, tC) / (1 + tL'tC)
        kbn = dotv(kb)

        if !isapprox(kbn, 0.0;atol = 10^(-5))
            ax = kb / kbn
            phi = 2 * atan(dot(kb, ax) / 2)
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

function nn_eq_theta(
    n,
    l0,
    q_theta,
    q_pos
)
    # edges, tangent, kb, phi
    uq = roll_pos(q_pos, q_theta)

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

    for i = 1:n-2
        eC = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tC = normd(eC)
        kb = 2 * cross(tL, tC) / (1 + tL'tC)
        kbn = dotv(kb)

        if !isapprox(kbn, 0.0;atol = 10^(-5))
            ax = kb / kbn
            phi = 2 * atan(dot(kb, ax) / 2)
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

function Force(n,l0,q;fxn = nn_eq)
    E = x -> fxn(n,l0,x)
    return ForwardDiff.gradient(E,q)
end #function

function g!(dq,q,p,t)
    n,l0 = p
    dq[:] = -Force(n,l0,q,b_st)
end #function

function gTwINEXT!(out,duq,uq,p,t;nn_eq = nn_eq)
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

function gTwQ_quas!(duq,uq,p,t;nn_eq = nn_eq)
    n,l0,f1,f2,t2 = p
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

    filler = -Force(n,l0,uq) * 10
    println("this is filler, y: ",reshape(filler[7:end-(n-1)-4],(3,n-2))[3,:])
    contact = repf(n,l0,uq)

    #adding external force, clamping ends of rod
    # dq_thet[:] = zeros(n)
    dq_thet[:] = zeros(n)
    dq_thet[1] = 0.
    dq_thet[2] = 0.
    # dq_thet[end-1] = 0
    # dq_thet[end] = 0
    dq_thet[end-1] = t2
    dq_thet[end] = t2

    dq_int[:] = filler[7:end-(n-1)-4] #+ contact[4:end-3]
    gravity = reshape(zeros(length(dq_int)),(3,n-2))
    gravity[2,:] = 10.0*ones(n-2)
    gravity = vec(gravity)

    dq_int[:] += gravity
    # println(maximum(contact))
    dq_1st[:] = [0.,0.,0.]
    dq_2nd[:] = [-0.,0.,0.]
    dq_int[2:3] = [0.,0.]
    dq_int[end-1:end] = [0.,0.]
    du[:] .= 0.
end #function

function gTwQ!(duq,uq,p,t;nn_eq = nn_eq)
    n,l0,f1,f2,t2 = p
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
    contact = repf(n,l0,uq)
    #adding external force, clamping ends of rod
    dq_thet[:] = filler[end-(n-1):end]
    dq_thet[1] = 0.
    dq_thet[2] = 0.
    dq_thet[end-1] = 0.
    dq_thet[end] = 0.

    dq_int[:] = filler[7:end-(n-1)-4] #+ contact[4:end-3]
    # println(maximum(contact))
    dq_1st[:] = [0.,0.,0.]
    dq_2nd[:] = [0.,0.,0.]
    dq_int[2:3] = [0.,0.]
    dq_int[end-1:end] = [0.,0.]
    du[:] .= 0.
end #function

function gTw!(duq,uq,p,t;nn_eq = nn_eq)
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
    dq_thet[:] = filler[end-(n-1):end]
    dq_thet[1] = 0.
    dq_thet[end] = 0.
    dq_int[:] = filler[7:end-(n-1)-4]
    dq_1st[:] = [(f1 + filler[4:6])[1],0.,0.]
    dq_2nd[:] = [(f2 + filler[end-(n-1)-3:end-(n-1)-1])[1],0.,0.]
    dq_int[2:3] = [0.,0.]
    dq_int[end-1:end] = [0.,0.]
    du[:] .= 0.
end #function

function gTwFIX!(duq,uq,p,t;nn_eq = nn_eq)
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
    dq_1st[:] = [0.,0.,0.]
    dq_2nd[:] = [0.,0.,0.]
    dq_int[2:3] = [0.,0.]
    dq_int[end-1:end] = [0.,0.]
    du[:] .= 0.
end #function

function gTwF!(duq,uq,p,t;nn_eq = nn_eq)
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

function gNet!(fx_arr)
    #NOTE: complete this
end #func
