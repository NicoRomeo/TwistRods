using Plots, ColorSchemes, DelimitedFiles, BenchmarkTools, DataFrames
using IterativeSolvers, NLsolve, Zygote, LinearAlgebra, DifferentialEquations
using Base.threads, Distributed

include("newrod.jl")
## Functions to access states and back
# function state2vars(state::Array{Float64,1}, n::Integer)
# function vars2state(x::Array{Float64},theta::Array{Float64},u0::Array{Float64})
##

"""
ENERGY !!!OLD!!! (note can get rid of rodrigues rotation)
    energy_q(q, u0::Array{Float64,1}, p)
        q : position vector
        u0 : normal vector @ x1
        p : [n,undeformed vor length]
returns bend, twist, stretch energy for given config

ENERGY NEW
    n_eq(n,l0,q, mat_f;B, mfac)
        n : # of vertices
        l0 : undeformed vor length
        q : position vector
        mat_f : material frame
        B : bending matrix
        mfac : bending modulus
returns bend, twist, stretch energy for given config
notes: parallelize over vertices?? static?
"""

function energy_q(q, u0::Array{Float64,1}, p)
    n = Int(p[1])
    l0 = p[2]
    B = [1 0; 0 1]

    # edges, tangent, kb, phi
    X = q[1:3*n]
    theta = q[3*n+1:end]

    #edges = X[:, 2] - X[:, 1]
    edges = q[4:6] - q[1:3]
    tangent = normd(edges)
    ell = 0.5 * sqrt(edges'edges)

    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    u = normd(u0)
    v = cross(tangent, u)
    m1 = cos(theta[1]) * u + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u + cos(theta[1]) * v
    Ebend = 0.0
    Etwist = 0.0
    s = sqrt(edges'edges) - l0
    Estretch = s's
    for i = 1:n-2
        edges_1 = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tangent_1 = normd(edges_1)
        kb = 2 .* cross(tangent, tangent_1) / (1 + tangent'tangent_1)
        kbn = sqrt(kb'kb)
        ell = 0.5 * (sqrt(edges'edges) + sqrt(edges_1'edges_1))
        if !isapprox(kbn, 0.0)
            phi = 2 * atan(dot(kb, kb / kbn) / 2)
            ax = kb / kbn
            u =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end #conditional
        v = cross(tangent_1, u)
        m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
        k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]

        Ebend += k' * B * k / ell
        Etwist += m[i] .^ 2 / ell
        s = sqrt(edges_1'edges_1) .- l0
        Estretch += s * s
        # update for next vertex
        edges = edges_1
        tangent = tangent_1
        m1 = m1_1
        m2 = m2_1
    end #loop

    ell = 0.5 * sqrt(edges'edges)
    Etwist += m[end] .^ 2 / ell
    mfac = 6 #bending modulus
    Ebend = 0.5 * Ebend * mfac
    Estretch = 0.5 * Estretch
    Etwist = 0.5 * Etwist
    return Ebend + Etwist #+ Estretch
end # function

function n_eq(n,l0,q,mat_f;B = [1 0;0 1],mfac = 6)
    # edges, tangent, kb, phi
    x = q[1:3*n]
    theta = q[3*n + 1:end]
    println(theta)
    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    #init Ebend, twist, stretch
    eL = q[4:6] - q[1:3]
    tL = normd(eL)

    m1 = cos(theta[1]) * mat_f[:,:,1][:,1] + sin(theta[1]) * mat_f[:,:,1][:,2]
    m2 = -sin(theta[1]) * mat_f[:,:,1][:,1] + cos(theta[1]) * mat_f[:,:,1][:,2]
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

        # if !isapprox(kbn, 0.0)
        #     phi = 2 * atan(dot(kb, kb / kbn) / 2)
        #     ax = kb / kbn
        #     u =
        #         dot(ax, u) * ax +
        #         cos(phi) * cross(cross(ax, u), ax) +
        #         sin(phi) * cross(ax, u)
        # end #conditional, RRF
        u,v = mat_f[:,:,i+1][:,1],mat_f[:,:,i+1][:,2]

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

    return Ebend + Etwist #+ Estretch
end #function

#to-do: create mat frame function

function nn_eq(n,l0,q,mat_f;B = [1 0;0 1],mfac = 6)
    # edges, tangent, kb, phi
    x = q[1:3*n]
    theta = q[3*n + 1:end]
    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    #init Ebend, twist, stretch
    eL = q[4:6] - q[1:3]
    tL = normd(eL)

    u,v = mat_f[:,1],mat_f[:,2]
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

    return Ebend + Etwist #+ Estretch
end #function

function F(n,l0,q,mat_f)
    E = mu -> n_eq(n,l0,mu,mat_f)
    f = -1.0 * Zygote.gradient(E, q)[1]
    return f
end #func

function FF(n,l0,q,mat_f)
    E = mu -> nn_eq(n,l0,mu,mat_f)
    f = -1.0 * Zygote.gradient(E, q)[1]
    return f
end #func

# n = 5
# l0 = 1.
# q = [1. 0. 0. 2. 0. 0. 3. 0. 0. 4. 0. 0. 5. 0. 0. 0. 1. 2. 3. 4.]
# u0 = [0., 0., 1.]
# mat_f = zeros(3,3,5)
# mat_f[:,:,1] = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'
# mat_f[:,:,2] = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'
# mat_f[:,:,3] = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'
# mat_f[:,:,4] = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'
# mat_f[:,:,5] = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'

# #TEST OF FF
# n = 5
# l0 = 1.
# q = [1. 0. 0. 2. 0. 0. 3. 0. 0. 4. 0. 0. 5. 0. 0. 0. 1. 2. 3. 4.]
# mat_f = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'
#
# # d = @view u[1:3,:,1:n-1]
# FF(n,l0,q,mat_f)
# # n_eq(n,l0,q,mat_f;B = [1 0;0 1],mfac = 6)

"""
HELPER FUNCTIONS

    skewmat(a)

    rotab(a, b, c)
rotation matrix sending unit vectors a to b, applied to c

    F! !!!OLD!!!
in-place dE/dx method, internal
"""
function skewmat(a::Array{Float64})
    return [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
end # function

function rotab(a::Array{Float64,1}, b::Array{Float64,1}, c::Array{Float64,1})
    axb = cross(a, b)
    V = skewmat(axb)
    u = c + cross(axb, c) + V * V * c / (1 + dot(a, b))
    return u
end # func

function F!(f::Array{Float64,1}, q::Array{Float64,1}, u0, param)
    E = mu -> energy_q(mu, u0, param)
    f[:] = -1.0 * Zygote.gradient(E, q)[1]
end #func

n_eq(n,l0,q,mat_f;B = [1 0;0 1],mfac = 6)

function G!(f,n,l0,q,mat_f)
    E = mu -> n_eq(n,l0,mu,mat_f)
    f[:] = -1.0 * Zygote.gradient(E, q)[1]
end #func
"""
HELPER FUNCTIONS CONT.

    vor_length(x_inp)
        x_inp : q in Bergou 2008
return vector of voronoi length for given config

    mass_matrix(n,ρ,r)
        n : # of vertices
        ρ : mass density / unit volume
        r : radius in unit
        l : vor length in unit
returns WHOLE mass matrix for right cylindrical rod

    mass_matrix_int(n,ρ,r)
        n : # of vertices
        ρ : mass density / unit volume
        r : radius in unit
        l : vor length in unit
returns INTERIOR mass matrix for right cylindrical rod (fast projection)
"""

function vor_length(x_inp)
    x_inp = vec(x_inp)
    n = size(x_inp)[1] / 3
    n = Int(n)
    x_inp = reshape(x_inp, (3, n))
    edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]
    vl = zeros(n - 1) #1 per edge
    for i = 1:n-1
        vl[i] = sqrt(dot(edges[:, i], edges[:, i]))
    end #loop
    return vl
end #function

function mass_matrix(n,ρ,r,l)
    M = zeros(4n,4n)
    mI = 1. * I
    Mn = ρ * pi * (r^2) * l
    M[1:3n,1:3n] = Mn * Matrix(mom_of_i,3n,3n)
    M[3n + 1:end,3n + 1:end] = 0.5 * Mn * (r^2) * Matrix(mom_of_i,n,n)
    return M
end #function

function mass_matrix_int(n,ρ,r,l)
    M = zeros(4n-6,4n-6)
    mI = 1. * I
    Mn = ρ * pi * (r^2) * l
    M[1:3n - 6,1:3n - 6] = Mn * Matrix(mI,3n,3n)
    M[3n - 5:end,3n - 5:end] = 0.5 * Mn * (r^2) * Matrix(mI,n,n)
    return M
end #function

"""
START OF TIMEP + IMPLICIT PSEUDOCODE

    f_int(q,mf)
        q : position vector
        mf : material frame @ each of n vertices
returns dx/dt (internal force on each vertex)

    f_ext(q)
        q : position vector
returns dx/dt (external force on each vertex)
"""

#f_int calculations. need to parralelize
# function f_int(q,mf)
#     F!()
# end #function
#
# function f_ext(q)
# end #function

"""
1. NEWTON'S METHOD, simulation loop

    newt_s
        n : # of vertices
        M : 4n x 4n mass matrix
        h : dt
        q : orientation vector
        q̇ : time derivative of q
        f_ext : external force
returns Δq & Δq̇ using newton's method

WANT TO FIND Δq̇,Δq w/ newton's method (implicit)
(1) N1(Δq,Δq̇) = (M * Δq̇)
                - h*(f_int(q + Δq) + f_ext(t,q,q̇))
(2) N2(Δq,Δq̇) = Δq - h*(Δq̇ + q̇)
"""

function newt_s!(F,Δq,Δq̇,n,M,h,q,q̇,f_ext)
    F[1] = (M * dq[2]) - h*(f_int(q + dq[1]) + f_ext(q))
    F[2] = dq[1] - h*(dq[2] + q̇)
end #function

# #actually solving, include in timestep
# newt_f!(F,dq) = newt_s!(F,dq,n,M,h,q,q̇,f_ext)
# res = nlsolve(newt_f!,dq,method=:newton) #include inline

"""
2. DiffEq.jl SIMULATION LOOP
"""

function ts(du,u,p,t;f = n_eq)
    n,l0 = Int(p[1]),p[2]

    q = @view u[2,:,1:n]
    d = @view u[1,1:3,1:3]
    dq = @view du[2,:,1:n]
    dd1 = @view du[1,1:3,1]
    dd2 = @view du[1,1:3,2]
    dd3 = @view du[1,1:3,3]

    dq = -FF(n,l0,q,d)
    dd1 = cross(cross(dd3,dq[4:6] - dq[1:3]),dd1)
    dd2 = cross(cross(dd3,dq[4:6] - dq[1:3]),dd2)
    dd3 = cross(cross(dd3,dq[4:6] - dq[1:3]),dd3)
end #function

n = 5
l0 = 1.
q = [1. 0. 0. 2. 0. 0. 3. 0. 0. 4. 0. 0. 5. 0. 0. 0. 1. 2. 3. 4.]
mat_f = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'
u0 = zeros(2,4,n)
# q = @view u[2,:,1:n]
# d = @view u[1,1:3,1:3]
u0[2,:,1:n] = q
u0[1,1:3,1:3] = mat_f

prob = ODEProblem(ts,u0,(0.,2.),[n,l0])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

println(sol)
"""
T-PARALLEL MAT FRAME UPDATE

    upd_matp(mat_f)
        n : # of vertices
        q̇ : time derivative of q
        mat_f : material frame defined on n vertices
return d(D_i)/dt
"""

#note mat_f[:,3] (tangent) must be updated before updating d1,d2
#!!! CHECK TPARALLEL UPDATE !!!
function upd_matp(n,q̇,mat_f)
    @Threads.thread for i = 1:n - 1
        mat_f[i,1] = cross(cross(mat_f[i,3],q̇[i+1] - q̇[i]),mat_f[i,1])
        mat_f[i,2] = cross(cross(mat_f[i,3],q̇[i+1] - q̇[i]),mat_f[i,2])
    end #loop
end #function

"""
END
"""

function timestep_axis(
    F!,
    f::Array{Float64,1},
    state::Array{Float64,1},
    dt::Float64,
    param,
    txt_array,
    err_tol,
    fixed_ends,
    constraint_func,
    gen_t,
    imp_F
)
    n = Int(param[1])
    l0 = param[2]
    x, theta, u0 = state2vars(state, n)
    init_state = state[4:end]
    init_x = x
    # println("init_state, start of timestep: ", init_state)

    x_endpt_init = x[3n-2:3n]

    q_i = vcat(vec(x), theta)
    tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])

    # Compute explicit initial guess for X, theta at t+dt

    #RK4
    q0 = vcat(vec(x), theta)
    tangent0 = LinearAlgebra.normalize!(q0[4:6] - q0[1:3])

    F!(f, q0, u0, param)
    k1 = f
    q1 = (dt / 2) * f + q0
    e1 = LinearAlgebra.normalize!(q1[4:6] - q1[1:3])
    u1 = rotab(tangent0, e1, u0)

    F!(f, q1, u1, param)
    k2 = f
    q2 = (dt / 2) * f + q1
    e2 = LinearAlgebra.normalize!(q2[4:6] - q2[1:3])
    u2 = rotab(tangent0, e2, u0)

    F!(f, q2, u2, param)
    k3 = f
    q3 = (dt / 2) * f + q2
    e3 = LinearAlgebra.normalize!(q3[4:6] - q3[1:3])
    u3 = rotab(tangent0, e3, u0)

    F!(f, q3, u3, param)
    k4 = f
    q4 = (dt) * f + q3
    e4 = LinearAlgebra.normalize!(q4[4:6] - q4[1:3])
    u4 = rotab(tangent0, e4, u0)

    x_fin = q0 + (1 / 6 * dt * (k1 + 2k2 + 2k3 + k4))
    e_fin = LinearAlgebra.normalize!(x_fin[4:6] - x_fin[1:3])
    u_fin = rotab(tangent0, e_fin, u0)

    #fixing force on x_fin
    force_internal = x_fin[1:3n] - vec(init_x)
    # println("this is force_internal: ", force_internal)

    x_fin[1:3] = x[1:3]
    x_fin[3n-1:3n] = zeros(2)
    # x_fin[3n-1] = x_endpt_init[2]+ (imp_F* 0.1 * dt)
    x_fin[3n-2] = x_fin[3n-2] + (imp_F * dt)
    x_fin[3n+1] = fixed_ends[1][2]
    x_fin[end] = fixed_ends[2][2]

    x_pre_proj = x_fin[1:3n]
    init = vcat(u_fin, x_fin)

    """
    CONSTRAINTS
    """
    j = 0

    #lumped model
    #init mass
    ℳ = n #total scalar mass
    m_k = ℳ / n #assuming uniform rod

    x = reshape(x_fin[1:3n], (3, n)) #reshape x, twist last row
    theta_proj = x_fin[3n+1:end]
    edges = x[:, 2:end] - x[:, 1:end-1] #defining edges

    # #init generalized mass matrix
    M = zeros(4n - 6, 4n - 6)
    mass_dens = 1.0
    rad_edge = 1.0

    # mom_of_i = ((mass_dens * rad_edge^3 * 2*pi *l0)/3)I
    mom_of_i = 1. * I
    M[1:3n-6, 1:3n-6] = m_k * Matrix(mom_of_i, 3n - 6, 3n - 6)

    #moment of inertia defined on edges, cylindrical rod
    #this is assuming i know how to integrate
    M[3n-5:end, 3n-5:end] = Matrix(mom_of_i, n, n) #(mass_dens * rad_edge^3 * 2*pi)/3 *

    #computing inverse
    #check if faster way to do this
    M_inv = inv(M)

    C = constraint_func(x, theta_proj, l0, fixed_ends, gen_t)
    C_abs = zeros(size(C)[1])

    for i = 1:size(C)[1]
        C_abs[i] = abs(C[i])
    end #for
    maxC = maximum(C_abs)

    vor_len = vor_length(x)

    io = open(txt_array[1], "a")
    writedlm(io, C)
    write(io, "* \n")
    close(io)

    io = open(txt_array[2], "a")
    writedlm(io, "* ", vor_len)
    close(io)

    itc = 1

    cap = 100
    # while maxC >= err_tol || maxC <= -err_tol
    while maxC >= err_tol
        #vectorize
        x = vec(x)
        gradC = zeros(n - 1 + 2 + 2, 4 * n - 6)

        for r = 2:n-2
            gradC[r, 3*(r-2)+1:3*(r-2)+3] = -2 * edges[:, r]
            gradC[r, 3*(r-2)+4:3*(r-2)+6] = 2 * edges[:, r]
        end #for

        #NOTE gradC term for first and last vertex is removed
        #similarly, first and last vertex removed from mass matrix
        gradC[1, 1:3] = 2 * (edges[:, 1])
        gradC[n-1, 3*((n-1)-2)+1:3*((n-1)-2)+3] = -2 * (edges[:, n-1])

        # println(gradC)
        # v = l0 #v = velocity
        # trunc_len = 0.3
        total_theta = fixed_ends[2][2]

        # fin_first = fixed_ends[1]
        # fin_last = fixed_ends[2] - (gen_t * v) * [trunc_len;0.;trunc_len/5]

        #NOTE: theta terms remain unchanged
        #2
        #Note we get rid of these terms
        # gradC[n, 1:3] = 2 * (x[1:3] - fin_first)
        # gradC[n+1,3n-2:3n] = 2 * (x[3n-2:3n] - fin_last)

        #theta ==> changed
        # gradC[n+2, 3(n-2)+1] = 2 * (theta_proj[1] - 0.0)
        # gradC[n+3, end] = 2 * (theta_proj[end] - total_theta)

        #solving for lagrange multipliers
        gradC_t = transpose(gradC)
        G = gradC * M_inv * gradC_t

        # println("G: ", G)
        #note: G is sparse. Not quite banded (note staggered rows)
        # fin_det = det(G)
        # pinv_fin = inv(G)
        # println("pinv_fin: ", pinv_fin)

        δλ = IterativeSolvers.lsmr(G, C, atol = btol = 10^-12)
        δλ_next = δλ / dt^2
        δx_next = -dt^2 * (M_inv * gradC_t * δλ_next)

        #println("this is delta: ", δx_next)
        #x_next_fp is vec
        x_next_fp_pos = x_fin[4:3n-3] + δx_next[1:3n-6]
        x_next_fp_theta = x_fin[3n+1:end] + δx_next[3n-5:end]

        x_next_fp = zeros(4n)
        x_next_fp[4:3n-3] = x_next_fp_pos
        x_next_fp[3n+1:end] = x_next_fp_theta

        x_next_fp[1:3] = x_fin[1:3]
        x_next_fp[3n-2:3n] = x_fin[3n-2:3n]

        # println("x_next_fp: ", x_next_fp)

        #______________re_init___________________
        #updating x, edges, C
        x_fin = x_next_fp
        x_arr = reshape(x_fin[1:3n], (3, n))
        x = x_fin[1:3n]

        theta_proj = x_fin[3n+1:end]
        edges = x_arr[:, 2:end] - x_arr[:, 1:end-1] #defining edges

        M = zeros(4n - 6, 4n - 6)
        mass_dens = 1.0
        rad_edge = 1.0

        #SWITCH: MASS MATRIX
        # mom_of_i = ((mass_dens * rad_edge^3 * 2*pi *l0)/3)I
        mass = 1. * I
        M[1:3n-6, 1:3n-6] = m_k * Matrix(mass, 3n - 6, 3n - 6)

        #moment of inertia defined on edges, cylindrical rod
        #this is assuming i know how to integrate
        M[3n-5:end, 3n-5:end] = Matrix(mom_of_i, n, n) #(mass_dens * rad_edge^3 * 2*pi)/3 *

        #computing inverse
        #check if faster way to do this
        M_inv = inv(M)

        C = constraint_func(
            x,
            theta_proj,
            l0,
            fixed_ends,
            gen_t
        )
        C_abs = zeros(size(C)[1])

        for i = 1:size(C)[1]
            C_abs[i] = abs(C[i])
        end #for
        maxC = maximum(C_abs)


        vor_len = vor_length(x)
        io = open(txt_array[1], "a")
        writedlm(io, C)
        write(io, "* \n")
        close(io)

        io = open(txt_array[2], "a")
        writedlm(io, "* ", vor_len)
        close(io)

        itc += 1

        if itc >= cap
            println("%%%%%%%%%%%%%%%STUCK!!!!!!%%%%%%%%%%%%%%")
            cap = cap*2
        end #conditional

        # println("this is itc", itc)
    end #while


    """

    insert velocity update

    vvv ISSUE BELOW vvv
    """
    #h^2(∇C(x_j) M^-1 ∇C(x_j)^T)δλ_{j+1} = C(x_j)

    # package things use, and return the new state
    x = reshape(x, (3, n))

    e_fin_proj = LinearAlgebra.normalize!(x[:, 2] - x[:, 1])
    u_fin_proj = rotab(tangent0, e_fin_proj, u0)

    x_fin_proj = vec(x)
    u_fin_proj = vec(u_fin_proj)

    fin_proj = Vector{Float64}(undef, 4n + 3)
    fin_proj[1:3] = u_fin_proj
    fin_proj[4:3n+3] = x_fin_proj
    fin_proj[3n+4:end] = theta_proj

    #calculating whether rod buckles
    pos_delt = fin_proj[4:end] - init_state

    for i = 1:3n-3
        if abs(pos_delt[i]) > err_tol * (10)
            crit_buck_F = (gen_t * imp_F) * (-1)
            println("BUCKLE ALERT @ ", gen_t)
            # println("FORCE ", crit_buck_F)
        end #cond
    end #loop

    # crit_buck_F = gen_t * imp_F
    f_prop = x_fin_proj - x_pre_proj
    # println("this is proj len: ", f_prop)

    #writing to text file
    io = open(txt_array[1], "a")
    # write(io, maxC, "\n")
    # C = constraint(x)
    write(io, "______ \n")
    write(io, "______ \n")
    close(io)

    io = open(txt_array[2], "a")
    write(io, "fin: ")
    writedlm(io, vor_len)
    write(io, "______ \n")
    close(io)

    io = open(txt_array[3], "a")
    writedlm(io, f_prop)
    write(io, "________\n")
    close(io)

    return fin_proj #vcat(u4, q4),
end # function

function DP5()
end #function

# function timestep_stationary(
#     F!,
#     f::Array{Float64,1},
#     state::Array{Float64,1},
#     dt::Float64,
#     param,
#     txt_array,
#     err_tol,
#     fixed_ends,
#     constraint_func,
#     gen_t,
# )
#     n = Int(param[1])
#     l0 = param[2]
#     x, theta, u0 = state2vars(state, n)
#     init_state = state[4:end]
#
#     q_i = vcat(vec(x), theta)
#     tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])
#
#     #RK4
#     q0 = vcat(vec(x), theta)
#     tangent0 = LinearAlgebra.normalize!(q0[4:6] - q0[1:3])
#
#     F!(f, q0, u0, param)
#     k1 = f
#     q1 = (dt / 2) * f + q0
#     e1 = LinearAlgebra.normalize!(q1[4:6] - q1[1:3])
#     u1 = rotab(tangent0, e1, u0)
#
#     F!(f, q1, u1, param)
#     k2 = f
#     q2 = (dt / 2) * f + q1
#     e2 = LinearAlgebra.normalize!(q2[4:6] - q2[1:3])
#     u2 = rotab(tangent0, e2, u0)
#
#     F!(f, q2, u2, param)
#     k3 = f
#     q3 = (dt / 2) * f + q2
#     e3 = LinearAlgebra.normalize!(q3[4:6] - q3[1:3])
#     u3 = rotab(tangent0, e3, u0)
#
#     F!(f, q3, u3, param)
#     k4 = f
#     q4 = (dt) * f + q3
#     e4 = LinearAlgebra.normalize!(q4[4:6] - q4[1:3])
#     u4 = rotab(tangent0, e4, u0)
#
#     x_fin = q0 + (1 / 6 * dt * (k1 + 2k2 + 2k3 + k4))
#     e_fin = LinearAlgebra.normalize!(x_fin[4:6] - x_fin[1:3])
#     u_fin = rotab(tangent0, e_fin, u0)
#
#     # println("this is ufin, xfin pre-projection")
#     init = vcat(u_fin, x_fin)
#     # println(vcat(u_fin,x_fin))
#
#     """
#     CONSTRAINTS
#     """
#
#     j = 0
#
#     #lumped model
#     #init mass
#     ℳ = n #total scalar mass
#     m_k = ℳ / n #assuming uniform rod
#
#     x = reshape(x_fin[1:3n], (3, n)) #reshape x, twist last row
#     theta_proj = x_fin[3n+1:end]
#     # println("theta_proj: ", theta_proj)
#     edges = x[:, 2:end] - x[:, 1:end-1] #defining edges
#
#     #mass matrix: uniform, lumped rod, CHECK THISß
#
#     # #init generalized mass matrix
#     M = zeros(4n, 4n)
#     mass_dens = 1.0
#     rad_edge = 1.0
#
#     M[1:3n, 1:3n] = m_k * Matrix(1.0I, 3n, 3n)
#
#     #moment of inertia defined on edges, cylindrical rod
#     #this is assuming i know how to integrate
#     M[3n+1:4n, 3n+1:4n] = Matrix(1.0I, n, n) #(mass_dens * rad_edge^3 * 2*pi)/3 *
#
#     #computing inverse
#     #check if faster way to do this
#     M_inv = inv(M)
#
#     C = constraint_func(x, theta_proj, l0, fixed_ends[1], fixed_ends[2], gen_t)
#     C_abs = zeros(size(C)[1])
#
#     for i = 1:size(C)[1]
#         C_abs[i] = abs(C[i])
#     end #for
#     maxC = maximum(C_abs)
#
#     vor_len = vor_length(x)
#
#     io = open(txt_array[1], "a")
#     writedlm(io, C)
#     write(io, "* \n")
#     close(io)
#
#     io = open(txt_array[2], "a")
#     writedlm(io, "* ", vor_len)
#     close(io)
#
#     iteration = 1
#
#     # while maxC >= err_tol || maxC <= -err_tol
#     while maxC >= err_tol
#         #vectorize
#         x = vec(x)
#
#         #initializing gradC
#         #n-1 ==> pos
#         #2 ==> boundary
#         #2 ==> boundary theta
#
#         gradC = zeros(n - 1 + 2 + 2, 4 * n)
#
#         #calculating gradient, must be changed depending on constraint
#         #4:n+2
#
#         for r = 1:n-1
#             gradC[r, 3*(r-1)+1:3*(r-1)+3] = -2 * edges[:, r]
#             gradC[r, 3*(r-1)+4:3*(r-1)+6] = 2 * edges[:, r]
#         end #for
#
#         # for r = 1:n
#         #     gradC[r, 3*(r-4)+1:3*(r-4)+3] = -2 * edges[:,r-3]
#         #     gradC[r, 3*(r-4)+4:3*(r-4)+6] = 2 * edges[:,r-3]
#         # end #for
#
#         #2
#         gradC[n, 1:3] = 2 * (x[1:3] - fixed_ends[1])
#         gradC[n+1, 3n-2:3n] = 2 * (x[3n-2:3n] - fixed_ends[2])
#
#         #theta
#         gradC[n+2, 3n+1] = 2*(theta_proj[1] - 0.)
#         gradC[n+3, end] = 2*(theta_proj[end] - 0.)
#
#         gradC_t = transpose(gradC)
#         G = gradC * M_inv * gradC_t
#
#         # println("G: ", G)
#         #note: G is sparse. Not quite banded (note staggered rows)
#         fin_det = det(G)
#
#         δλ = IterativeSolvers.lsmr(G, C, atol = btol = 10^-9)
#         δλ_next = δλ / dt^2
#         δx_next = -dt^2 * (M_inv * gradC_t * δλ_next)
#
#         #x_next_fp is vec
#         # δx_next[1:3] = zeros(3)
#         # δx_next[end-2:end] = zeros(3)
#         x_next_fp = x_fin + δx_next
#         j += 1
#
#         iteration += 1
#         #______________re_init___________________
#         #updating x, edges, C
#         x_fin = x_next_fp
#         x_arr = reshape(x_fin[1:3n], (3, n))
#         x = x_fin[1:3n]
#         theta_proj = x_fin[3n+1:end]
#
#         # x_arr[:,1] = fixed_ends[1] + [(gen_t * dt), 0, 0]
#         # x_arr[:,end] = fixed_ends[2] - [(gen_t * dt), 0, 0]
#
#         edges = x_arr[:, 2:end] - x_arr[:, 1:end-1] #defining edges
#
#         M[1:3n, 1:3n] = m_k * Matrix(1.0I, 3n, 3n)
#
#         #moment of inertia defined on edges, cylindrical rod
#         #this is assuming i know how to integrate
#         M[3n+1:4n, 3n+1:4n] = Matrix(1.0I, n, n) #(mass_dens * rad_edge^3 * 2pi)/3 *
#
#         #computing inverse
#         M_inv = inv(M)
#
#         #initializing C
#         C = constraint_func(
#             x,
#             theta_proj,
#             l0,
#             fixed_ends[1],
#             fixed_ends[2],
#             gen_t,
#         )
#         C_abs = zeros(size(C)[1])
#
#         for i = 1:size(C)[1]
#             C_abs[i] = abs(C[i])
#         end #for
#
#         maxC = maximum(C_abs)
#
#         #initializing vor length
#         vor_len = vor_length(x)
#
#         io = open(txt_array[1], "a")
#         writedlm(io, C)
#         write(io, "* \n")
#         close(io)
#
#         io = open(txt_array[2], "a")
#         writedlm(io, "* ", vor_len)
#         close(io)
#
#     end #while
#
#     """
#
#     insert velocity update
#
#     vvv ISSUE BELOW vvv
#     """
#     #h^2(∇C(x_j) M^-1 ∇C(x_j)^T)δλ_{j+1} = C(x_j)
#
#     # package things use, and return the new state
#     x = reshape(x, (3, n))
#
#     e_fin_proj = LinearAlgebra.normalize!(x[:, 2] - x[:, 1])
#     u_fin_proj = rotab(tangent0, e_fin_proj, u0)
#
#     x_fin_proj = vec(x)
#     u_fin_proj = vec(u_fin_proj)
#
#     fin_proj = Vector{Float64}(undef, 4n + 3)
#     fin_proj[1:3] = u_fin_proj
#     fin_proj[4:3n+3] = x_fin_proj
#     fin_proj[3n+4:end] = theta_proj
#
#     #calculating projection vector
#     f_prop = x_fin_proj - x_fin
#
#     #writing to text file
#     io = open(txt_array[1], "a")
#     write(io, "______ \n")
#     write(io, "______ \n")
#     close(io)
#
#     io = open(txt_array[2], "a")
#     write(io, "fin: ")
#     writedlm(io, vor_len)
#     write(io, "______ \n")
#     close(io)
#
#     io = open(txt_array[3], "a")
#     writedlm(io, f_prop)
#     write(io, "________\n")
#     close(io)
#
#     return fin_proj #vcat(u4, q4),
# end # function

"""
misc. helper functions
    twist_color: maps theta to color on cyclical color scale
"""

function twist_color(inp_theta)
    fin_col = zeros(size(inp_theta))

    for i = 1:size(inp_theta)[1]
        col = mod(inp_theta[i], 2 * pi)
        col = col / (2 * pi)
        fin_col[i] = col
    end #for

    # println("fin_col: ", fin_col)
    return fin_col
end #function

"""
runsim functions
    runsim_axis
    runsim_stationary
"""

# function runsim_stationary(
#     title,
#     dims,
#     err_tol,
#     tspan,
#     n_t,
#     N,
#     l0,
#     pos_0,
#     theta_0,
#     u_0,
#     constraint_type,
#     limits,
# )
#     println("%%%%%%%%%%%%", title, "%%%%%%%%%%%%%%%")
#
#     #initializing text files
#     max_txt = string(title, "_maxc", ".txt")
#     vor_txt = string(title, "_vor", ".txt")
#     sum_f = string(title, "_sum_f", ".txt")
#
#     txt_array = [max_txt, vor_txt, sum_f]
#
#     #re-initializing files
#     io = open(max_txt, "a")
#     close(io)
#     rm(max_txt)
#
#     io = open(vor_txt, "a")
#     close(io)
#     rm(vor_txt)
#
#     io = open(sum_f, "a")
#     close(io)
#     rm(sum_f)
#
#     #opening files
#     io = open(max_txt, "a")
#     io = open(vor_txt, "a")
#     io = open(sum_f, "a")
#
#     # setting txt_switch
#     txt_switch = title
#
#     println("%%%%%%%%%%%% start of calcs ==> change below %%%%%%%%%%%%%%%")
#     #initializing sim
#     dt = (tspan[2] - tspan[1]) / n_t
#     param = [N, l0]  #parameter vector
#
#     state_0 = vars2state(pos_0, theta_0, u_0)
#     x, theta, u_0 = state2vars(state_0, N)
#     println("x, theta, u_0: ", x, theta, u_0)
#
#     f = zeros(Float64, 4 * N)
#
#     plt = plot(aspect_ratio = :equal)
#     xlims!(limits[1])
#     ylims!(limits[2])
#     zlims!(limits[3])
#     state = state_0[:]
#     x_cur = reshape(state[4:3*(N+1)], (3, N))
#     x_1, x_2, x_3 = x_cur[1, :], x_cur[2, :], x_cur[3, :]
#     twist_weights = twist_color(theta_0)
#
#     # println(twist_weights)
#     for i = 1:N-1
#         # scatter!(plt,  x_cur[1,i:i+1],x_cur[2,i:i+1], x_cur[3,i:i+1],
#         #         label = legend = false) #
#         plot!(
#             plt,
#             x_cur[1, i:i+1],
#             x_cur[2, i:i+1],
#             #x_cur[3, i:i+1],
#             label = legend = false,
#             linewidth = 2,
#             linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
#         )
#     end #for loop #
#
#     title!(title)
#     display(plt)
#
#     force = 0.1 #stationary
#     function step!(gen_t)
#         state = timestep_stationary(
#             F!,
#             f,
#             state,
#             dt,
#             param,
#             txt_array,
#             err_tol,
#             [pos_0[:, 1], pos_0[:, end]],
#             constraint_type,
#             gen_t,
#         )
#         t += dt
#
#         x_cur = reshape(state[4:3*(N+1)], (3, N))
#         # println("theta =", state[3N+4:end])
#
#         theta_cur = state[3*(N+1)+1:end]
#         twist_weights = twist_color(theta_cur)
#
#         # println("time: ", t)
#         # # println("this is theta, timsetp 1: ", theta_cur)
#         # println("this is x, timsetp 1: ", x_cur)
#         # println("************** * * * BOINK *8*** *8**88   8 ")
#
#         plt = plot(aspect_ratio = :equal)
#         xlims!(limits[1])
#         ylims!(limits[2])
#         zlims!(limits[3])
#         for i = 1:N-1
#             # scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
#             #         label = legend = false) #
#             plot!(
#                 plt,
#                 x_cur[1, i:i+1],
#                 x_cur[2, i:i+1],
#                 #x_cur[3, i:i+1],
#                 label = legend = false,
#                 linewidth = 2,
#                 linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
#             )
#         end #for #
#
#         title!(title)
#     end #function
#
#     t = 0.0
#     @time anim = @animate for i = 1:n_t
#         step!(t)
#     end every 10
#     gif(anim, string(title, ".gif"), fps = 100)
#
#     # @time for i = 1:n_t
#     #     step!(t)
#     # end
#
#     display(plt)
#     println("u =", state[1:3])
#     png(title)
#
#     return state
# end #function

function runsim_axis(
    title,
    dims,
    err_tol,
    tspan,
    n_t,
    N,
    l0,
    pos_0,
    theta_0,
    u_0,
    constraint_type,
    limits,
    force
)
    println("%%%%%%%%%%%%", title, "%%%%%%%%%%%%%%%")

    #initializing text files
    max_txt = string(title, "_maxc", ".txt")
    vor_txt = string(title, "_vor", ".txt")
    sum_f = string(title, "_proj_arr", ".txt")

    txt_array = [max_txt, vor_txt, sum_f]

    #re-initializing files
    io = open(max_txt, "a")
    close(io)
    rm(max_txt)

    io = open(vor_txt, "a")
    close(io)
    rm(vor_txt)

    io = open(sum_f, "a")
    close(io)
    rm(sum_f)

    #opening files
    io = open(max_txt, "a")
    io = open(vor_txt, "a")
    io = open(sum_f, "a")

    # setting txt_switch
    txt_switch = title

    println("%%%%%%%%%%%% start of calcs ==> change below %%%%%%%%%%%%%%%")
    #initializing sim
    dt = (tspan[2] - tspan[1]) / n_t
    param = [N, l0]  #parameter vector

    state_0 = vars2state(pos_0, theta_0, u_0)
    x, theta, u_0 = state2vars(state_0, N)
    println("x, theta, u_0: ", x, theta, u_0)

    f = zeros(Float64, 4 * N)

    plt = plot(aspect_ratio = :equal)
    xlims!(limits[1])
    ylims!(limits[2])
    zlims!(limits[3])
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1, x_2, x_3 = x_cur[1, :], x_cur[2, :], x_cur[3, :]
    twist_weights = twist_color(theta_0)

    # xlims!(limits[2])
    # ylims!(limits[3])
    # # zlims!(limits[3])
    # # println(twist_weights)
    for i = 1:N-1
        # scatter!(plt,  x_cur[1,i:i+1],x_cur[2,i:i+1], x_cur[3,i:i+1],
        #         label = legend = false) #
        plot!(
            plt,
            x_cur[1, i:i+1],
            x_cur[2, i:i+1],
            # x_cur[2, i:i+1],
            # x_cur[3, i:i+1],
            #x_cur[3, i:i+1],
            label = legend = false,
            linewidth = 2,
            linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
        )
    end #for loop #

    title!(title)
    display(plt)

    # force = 0.001 #correct NO BUCKLE
    force = force
    function step!(gen_t)
        state = timestep_axis(
            F!,
            f,
            state,
            dt,
            param,
            txt_array,
            err_tol,
            [[pos_0[:, 1],theta_0[1]], [pos_0[:, end],theta_0[end]]],
            constraint_type,
            gen_t,
            -force,
        )
        t += dt

        x_cur = reshape(state[4:3*(N+1)], (3, N))
        # println("theta =", state[3N+4:end])

        theta_cur = state[3*(N+1)+1:end]
        twist_weights = twist_color(theta_cur)

        # println("time: ", t)
        # println("this is theta, timsetp 1: ", theta_cur)
        # println("this is x, timsetp 1: ", x_cur)
        # println("************** * * * BOINK *8*** *8**88   8 ")

        plt = plot(aspect_ratio = :equal)
        xlims!(limits[1])
        ylims!(limits[2])
        zlims!(limits[3])
        #
        # xlims!(limits[2])
        # ylims!(limits[3])
        # # zlims!(limits[3])
        for i = 1:N-1
            # scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
            #         label = legend = false) #
            plot!(
                plt,
                x_cur[1, i:i+1],
                x_cur[2, i:i+1],
                # x_cur[2, i:i+1],
                # x_cur[3, i:i+1],
                #x_cur[3, i:i+1],
                label = legend = false,
                linewidth = 2,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
            )
        end #for #

        title!(title)
    end #function

    t = 0.0
    @time anim = @animate for i = 1:n_t
        step!(t)
    end every 50
    gif(anim, string(title, ".gif"), fps = 100)

    # @time for i = 1:n_t
    #     step!(t)
    # end

    display(plt)
    println("u =", state[1:3])
    png(title)

    return state
end #function

"""
    constraints(pos)

function that generates constraints array as a function of position*
    *may add additional args,
"""

function constraint_twist(x_inp, theta_inp, vor1, start1, end1, t_cur)
    n = size(vec(x_inp))[1] / 3
    n = Int(n)

    # println(theta_inp)
    x_inp = reshape(x_inp, (3, n))
    edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]

    C = zeros(n - 1) #1 per edge, 2 for endpoints
    undeformed_config = vor1^2 #vor length, for now

    # CONSTRAINT 1: inextensibility
    for i = 1:n-1
        C[i] = dot(edges[:, i], edges[:, i]) - undeformed_config
        # println(C[i])
    end #loop

    C_fin = zeros(n - 1 + 4)

    C_fin[n] = dot(x_inp[:, 1] - start1, x_inp[:, 1] - start1) #note first vertex initializes @ origin
    C_fin[n+1] = dot(x_inp[:, end] - end1, x_inp[:, end] - end1)
    C_fin[n+2] = (0.0)^2
    C_fin[n+3] = (theta_inp[end] - 0.0)^2

    C_fin[1:n-1] = C

    C_fin = vec(C_fin)
    return C_fin
end #function
# (x_inp, theta_inp, vor1, start1, end1, t_cur)

function constraint_axis(x_inp, theta_inp, vor1, fixed_ends1, t_cur)
    n = size(vec(x_inp))[1] / 3
    n = Int(n)

    # println(theta_inp)
    x_inp = reshape(x_inp, (3, n))
    edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]

    C = zeros(n - 1) #1 per edge, 2 for endpoints
    undeformed_config = vor1^2 #vor length, for now

    # CONSTRAINT 1: inextensibility
    for i = 1:n-1
        C[i] = (dot(edges[:, i], edges[:, i]) - undeformed_config)
    end #loop

    C_fin = zeros(n - 1 + 4)

    v = vor1 #v = velocity
    total_theta = fixed_ends1[2][2]
    #C_fin[n] = dot(x_inp[:, 1] - start1, x_inp[:, 1] - start1) #note first vertex initializes @ origin
    # C_fin[n+1] = dot(x_inp[:,end] - (end1 - (v * t_cur) * [trunc_len;0.;trunc_len/5]), x_inp[:,end] - (end1 - (v * t_cur) * [trunc_len;0.;trunc_len/5]))
    # C_fin[n+2] = (theta_inp[1] - 0)^2
    # C_fin[n+3] = (theta_inp[end] - total_theta)^2

    C_fin[1:n-1] = C
    C_fin = vec(C_fin)

    # println("C_fin: ", C_fin)
    return C_fin
end #function

# function constraint_stat(x_inp, theta_inp, vor1, start1, end1, t_cur)
#     n = size(vec(x_inp))[1] / 3
#     n = Int(n)
#
#     # println(theta_inp)
#     x_inp = reshape(x_inp, (3, n))
#     edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]
#
#     C = zeros(n - 1) #1 per edge, 2 for endpoints
#     undeformed_config = vor1^2 #vor length, for now
#
#     # CONSTRAINT 1: inextensibility
#     for i = 1:n-1
#         C[i] = (dot(edges[:, i], edges[:, i]) - undeformed_config)
#     end #loop
#
#     C_fin = zeros(n - 1 + 4)
#     total_theta = 0.0
#
#     C_fin[n] = dot(x_inp[:, 1] - start1, x_inp[:, 1] - start1) #note first vertex initializes @ origin
#     C_fin[n+1] = dot(x_inp[:, end] - end1, x_inp[:, end] - end1)
#     C_fin[n+2] = (theta_inp[end] - 0)^2
#     C_fin[n+3] = (theta_inp[end] - total_theta)^2
#
#     C_fin[1:n-1] = C
#     C_fin = vec(C_fin)
#     # println("C_fin: ", C_fin)
#
#     return C_fin
# end #function

"""
init_ran() function
    randomly perturbs rod in x,y,z plane
    pos
    theta
    scale :: some 10^(-k) float
"""

function init_ran(pos, scale_pos)
    pert_pos = rand(Float64, size(pos))
    negation = rand([-1, 1], size(pos))
    pert_pos = negation .* pert_pos

    pert_pos = pert_pos * scale_pos
    pos += pert_pos
    return pos
end #function

"""
main() function
"""

function main()

    """
    initial buckling stage
    """

    buff = 1.0
    dim = 3
    # error_tolerance_C = 10^-5
    # timespan = (0.0, 0.01)
    # num_tstep = (2*10^4*0.01)*3
    # N_v = 21
    # vor = 1/5
    error_tolerance_C = 10^-4
    timespan = (0.0, 0.5)
    num_tstep = 0.5*0.3*10^4
    N_v = 20 + 1
    vor = 4/(N_v - 1)
    f = 5.
    name = string("twist_buckle")
    # name = string("twist_buckle, f=",
    #             string(f),", N=",
    #             string(N_v),", vor=",
    #             string(vor),", err_tol=",
    #             string(error_tolerance_C))

    # name = "twist_buckle"
    init_pos = zeros(3, N_v)
    init_theta = zeros(N_v)
    init_norm = [0.0, 1.0, 0.0]

    tot_tw = 0.
    for i = 1:N_v
        init_pos[:, i] = [vor * (i-1), 0.0, 0.0]
        init_theta[i] = (tot_tw/(N_v-1))*i
    end #loop

    # init_pos = init_ran(init_pos, 10^(-3))

    #PERTURB IN Y/Z
    pert_vec = [0., 10^-4, 0.]
    init_pos[:,N_v - 1] += pert_vec
    # init_pos[:,N_v - 2] += pert_vec
    # init_pos[:,N_v - 3] += pert_vec
    # init_pos[:,N_v - 3] += pert_vec
    #PERTURB IN Z
    # init_pos[:,4] += [0., 0., 10^(-3)]
    #init_theta = init_ran(init_theta, 10^(-6))

    println("INIT POS: ", init_pos)

    println("%%%%%8*%%***** %*%%%%%%%%*%%%%%   %*%*%%* % %*%*%*% ")
    # println("THIS IS PERT VEC ON X_4: ", pert_vec)

    lims = [
        (minimum(init_pos[1, :]) - buff, maximum(init_pos[1, :]) + buff),
        (minimum(init_pos[2, :]) - buff, maximum(init_pos[2, :]) + buff),
        (minimum(init_pos[3, :]) - buff, maximum(init_pos[3, :]) + buff),
    ]

    stat_state = runsim_axis(
        name,
        dim,
        error_tolerance_C,
        timespan,
        num_tstep,
        N_v,
        vor,
        init_pos,
        init_theta,
        init_norm,
        constraint_axis,
        lims,
        f
    )

    println("%%%%%8*%%***** %*%%%%%%%%*%%%%%   %*%*%%* % %*%*%*% ")
    # println("THIS IS PERT VEC ON X_4: ", pert_vec)
    println("%%%%%8*%%***** %*%%%%%%%%*%%%%%   %*%*%%* % %*%*%*% ")

    # """
    # evolution of rod in-place
    # """
    #
    # buff = 1.
    # name = "buckle_2_axial_STAT"
    # dim = 3
    # error_tolerance_C = 10^-3
    # timespan = (0.,1.)
    # num_tstep = 2*10^3
    # N_v = 5
    # vor = 1.
    #
    # init_pos, init_theta, init_norm = state2vars(stat_state, N_v)
    # #
    # # lims = [(minimum(init_pos[1,:]) - buff, maximum(init_pos[1,:]) + buff),
    # #         (minimum(init_pos[2,:]) - buff, maximum(init_pos[2,:]) + buff),
    # #         (minimum(init_pos[3,:]) - buff, maximum(init_pos[3,:]) + buff)]
    #
    # stat_state = runsim_stationary(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
    #         init_pos, init_theta, init_norm,
    #         constraint_stat, lims)

end #func

@time main()
