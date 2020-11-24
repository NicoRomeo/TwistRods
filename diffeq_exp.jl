using Plots, ColorSchemes, DelimitedFiles, BenchmarkTools, DataFrames
using IterativeSolvers, NLsolve, Zygote, LinearAlgebra, DifferentialEquations

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

"""
2. DiffEq.jl SIMULATION LOOP
"""

function ts!(du,u,p,t)
    n,l0 = Int(p[1]),p[2]
    dq = FF(n,l0,vcat(vec(u[2,:,1:n]),u[3,1,1:n]),u[1,1:3,1:3])
    du[2,:,1:n] = reshape(dq[1:3n],3,n)
    du[3,1,1:n] = dq[3n+1:end]
    t = normd(u[2,:,2] - u[2,:,1])
    du[1,1:3,1] = cross(cross(t,du[2,:,2] - du[2,:,1]),du[1,1:3,1])
    du[1,1:3,2] = cross(cross(t,du[2,:,2] - du[2,:,1]),du[1,1:3,2])
    du[1,1:3,3] = cross(cross(t,du[2,:,2] - du[2,:,1]),du[1,1:3,3])
    # println(u)
end #function

n = 5
l0 = 1.
q = [1. 0. 0. 2. 0. 0. 3. 0. 0. 4. 0. 0. 5. 0. 0. 0. 1. 2. 3. 4.]
mat_f = [0. 1. 0.;0. 0. 1.;1. 0. 0.]'
u0 = zeros(3,3,n)
# q = @view u[2,:,1:n]
# d = @view u[1,1:3,1:3]
u0[3,1,1:n] = q[3n+1:end]
u0[2,:,1:n] = q[1:3n]
u0[1,1:3,1:3] = mat_f

prob = ODEProblem(ts!,u0,(0.,0.5),[n,l0])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
