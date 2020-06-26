###
#  Time-parallel implicit solver for discrete elastic rods
#
#
####

using Plots
#using Makie
#using Flux
using Zygote
using LinearAlgebra
using NLsolve

include("newrod.jl")

function energy_timep(q, u, p)
    n = Int(p[1])
    l0 = p[2]
    B = [1 0; 0 1]

    # edges, tangent, kb, phi
    #X = reshape(q[1:3*n], (3,n))
    X = q[1:3*n]
    theta = q[3*n+1:end]

    #edges = X[:, 2] - X[:, 1]
    edges = q[4:6] - q[1:3]
    tangent = normd(edges)
    ell = 0.5 * sqrt(edges'edges)

    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    #u = normd(u0)
    v = cross(tangent, u[1:3])
    m1 = cos(theta[1]) * u[1:3] + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u[1:3] + cos(theta[1]) * v
    Ebend = 0.0
    Etwist = 0.0
    s = sqrt(edges'edges) - l0
    Estretch = s's
    # inner points
    #i = 2
    #while i < n
    for i = 2:(n-1)
        #edges_1 = X[:, i+2] - X[:, i+1]
        edges_1 = q[3*i+1:3*(i+1)] - q[3*(i-1)+1:3*i]
        u_c = u[3*(i-1)+1:3*i]

        tangent_1 = normd(edges_1)
        kb = 2 .* cross(tangent, tangent_1) / (1 + tangent'tangent_1)
        kbn = sqrt(kb'kb)
        ell = 0.5 * (sqrt(edges'edges) + sqrt(edges_1'edges_1))
        v = cross(tangent_1, u_c)
        m1_1 = cos(theta[i+1]) * u_c + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u_c + cos(theta[i+1]) * v
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
    end
    #final point
    ell = 0.5 * sqrt(edges'edges)
    Etwist += m[end] .^ 2 / ell
    Ebend = 0.5 * Ebend
    Estretch = 0.5 * Estretch
    Etwist = 0.5 * Etwist
    return Ebend + Etwist + 3* Estretch

end

function skewmat(a::Array{Float64})
    return [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
end # function

"""
    rotab(a, b, c)

rotation matrix sending unit vectors a to b, applied to c
"""
function rotab(a::Array{Float64,1}, b::Array{Float64,1}, c::Array{Float64,1})
    axb = cross(a, b)
    V = skewmat(axb)
    u = c + cross(axb, c) + V*V * c / (1 + dot(a, b))
    return u
end # function

"""
    F_impl(args)

Force for the implicit/time-parallel approach to integration
"""
function F_impl( q::Array{Float64,1}, u, param, dt)
    E = mu -> energy_q(mu, u, param)
    f = -1.0 * Zygote.gradient(E, q)[1]
    # update  u: euler-
    n = Int(param[1])
    u_update = zeros_like(u)
    q_n = q + dt*f
    for i = 1:(n-1)
        tan_0 = LinearAlgebra.normalize!(q[3*i+1:3*(i+1)] - q[3*(i-1)+1:3*i])
        e_fin = LinearAlgebra.normalize!(q_n[3*i+1:3*(i+1)] - q_n[3*(i-1)+1:3*i])
        u_updated[3*(i-1)+1:3*i] = rotab(tangent0, e_fin, u[3*(i-1)+1:3*i])
    return -1.0 * Zygote.gradient(E, q)[1], u_updated

end # function

"""
    timestep_impl(args)

Implicit integrator to use with the time-parallel approach
"""
function timestep_impl(
    F,
    f::Array{Float64,1},
    q::Array{Float64,1},
    u::Array{Float64,1},
    dt::Float64,
    param,
    t,
)
    n = Int(param[1])
    tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])
    #first guess by single Euler step
    E = mu -> energy_q(mu, u, param)
    f = -1.0 * Zygote.gradient(E, q)[1]
    q_guess = q + dt* f

    Res = zeros(q)
    function impl!(R, q_t, gamma)  # gamma is viscosity parameter
        E = mu -> energy_q(mu, u, param)
        f = -1.0 * Zygote.gradient(E, q_t)[1]
        return q_t - q_i + f/gamma
    end # function
    sres= nlsolve(impl!, q_guess)
    #



end # function

function test_impl()
    tspan = (0.0, 10.0)
    n_t = 1000
    dt = (tspan[2] - tspan[1]) / n_t
    N = 3
    l0 = 1
    param = [N, l0]
    pos_0 = permutedims([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 0.0], (2, 1))

    #straight line
    #pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0], (2, 1))

    println("this is pos_0: ")
    println(pos_0)

    #pos_0 = [0.0 0.0 0.0; 0.0 1.0 4.0; 0.0 0.0 0.0]

    theta_0 = [0.0, 0.0, 0.0]
    q_0 = vcat(vec(pos_0), theta_0)
    u_01 = [0.0, 1.0, 0.0]

    u_0 = zeros(3*(N-1))
    u_0[1:3] = u_0[:]
    # create initial material frame
    for i = 1:(N-2)
        #edges_1 = X[:, i+2] - X[:, i+1]
        edges_1 = q_0[3*(i+1)+1:3*(i+2)] - q_0[3*i+1:3*(i+1)]
        tangent_1 = normd(edges_1)
        kb = 2 .* cross(tangent, tangent_1) / (1 + tangent'tangent_1)
        kbn = sqrt(kb'kb)
        ell = 0.5 * (sqrt(edges'edges) + sqrt(edges_1'edges_1))
        if !isapprox(kbn, 0.0)
            phi = 2 * atan(dot(kb, kb / kbn) / 2)
            ax = kb / kbn
            u_0[3*(i+1)+1:3*(i+2)] =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end
