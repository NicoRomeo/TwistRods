##
## Test for the DiffEq solver
##
##


using Plots
#using Makie
#using Flux
using Zygote
using LinearAlgebra

include("newrod.jl")

## Functions to access states and back
# function state2vars(state::Array{Float64,1}, n::Integer)
# function vars2state(x::Array{Float64},theta::Array{Float64},u0::Array{Float64})
##

function energy_q(
    q,
    u0::Array{Float64,1},
    p
)
    n = p[1]
    l0 = p[2]
    B = [1 0; 0 1]

    # edges, tangent, kb, phi
    X = reshape(q[1:3*n], (3,n))
    theta = q[3*n+1:end]

    edges = X[:, 2] - X[:, 1]
    tangent = normd(edges)
    ell = 0.5 * sqrt(edges'edges)

    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    u = normd(u0)
    v = cross(tangent, u)
    m1 = cos(theta[1]) * u + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u + cos(theta[1]) * v
    Ebend = 0.0
    Etwist = 0.0
    s = edges'edges - l0
    Estretch = s's
    for i = 1:(n-2)
        edges_1 = X[:, i+2] - X[:, i+1]
        tangent_1 = normd(edges_1)
        kb = 2 .* cross(tangent, tangent_1) / (1 + tangent'tangent_1)
        kbn = sqrt(kb'kb)
        ell = 0.5 * (sqrt(edges'edges) + sqrt(edges_1'edges_1))
        phi = 2 * atan(kbn / 2)

        if !isapprox(kbn, 0.0)
            ax = kb / kbn
            u =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end
        v = cross(tangent_1, u)
        m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
        k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]

        Ebend += k' * B * k / ell
        Etwist += m[i] .^ 2 / ell
        s = edges_1'edges_1 .- l0
        Estretch += s*s
        # update for next vertex
        edges = edges_1
        tangent = tangent_1
        m1 = m1_1
        m2 = m2_1
    end
    ell = 0.5 * sqrt(edges'edges)
    Etwist += m[end] .^ 2 / ell
    Ebend = 0.5 * Ebend
    Estretch = 0.5 * Estretch
    Etwist = 0.5 * Etwist
    return Ebend + Etwist + Estretch
end # function

function skewmat(a::Array{Float64})
    return [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
end # function

"""
    rotab(a, b, c)

rotation matrix sending unit vectors a to b, aplied to c
"""
function rotab(a::Array{Float64,1}, b::Array{Float64,1}, c::Array{Float64,1})
    axb  = cross(a,b)
    V = skewmat(axb)
    u = c + cross(axb, c) + V^2 * c/(1+dot(a,b))
    return u
end # function


# function energy_q(q, u0::Array{Float64,1}, p)
#     n = p[1]
#     x = reshape(q[1:3*n], (3, n))
#     theta = q[3*n+1:end]
#     return energy_clean(x, theta, u0, p)
# end

function F!(f::Array{Float64,1}, q::Array{Float64,1}, u0, param)
    E = mu -> energy_q(mu, u0, param)
    f[:] = - 1.0 * Zygote.gradient(E, q)[1]
end


"""
    timestep(args)

function that runs a single timestep using a RK2 time step.
    args: F force function
    state
"""
function timestep(F!, f::Array{Float64,1}, state::Array{Float64,1}, dt::Float64, param, t)
    # unpack state
    n = param[1]
    x, theta, u0 = state2vars(state, n)
    q_i = vcat(vec(x), theta)
    tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])
    # Compute explicit initial guess for X, theta at t+dt

    F!(f, q_i, u0, param)

    q_g  = (dt/2) * f + q_i
    e1 = LinearAlgebra.normalize!(q_g[4:6] - q_g[1:3])
    u1 = rotab(tangent0, e1, u0)

    F!(f, q_g, u1, param)

    q = dt * f + q_i
    e1 = LinearAlgebra.normalize!(q[4:6] - q[1:3])
    u1 = rotab(tangent0, e1, u0)

    # package things use, and return the new state
    return vcat(u0, q)

end # function




function main()
    tspan = (0.0, 3.0)
    n_t = 200
    dt = (tspan[2] - tspan[1]) / n_t
    #param = [3, 1]
    N = 3
    l0 = 1
    param = [N, l0]  #parameter vector
    pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 1.0 1.0 0.0; 4.0 1.0 2.0], (2,1))
    pos_0 = permutedims([1. 0. 0.;0. 0. 0.;0. 1. 0.], (2,1))

    #straight line
    #pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0], (2, 1))

    println("this is pos_0: ")
    println(pos_0)

    #pos_0 = [0.0 0.0 0.0; 0.0 1.0 4.0; 0.0 0.0 0.0]

    theta_0 = [0.0, 1.0, 2.0]
    u_0 = [0., 1.0, 0.0]

    state_0 = vars2state(pos_0, theta_0, u_0)

    f = zeros(Float64, 4*N)

    plt = plot(1, xlim = (-1, 2),
            ylim = (-1, 2))
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1 = x_cur[1, :]
    x_2 = x_cur[2, :]
    scatter!(plt, x_1, x_2, label = legend = false)
    plot!(plt, x_1, x_2, label = legend = false)

    display(plt)

    for i = 1:n_t
        state = timestep(F!, f, state, dt, param, i)
        x_cur = reshape(state[4:3*(N+1)], (3, N))
        x_1 = x_cur[1, :]
        x_2 = x_cur[2, :]
        scatter!(plt, x_1, x_2, legend = false)
        plot!(plt, x_1, x_2, legend = false)
    end
    display(plt)
    png("test_rod_twist")
    # scene = Scene()
    #
    # state = state_0[:]
    # x_cur = reshape(state[4:3*(N+1)], (3, N))
    # x_1 = x_cur[1, :]
    # x_2 = x_cur[2, :]
    # scene = lines(x_cur[1:2,:], color = :blue)
    # scatter!(scene, x_cur[1:2,:], color = :blue, markersize = 0.1)
    #
    # timestep(F!, f, state, dt, param ,1)
    #
    # record(scene, "line_changing_colour.mp4", 1:n_t; framerate = 30) do i
    #     state = timestep(F!, f, state, dt, param, i)
    #     x_cur[] = reshape(state[4:3*(N+1)], (3, N))
    #     #x_1 = x_cur[1, :]
    #     # x_2 = x_cur[2, :]
    #     # lines!(scene, x_1, x_2)
    #     # scatter!(scene, x_1, x_2)
    # end every 5



end


main()