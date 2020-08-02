##
## Test for the DiffEq solver
##
##

##
## ISSUES
# 1. NaN issue when taking gradients
# 2. Slow run-time for systems w/o stretch
# 3. Asymmetry in resolution of twist
##
## CODE
#
#

using Plots
#using Makie
#using Flux
using Zygote
using LinearAlgebra
using NLsolve
using DelimitedFiles
using ColorSchemes

include("newrod.jl")

## Functions to access states and back
# function state2vars(state::Array{Float64,1}, n::Integer)
# function vars2state(x::Array{Float64},theta::Array{Float64},u0::Array{Float64})
##

function energy_q(q, u0::Array{Float64,1}, p)
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

    u = normd(u0)
    v = cross(tangent, u)
    m1 = cos(theta[1]) * u + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u + cos(theta[1]) * v
    Ebend = 0.0
    Etwist = 0.0
    s = sqrt(edges'edges) - l0
    Estretch = s's
    i= 1
    while i < n - 1
    #for i = 1:(n-2)
        #edges_1 = X[:, i+2] - X[:, i+1]
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
        end
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
        i += 1
    end
    ell = 0.5 * sqrt(edges'edges)
    Etwist += m[end] .^ 2 / ell
    Ebend = 0.5 * Ebend
    Estretch = 0.5 * Estretch
    Etwist = 0.5 * Etwist
    return Ebend + Estretch #+ Etwist
end # function



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



function F!(f::Array{Float64,1}, q::Array{Float64,1}, u0, param)
    E = mu -> energy_q(mu, u0, param)
    f[:] = -1.0 * Zygote.gradient(E, q)[1]


end


"""
    timestep(args)

function that runs a single timestep using a RK2 time step.
    args: F force function
    state
    ref, r0, x0

ISSUES:
1.packaging x and x_fin, u_fin
2.adding in ref, r0, x0
3.speeding up inv

"""
function timestep(
    F!,
    f::Array{Float64,1},
    state::Array{Float64,1},
    dt::Float64,
    param,
    t
)
    # unpack state
    n = Int(param[1])
    l0 = param[2]
    x, theta, u0 = state2vars(state, n)
    q_i = vcat(vec(x), theta)
    tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])

    # Compute explicit initial guess for X, theta at t+dt

    #RK 2
    F!(f, q_i, u0, param)

    q_g = (dt / 2) * f + q_i
    e1 = LinearAlgebra.normalize!(q_g[4:6] - q_g[1:3])
    u1 = rotab(tangent0, e1, u0)

    F!(f, q_g, u1, param)

    q = dt * f + q_i
    e1 = LinearAlgebra.normalize!(q[4:6] - q[1:3])
    u1 = rotab(tangent0, e1, u0)

    #RK4
    q0 = vcat(vec(x), theta)
    tan0 = LinearAlgebra.normalize!(q0[4:6] - q0[1:3])

    F!(f, q0, u0, param)
    k1 = f
    q1 = (dt / 2) * f + q0
    e1 = LinearAlgebra.normalize!(q1[4:6] - q1[1:3])
    # u1 = rotab(tangent0, e1, u0)

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

    x_fin = q_i + (1/6 * dt * (k1 + 2k2 + 2k3 + k4))
    e_fin = LinearAlgebra.normalize!(x_fin[4:6] - x_fin[1:3])
    u_fin = rotab(tangent0, e_fin, u0)

    return vcat(u_fin, x_fin) #vcat(u4, q4),
end # function

"""
Maps inp_theta to float between 0., 1.
"""
function twist_color(inp_theta)
    fin_col = zeros(size(inp_theta))

    for i=1:size(inp_theta)[1]
        col =  mod(inp_theta[i], 2*pi)
        col = col / (2*pi)
        fin_col[i] = col
    end #for

    return fin_col
end #function

function main()
    per_col = get(colorschemes[:cyclic_mygbm_30_95_c78_n256_s25],
                    range(1.0,stop = 256.0), :clamp)

    println("%%%%%%%%%%%%%%%%%%% Twist, straight %%%%%%%%%%%%%%%%%%%")

    tspan = (0.0, 5.0)
    n_t = 500
    dt = (tspan[2] - tspan[1]) / n_t
    #param = [3, 1]
    N = 5
    l0 = 1
    param = [N, l0]  #parameter vector
    pos_0 = permutedims(
        [0.0 0.0 0.0; 0.0 1.0 0.0; 1.0 1.0 0.0; 4.0 1.0 2.0],
        (2, 1),
    )
    pos_0 = permutedims([0.0 0.0 0.0; 1.0 0.0 0.0; 2.0 0.0 0.0;
                        3.0 0.0 0.0; 4.0 0.0 0.0], (2, 1))

    pos_0 = permutedims([0.0 0.0 0.0; 1.0 0.0 0.0; 2.0 0.0 0.0;
                        3.0 0.0 0.0; 4.0 0.0 0.0], (2, 1))

    #straight line
    #pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0], (2, 1))

    println("this is pos_0: ")
    println(pos_0)

    #pos_0 = [0.0 0.0 0.0; 0.0 1.0 4.0; 0.0 0.0 0.0]

    theta_0 = [0., 1.0, 2.0, 3., 4.]
    # theta_0 = [0.0, 0.0, 0.0]
    u_0 = [0.0, 1.0, 0.0]
    # u_0 = [0.0, 0.0, 1.0]
    # u_0 = [0.0, 1/sqrt(2), 1/sqrt(2)]

    state_0 = vars2state(pos_0, theta_0, u_0)

    println("state 0")
    println(state_0)
    x, theta, u_0 = state2vars(state_0, N)
    println("x, theta, u_0")
    println(x)
    println(theta)
    println(u_0)

    f = zeros(Float64, 4 * N)

    plt = plot(1, xlim = (-1, 5), ylim = (-1, 5))
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1 = x_cur[1, :]
    x_2 = x_cur[2, :]

    twist_weights = twist_color(theta_0)
    fin_twist_weights = zeros(N-1)

    for i = 1:N-1
        fin_twist_weights[i] = (twist_weights[i] + twist_weights[i+1])  / 2
    end #loop

    for i = 1:N-1
        scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false)
        plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
                aspect_ratio=:equal)
    end #for loop


    display(plt)

    for i = 1:n_t
        state = timestep(F!, f, state, dt, param, i)

        x_cur = reshape(state[4:3*(N+1)], (3, N))
        # println("theta =", state[3N+4:end])

        x_1 = x_cur[1, :]
        x_2 = x_cur[2, :]

        theta_cur = state[3*(N+1) + 1:end]
        twist_weights = twist_color(theta_cur)

        for i = 1:N-1
            fin_twist_weights[i] = (twist_weights[i] + twist_weights[i+1]) / 2
        end #loop

        for i = 1:N-1
            scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                    label = legend = false)
            plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                    label = legend = false,
                    linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
                    aspect_ratio=:equal)
        end #for
    end #for

    println(f)
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f) = ", sum(f))
    display(plt)

    println("u =", state[1:3])

    png("test_rod_twist_straight")

    println("%%%%%%%%%%%%%%%%%%% bend w/o twist %%%%%%%%%%%%%%%%%%%")

    tspan = (0.0, 10.0)
    n_t = 1000
    dt = (tspan[2] - tspan[1]) / n_t
    #param = [3, 1]
    N = 3
    l0 = 1
    param = [N, l0]  #parameter vector
    pos_0 = permutedims(
        [0.0 0.0 0.0; 0.0 1.0 0.0; 1.0 1.0 0.0; 4.0 1.0 2.0],
        (2, 1),
    )
    pos_0 = permutedims([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 0.0], (2, 1))

    #straight line
    #pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0], (2, 1))

    println("this is pos_0: ")
    println(pos_0)

    #pos_0 = [0.0 0.0 0.0; 0.0 1.0 4.0; 0.0 0.0 0.0]

    theta_0 = [0.0, 0.0, 0.0]
    # theta_0 = [0.0, 0.0, 0.0]
    u_0 = [0.0, 0.0, 1.0]
    u_0 = [0.0, 0.0, -1.0]

    state_0 = vars2state(pos_0, theta_0, u_0)

    println("state 0")
    println(state_0)
    x, theta, u_0 = state2vars(state_0, N)
    println("x, theta, u_0")
    println(x)
    println(theta)
    println(u_0)

    f = zeros(Float64, 4 * N)

    plt = plot(1, xlim = (-1, 2), ylim = (-1, 2))
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1 = x_cur[1, :]
    x_2 = x_cur[2, :]


    twist_weights = twist_color(theta_0)
    fin_twist_weights = zeros(N-1)

    for i = 1:N-1
        fin_twist_weights[i] = (twist_weights[i] + twist_weights[i+1])  / 2
    end #loop

    for i = 1:N-1
        scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false)
        plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
                aspect_ratio=:equal)
    end #for loop

    display(plt)

    for i = 1:n_t
        state = timestep(F!, f, state, dt, param, i)

        x_cur = reshape(state[4:3*(N+1)], (3, N))
        # println("theta =", state[3N+4:end])
        theta_cur = state[3*(N+1) + 1:end]
        twist_weights = twist_color(theta_cur)

        for i = 1:N-1
            fin_twist_weights[i] = (twist_weights[i] + twist_weights[i+1]) / 2
        end #loop

        for i = 1:N-1
            scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                    label = legend = false)
            plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                    label = legend = false,
                    linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
                    aspect_ratio=:equal)
        end #for
    end #for

    println(f)
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f) = ", sum(f))
    display(plt)

    println("u =", state[1:3])

    png("test_rod_bend_notwist")

    println("%%%%%%%%%%%%%%%%%%% bend w/ twist %%%%%%%%%%%%%%%%%%%")

    tspan = (0.0, 0.1)
    n_t = 1000
    dt = (tspan[2] - tspan[1]) / n_t
    #param = [3, 1]
    N = 3
    l0 = 1
    param = [N, l0]  #parameter vector
    pos_0 = permutedims(
        [0.0 0.0 0.0; 0.0 1.0 0.0; 1.0 1.0 0.0; 4.0 1.0 2.0],
        (2, 1),
    )
    pos_0 = permutedims([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 0.0], (2, 1))
    pos_0 = permutedims([0.0 1.0 0.0; 0.0 0.0 0.0; 1.0 0.0 0.0], (2, 1))

    #straight line
    #pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0], (2, 1))

    println("this is pos_0: ")
    println(pos_0)

    #pos_0 = [0.0 0.0 0.0; 0.0 1.0 4.0; 0.0 0.0 0.0]

    theta_0 = [0.0, 1., 2.]
    # theta_0 = [0.0, 0.0, 0.0]
    u_0 = [0.0, 0.0, 1.0]
    u_0 = [0.0, 0.0, -1.0]

    state_0 = vars2state(pos_0, theta_0, u_0)

    println("state 0")
    println(state_0)
    x, theta, u_0 = state2vars(state_0, N)
    println("x, theta, u_0")
    println(x)
    println(theta)
    println(u_0)

    f = zeros(Float64, 4 * N)

    plt = plot(1, xlim = (-1, 2), ylim = (-1, 2))
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1 = x_cur[1, :]
    x_2 = x_cur[2, :]
    twist_weights = twist_color(theta_0)
    fin_twist_weights = zeros(N-1)

    for i = 1:N-1
        fin_twist_weights[i] = (twist_weights[i] + twist_weights[i+1])  / 2
    end #loop

    for i = 1:N-1
        scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false)
        plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
                aspect_ratio=:equal)
    end #for loop

    display(plt)

    for i = 1:n_t
        state = timestep(F!, f, state, dt, param, i)

        x_cur = reshape(state[4:3*(N+1)], (3, N))
        # println("theta =", state[3N+4:end])
        theta_cur = state[3*(N+1) + 1:end]
        twist_weights = twist_color(theta_cur)

        for i = 1:N-1
            fin_twist_weights[i] = (twist_weights[i] + twist_weights[i+1]) / 2
        end #loop

        for i = 1:N-1
            scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                    label = legend = false)
            plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                    label = legend = false,
                    linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]],
                    aspect_ratio=:equal)
        end #for

        println("this is theta", theta_cur)
    end #for

    println(f)
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f) = ", sum(f))
    display(plt)

    println("u =", state[1:3])

    png("test_rod_bent_twist")

    println("%%%%%%%%%%%%%%%%%%% Circle, no twist %%%%%%%%%%%%%%%%%%%")

    X = zeros(Float64, 3, 10)
    for i = 1:10
        X[1, i] = 10 * cos(-2.0 * pi * (i - 1) / 10)
        X[2, i] = 10 * sin(-2.0 * pi * (i - 1) / 10)
    end #for loop
    N = 10

    rad = 10
    #l0 = sqrt(2*rad^2*(1-cos(2*pi/10)))

    theta_0 = zeros(Float64, N)
    e_0 = normalize(X[:, 2] - X[:, 1])
    l_i = X[:, 2] - X[:, 1]
    l0 = sqrt(l_i[1]^2 + l_i[2]^2)
    # l0 = 1
    param = [N, l0]

    println("e_0 norm ", e_0'e_0 )

    #change below ! fix
    u_0 = [e_0[2], -e_0[1], 0.0]
    # u_0 = [0., 0., 1.]
    state_0 = vars2state(X, theta_0, u_0)

    # tspan = (0.0, 500.0)
    #BELOW:
    # tspan = (0.0, 10^4 * 3)
    # n_t = 10^4 * 3

    tspan = (0.0, 1.)
    n_t = 10^3
    dt = (tspan[2] - tspan[1]) / n_t

    f = zeros(Float64, 4 * N)

    plt = plot()
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1 = x_cur[1, :]
    x_2 = x_cur[2, :]
    scatter!(plt, x_1, x_2, label = legend = false)
    plot!(plt, x_1, x_2, aspect_ratio=:equal, label = legend = false)

    display(plt)

    for i = 1:n_t
        state = timestep(F!, f, state, dt, param, i)

        # println("this is state")
        # println(state)
        if i % 100 == 0
            x_cur = reshape(state[4:3*(N+1)], (3, N))
            # println("this is xcur")
            # println(x_cur)

            x_1 = x_cur[1, :]
            x_2 = x_cur[2, :]

            # println("this is x_1, x_2")
            # println(x_1, x_2)
            # if x_1[1] != NaN && x_2[1] != NaN
            #     scatter!(plt, x_1, x_2, legend = false)
            # end #cond

            scatter!(plt, x_1, x_2, legend = false)
            plot!(plt, x_1, x_2, legend = false, aspect_ratio=:equal,
                title = "Open circle without twist")
        end

    # println("u =", state[1:3] )
    end

    println("this is f")
    println(f)
    #check below
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f) = ", sum(f))
    display(plt)
    png("test_circle_1")

    println("%%%%%%%%%%%%%%%%%%% Circle, twist %%%%%%%%%%%%%%%%%%%")
    X = zeros(Float64, 3, 10)
    for i = 1:10
        X[1, i] = 10 * cos(-2.0 * pi * (i - 1) / 10)
        X[2, i] = 10 * sin(-2.0 * pi * (i - 1) / 10)
    end #for loop
    N = 10

    rad = 10
    #l0 = sqrt(2*rad^2*(1-cos(2*pi/10)))

    # theta_0 = zeros(Float64, N)
    theta_0 = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]
    e_0 = normalize(X[:, 2] - X[:, 1])
    l_i = X[:, 2] - X[:, 1]
    l0 = sqrt(l_i[1]^2 + l_i[2]^2)
    # l0 = 1
    param = [N, l0]

    println("e_0 norm ", e_0'e_0 )

    #change below ! fix
    u_0 = [e_0[2], -e_0[1], 0.0]
    # u_0 = [0., 0., 1.]
    state_0 = vars2state(X, theta_0, u_0)

    # tspan = (0.0, 500.0)
    #BELOW:
    # tspan = (0.0, 10^4 * 3)
    # n_t = 10^4 * 3

    tspan = (0.0, 1.)
    n_t = 10^3
    dt = (tspan[2] - tspan[1]) / n_t

    f = zeros(Float64, 4 * N)

    plt = plot()
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1 = x_cur[1, :]
    x_2 = x_cur[2, :]
    scatter!(plt, x_1, x_2, label = legend = false)
    plot!(plt, x_1, x_2, aspect_ratio=:equal, label = legend = false)

    display(plt)

    for i = 1:n_t
        state = timestep(F!, f, state, dt, param, i)

        # println("this is state")
        # println(state)
        if i % 100 == 0
            x_cur = reshape(state[4:3*(N+1)], (3, N))
            # println("this is xcur")
            # println(x_cur)

            x_1 = x_cur[1, :]
            x_2 = x_cur[2, :]
            # println("theta =", state[3N+4:end])

            # println("this is x_1, x_2")
            # println(x_1, x_2)
            # if x_1[1] != NaN && x_2[1] != NaN
            #     scatter!(plt, x_1, x_2, legend = false)
            # end #cond

            scatter!(plt, x_1, x_2, legend = false)
            plot!(plt, x_1, x_2, legend = false, aspect_ratio=:equal,
                title = "Open circle with twist")
        end

    # println("u =", state[1:3] )
    end

    println("this is f")
    println(f)
    #check below
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f) = ", sum(f))
    display(plt)
    png("test_circle_2_twist")

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
