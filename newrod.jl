
using DifferentialEquations
using Flux
using LinearAlgebra

##########
# Draft code for non-linear implicit euler solver
#
#
#
##
#n = 10

#pos = zeros(4, n)
#pos = rand(4, n)

function initialize()


end # func

function normd(x)
    return x/sqrt(x'x)
end

function energy(X, theta, u0, p, t)
    n = p[1]
    B = [1 0; 0 1]

    edges = diff(X, dims=2)
    m = diff(theta)

    tangent = zeros(Float64, (3, n-1))

    for i in 1:(n-1)
        tangent[:,i] = normalize(edges[:,i])
    end

    kb = zeros(Float64, (3, n-2))
    phi = zeros(Float64, n-2)
    for i in 1:(n-2)
        kb[:,i] = 2 .* cross(tangent[:,i], tangent[:,i+1])/(1 + dot(tangent[:,i], tangent[:,i+1]))
        phi[i] = 2*atan(norm(kb[:,i])/2)
    end

    # Bishop frame
    u = zeros(Float64, (3, n-1))
    u[:,1] = normalize(u0)
    for i in 1:(n-2)
        ax = normalize(kb[:,i])
        u[:,i+1] = dot(ax, u[:,i]) * ax + cos(phi[i]) *cross(cross(ax, u[:,i]), ax) + sin(phi[i]) * cross(ax, u[:,i])
    end
    v = zeros(Float64, (3, n-1))
    for i in 1:(n-1)
        v[:,i] = cross(tangent[:,i], u[:,i])
    end
    m1 = zeros(Float64, (3, n-1))
    m2 = zeros(Float64, (3, n-1))
    for i in 1:(n-1)
        m1[:,i] = cos.(theta[i]) * u[:,i] + sin.(theta[i]) * v[:,i]
        m2[:,i] = -sin.(theta[i]) * u[:,i] + cos.(theta[i]) * v[:,i]
    end

    kappa = zeros(Float64, (2, n-2))
    for i in 1:(n-2)
        kappa[:,i] = .5 * [dot(kb[:,i], m2[:,i] + m2[:,i+1]), -dot(kb[:,i], m1[:,i] + m1[:,i+1])]
    end

    ell = zeros(Float64, n)
    ell[1] = .5 * norm(edges[:,1])
    ell[n] = .5 * norm(edges[:,n-1])
    for i in 2:(n-1)
        ell[i] = .5 *(norm(edges[:,i-1]) + norm(edges[:,i]))
    end
    Ebend = 0
    for i in 1:(n-2)
        Ebend += kappa[:,i]' * B * kappa[:,i] / ell[i+1]
    end
    Ebend = .5 * Ebend

    Etwist = sum(m .^2 ./ ell[2:n-1])

    return Ebend + Etwist
end

function energydict(X::AbstractArray, theta::AbstractArray, u0::AbstractArray, p)
    n = p[1]
    l0 = p[2]
    B = [1 0; 0 1]

    # edges, tangent, kb, phi
    edges = Dict{Int32, Array{Float64,1}}()
    tangent = Dict{Int32, Array{Float64,1}}()
    kb = Dict{Int32, Array{Float64,1}}()
    phi = Dict{Int32, Float64}()

    # voronoi domain
    ell = Dict{Int32, Float64}()

    edges[1] = X[:,2] - X[:,1]
    tangent[1] = normd(edges[1])
    ell[1] = .5 * sqrt(edges[1]'edges[1])

    m = diff(theta, dims=1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    # Bishop frame
    u = Dict{Int32, Array{Float64,1}}()
    v = Dict{Int32, Array{Float64,1}}()
    # Material frame
    m1 = Dict{Int32, Array{Float64,1}}()
    m2 = Dict{Int32, Array{Float64,1}}()

    u[1] = normd(u0)
    v[1] = cross(tangent[1], u[1])
    m1[1] = cos(theta[1]) * u[1] + sin(theta[1]) * v[1]
    m2[1] = -sin(theta[1]) * u[1] + cos(theta[1]) * v[1]
    Ebend = 0.
    Etwist = 0.
    Estretch = (edges[1] .- l0)' * (edges[1] .- l0)
    for i in 1:(n-2)
        edges[i+1] = X[:,i+2] - X[:,i+1]
        tangent[i+1] = normd(edges[i+1])
        kb[i] = 2 .* cross(tangent[i], tangent[i+1])/(1 + tangent[i]'tangent[i+1])
        kbn = sqrt(kb[i]'kb[i])
        phi[i] = 2*atan(kbn/2)

        ax = kb[i]/kbn

        ell[i+1] = .5 * (sqrt(edges[i]'edges[i]) + sqrt(edges[i+1]'edges[i+1]))

        u[i+1] = dot(ax, u[i]) * ax + cos(phi[i]) *cross(cross(ax, u[i]), ax) + sin(phi[i]) * cross(ax, u[i])
        v[i+1] = cross(tangent[i+1], u[i+1])
        m1[i+1] = cos(theta[i+1]) * u[i+1] + sin(theta[i+1]) * v[i+1]
        m2[i+1] = -sin(theta[i+1]) * u[i+1] + cos(theta[i+1]) * v[i+1]
        k = .5 * [dot(kb[i], m2[i] + m2[i+1]), -dot(kb[i], m1[i] + m1[i+1])]
        #missing def of ell
        Ebend += k' * B * k / ell[i+1]
        Etwist += m[i] .^2 / ell[i+1]
        s = edges[i+1] .- l0
        Estretch += s's
    end
    Ebend = .5 * Ebend
    Estretch = .5 * Estretch
    return Ebend + Etwist + Estretch
end # function


function state2vars(state::Array{Float64, 2})
    x = state[1:3, 2:end]
    theta = state[4, 2:end-1]
    u0 = state[1:3, 1]
    return (x, theta, u0)
end

function vars2state(x::Array{Float64}, theta::Array{Float64}, u0::Array{Float64})
    h = vcat(u0, 0)
    t = vcat(theta, 0)
    xt = vcat(x, t')
    return hcat(h, xt)
end

function rotatea2b(a::Array{Float64,1},b::Array{Float64,1})
    return 2. * (a+b) * (a+b)'/((a+b)'*(a+b)) - [1 0 0; 0 1 0; 0 0 1]
end

function force(ds, state::Array{Float64, 2}, param, t)
    # unpack state variables
    pos = state[1:3, 2:end]
    theta = state[4, 2:end-1]
    u0 = state[1:3, 1]
    # define fucntions for energy/gradient
    Ex = x -> energydict(x, theta, u0, param)
    Et = t -> energydict(pos, t, u0, param)

    fx = -1. * Flux.gradient(Ex, pos)[1]
    ft = -1. * Flux.gradient(Et, theta)[1]

    # update u0...
    fu0 = fx[:,2] - fx[:,1]

    ds = vars2state(fx, ft, fu0)
end


g(u,p,t) = 0.  # noise function

tspan = (0.0, 3.)
param = [3, 1]
N = 3
l0 = 1
param = [N, l0]  #parameter vector
pos_0 = transpose([0. 0. 0.; 0. 1. 0.; 1. 1. 0.])
theta_0 = [0., 0.]
u_0 = [1., 0., 0.]

state_0 = vars2state(pos_0, theta_0, u_0)
# pos_0 =
#param = [1, 1, 2]  #parameter vector

#prob = SDEProblem(force, g, state_0, tspan, param)

MassMatrix = diagm(vcat(ones(Float64, 3), ones(Float64, 3*N), zeros(Float64, N-1)))

prob = ODEProblem(force, state_0, tspan, param)

sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

times = sol.t
plot(times, sol(times)[1:3,3])
