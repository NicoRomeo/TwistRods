
using DifferentialEquations
using Flux
using LinearAlgebra

##########
# Draft code for non-linear implicit euler solver
#
#
#
##
n = 10

pos = zeros(4, n)
pos = rand(4, n)

function initialize()


end # func

function energy(pos::Matrix, p, t)
    A = ones(4,4)
    return dot(pos, A * pos)
end

function force(pos, param, t)
    f = x -> energy(x, param, t)
    return -1 .* Flux.gradient(f, pos)
end


g(u,p,t) = 1.  # noise function

tspan = (0.0, 1.)

pos_0 = rand(4, n)
param = [1, 1, 2]  #parameter vector
prob = SDEProblem(force, g, pos_0, tspan, param)
