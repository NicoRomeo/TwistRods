# Stochastic Dynamics of Elastic Rods
# builds on the Discrete Elastic rods framework

using DifferentialEquations
using StaticArrays
using Statistics
using NLsolve
using Random


include("rodbase.jl")


function runstep(rod::aRod, parama::Array{Real, 1})
    # update midpoints, voronoi domains, ..

    # bending force
    bForce = bForce(rod, parama)

    # compute new frame, compute twisting force
    tForce = 0.

    # Extension

    # interaction
end

function runsim(sim::Simulation)
    rod = sim.rod
    param = sim.param

    param_array = Array{Any}(undef, 5)
    N = get(sim.param, "N")
    for i in range(N)
        runstep(rod, param_array)
    end
end




function main()
    X = rand(5, 3)
    nTwist = 0.
    rod = cRod(X, nTwist)
    #print(rod)

end
