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
    tForce =

    # Extension

    # interaction
end

function runsim(sim::Simulation)
    rod = sim.rod
    param = sim.param

    param_array = ...
    N = get(sim.param, "N")
    for i in range(N)
        runstep(rod, param)
    end
end




function main(args)


end
