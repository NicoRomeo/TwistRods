# Stochastic Dynamics of Elastic Rods
# builds on the Discrete Elastic rods framework

using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Statistics
using NLsolve
using Random


include("rodbase.jl")


function runstep(rod::aRod, parama::Array{T, 1}) where {T<:Number}
    # update Geometry and associated quantitites: midpoints, voronoi domains, ..
    #update_edges(rod)
    midpoints(rod)
    vDom(rod)
    kb(rod)

    # bending force
    E = bEnergy(rod, parama[1])
    print(E)
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
    print("\n___Rod1___\n")
    X = rand(5, 3)
    nTwist = 0.
    rod1 = cRod(X, nTwist)
    runstep(rod1, [1., 2., 4., 5])

    print("\n___Rod2___\n")
    X = zeros(Float64, 3, 3)
    for i in 1:3
        X[i,1] = cos(2. *pi*i/3.)
        X[i,2] = sin(2. *pi*i/3.)
    end
    nTwist = 0.
    rod2 = cRod(X, nTwist)
    print(rod2.kb)
    runstep(rod2, [1., 2., 4., 5])
    print(rod2.kb)
    print("\n___Rod3___\n")
    X = zeros(Float64, 10, 3)
    for i in 1:10
        X[i,1] = 8. *cos(2. *pi*i/10.)
        X[i,2] = 8. *sin(2. *pi*i/10.)
    end
    nTwist = 0.
    rod3 = cRod(X, nTwist)
    runstep(rod3, [1., 2., 4., 5])

    print("\n___end test___\n")

end
