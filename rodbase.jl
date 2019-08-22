using DifferentialEquations
using StaticArrays
using Plots
using LaTeXStrings
using Statistics
using NLsolve
import Random

include("rodsim.jl")

struct Rod
    n::Int32
    X::Array{Float64}
    theta::Array{Float64}
    frame::Array{Float64}
end # struct

struct Measure

end # struct

struct Simulation
    param
    rod::Rod
    measure::Measure
end
# Stochastic Dynamics of Elastic Rods
# builds on the Discrete Elastic rods framework





function main(args)




end # function

function dist(rod::Rod)
    midpoints = (rod.X .+ rod.X[[2:end; 1]]) ./2
end
