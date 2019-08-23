
## Structure definitions

abstract type aRod end # Abstract type for rods, closed or open

mutable struct cRod <: aRod  # structure for closed rod
    n::Int32
    X::Array{Float64,2}
    midp::Array{Float64,2}
    theta::Array{Float64,1}
    frame::Array{Float64,2}
end # struct

mutable struct oRod <: aRod # structure for open rod
    n::Int32
    X::Array{Float64,2}
    midp::Array{Float64,2}
    theta::Array{Float64,1}
    frame::Array{Float64,2}
end # struct

struct Measure

end # struct

struct Simulation
    param
    rod::Rod
    measure::Measure
end # struct

## function definitions

"""
    midpoints(rod)

returns array of midpoints of each edge  of the rod.
Output shape differs for closed or open rod
"""
function midpoints(rod::cRod)
    rod.midp = (rod.X + rod.X[[2:end; 1],:]) / 2.
end # function

function midpoints(rod::oRod)
    rod.midp = (rod.X[1:end-1,:] + rod.X[2:end,:]) / 2.
end # function

"""
    dist(aRod)

returns matrix of distances between points of the rod
"""
function dist(rod::aRod)
    Xm = rod.midp
    len = size(Xm)[1]
    L2 = diag(Xm * Xm') * ones(Float64, 1, len)
    R = -2*(Xm*Xm') + (L2 + L2')
    return R .^ 0.5  #distance between i'th and j'th links
end # function

# collision handling

function iMatrix(rod::cRod)

end # function

function iMatrix(rod::oRod)

end # function
