
## Structure definitions

abstract type aRod end # Abstract type for rods, closed or open

struct cRod <: aRod  # structure for closed rod
    n::Int32
    X::Array{Float64,2}
    midp::Array{Float64,2}
    edges::Array{Float64,2}
    voronoi::Array{Float64,1}
    theta::Array{Float64,1}
    frame::Array{Float64,3}
end # struct

struct oRod <: aRod # structure for open rod
    n::Int32
    X::Array{Float64,2}
    midp::Array{Float64,2}
    edges::Array{Float64,2}
    voronoi::Array{Float64,1}
    theta::Array{Float64,1}
    frame::Array{Float64,3}
end # struct

struct Measure
    pos::Array{Float64,2}
    thetam::Array{Float64,1}
end # struct

struct Simulation
    param::Dict{String, Any}
    rod::Rod
    measure::Measure
end # struct

## function definitions

function update_edges(rod::cRod)
    rod.edges .= rod.X[[2:end;1],:] .- rod.X
end # function

function update_edges(rod::oRod)
    rod.edges .= rod.X[2:end,:] .- rod.X
end # function

"""
    voronoi(cRod)
updates the voronoi domains' lengths of a closed Rod.
See voronoi(oRod) for equivalent function
"""
function voronoi(rod:cRod)
    rod.voronoi[1] = norm(rod.X[1,:]) + norm(rod.X[end,:])
    for i in 2:rod.n
        rod.voronoi[i] = norm(rod.X[i,:]) + norm(rod.X[i-1,:])
    end
end # function

"""
    voronoi(oRod)
updates the voronoi domains' lengths of an open Rod.
See voronoi(cRod) for equivalent function
"""
function voronoi(rod:oRod)
    for i in 1:(rod.n-1)
        rod.voronoi[i] = norm(rod.X[i+1,:]) + norm(rod.X[i,:])
    end
end # function



"""
    midpoints(rod)

returns array of midpoints of each edge  of the rod.
Output shape differs for closed or open rod
"""
function midpoints(rod::cRod)
    rod.midp .= (rod.X .+ rod.X[[2:end; 1],:]) ./ 2.
end # function

function midpoints(rod::oRod)
    rod.midp .= (rod.X[1:end-1,:] .+ rod.X[2:end,:]) ./ 2.
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
