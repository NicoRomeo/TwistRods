
## Structure definitions

abstract type aRod end # Abstract type for rods, closed or open

struct cRod <: aRod  # structure for closed rod
    n::Int32  # number of points; For cRod  number of points = number of edges
    X::Array{Float64,2} # Array of the position vectors of the nodes
    nTwist::Float64 # natural twist
    midp::Array{Float64,2}
    edges::Array{Float64,2} # Array of the edges vectors
    voronoi::Array{Float64,1} # Integrated lengths of each segment
    theta::Array{Float64,1} # array of twist angles
    frame::Array{Float64,3} # array of the reference frame
    function cRod(X::Array{Float64,2}, nTwist::Float64)
        n = size(X)[1]
        midp = (X + X[[2:end; 1],:]) / 2.
        edges = X[[2:end;1],:] - X
        voronoi = Array{Float64}(undef, n)
        voronoi[1] = norm(X[1,:]) + norm(X[end,:])
        for i in 2:n
            voronoi[i] = norm(X[i,:]) + norm(X[i-1,:])
        end
        normal = edges[[2:end;1],:] - edges
        binormal = Array{Float64}(undef, n, 3)
        for i in 1:n
            binormal[i,:] = cross(edges[i,:], normal[i,:])
        end
        frame = cat(edges, normal, binormal, dims=3)
        new(n, X, nTwist, midp, edges, voronoi, zeros(Float64, n), frame)
    end # function
end # struct

struct oRod <: aRod # structure for open rod
    n::Int32 # number of points; For oRod  no of points = no of edges - 1
    X::Array{Float64,2}
    nTwist::Float64 # Natural twist
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
    rod::aRod
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
function voronoi(rod::cRod)
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
function voronoi(rod::oRod)
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


function phi(rod::cRod)

end # function

function phi(rod::oRod)

end # function


###### collision handling ####
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






function iMatrix(rod::cRod)

end # function

function iMatrix(rod::oRod)

end # function
