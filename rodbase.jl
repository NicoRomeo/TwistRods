
## Structure definitions

abstract type aRod end # Abstract type for rods, closed or open

struct cRod <: aRod  # structure for closed rod
    n::Int32  # number of points; For cRod  number of points = number of edges
    X::Array{Float64,2} # Array of the position vectors of the nodes
    nTwist::Float64 # natural twist
    len::Float64  # Length
    midp::Array{Float64,2}
    edges::Array{Float64,2} # Array of the edges vectors
    voronoi::Array{Float64,1} # Integrated lengths of each segment
    kb::Array{Float64,2}  # Discrete curvature binormal scalar function.
    theta::Array{Float64,1} # array of twist angles
    frame::Array{Float64,3} # array of the reference
    J::Matrix{Int8,2} #matrix that rotates vectors 90 degrees
    function cRod(X::Array{Float64,2}, nTwist::Float64)
        n = size(X)[1]
        midp = (X + X[[2:end; 1],:]) / 2.
        edges = X[[2:end;1],:] - X
        vor = Array{Float64}(undef, n)
        vor[1] = norm(edges[1,:]) + norm(edges[end,:])
        for i in 2:n
            vor[i] = norm(edges[i,:]) + norm(edges[i-1,:])
        end
        normal = edges[[2:end;1],:] - edges
        for i in 1:n
            normalize!(normal[i,:])
        end
        binormal = Array{Float64}(undef, n, 3)
        for i in 1:n
            binormal[i,:] = normalize!(cross(edges[i,:], normal[i,:]))
        end
        norm_edges = Array{Float64}(undef, n, 3)
        for i in 1:n
            norm_edges[i,:] = normalize(edges[i,:])
        end
        frame = cat(norm_edges, normal, binormal, dims=3)
        kb = Array{Float64}(undef, n, 3)
        len = sum([norm(edges[i,:]) for i in 1:n])

        #adding J (counterclockwise pi/2 rotation matrix)
        J = [0 -1;1 0]

        new(n, X, nTwist, len, midp, edges, vor, kb, zeros(Float64, n), frame, J)
    end # function
end # struct

struct oRod <: aRod # structure for open rod
    n::Int32 # number of points; For oRod  no of points = no of edges - 1
    X::Array{Float64,2}
    nTwist::Float64 # Natural twist
    midp::Array{Float64,2}
    edges::Array{Float64,2}
    voronoi::Array{Float64,1}
    kb::Array{Float64,1}  # Discrete curvature binormal scalar function.
    theta::Array{Float64,1}
    frame::Array{Float64,3}
    J::Matrix{Int8,2}
    function oRod(X::Array{Float64,2}, nTwist::Float64)
        n = size(X)[1]
        midp = (X[[1:end-1],:] + X[[2:end],:]) / 2.

        edges = X[[2:end],:] - X[[1:end-1],:] # only n - 1 edges
        altedges = append!([edges[1]],edges) #assigns first vertex first edge for binormal calculations

        vor = Array{Float64}(undef,n)

        """is this (below) right? The vor calcs for cRod don't
        make sense either
        """

        for i in 1:n
            vor[i] = norm(X[i,:]) + norm(X[i-1,:])
        end

        normal = edges[[2:end],:] - edges[[1:end-1],:]

        normal = append!([edges[1]],normal) #setting normal of first and last vertices
        normal = append!(normal,[edges[end]])

        for i in 1:n
            normalize!(normal[i,:])
        end

        binormal = Array{Float64}(undef,n-2,3)

        """
        not sure if assumptions made (below) are correct:

        -normal exists for all n vertices
        -however, each vertex doesn't have an edge (n vertices, n-1 edges)
        -therefore, I assigned both vertex 1 & 2 the same edge (in altedge) to compute the binormal
        -At the same time, I could assign both vertices n-1 and n the same edge
        -???
        """

        for i in 1:n
            binormal[i,:] = normalize!(cross(altedges[i,:],normal[i,:]))
        end

        norm_edges = Array{Float64}(undef,n-1,3)
        for i in 1:n-1
            norm_edges[i,:] = normalize(edges[i,:])
        end

        """
        (below) cat fxn doesn't work because norm_edges, normal, binormal
        don't all have the same dimension.

        how to resolve this?
        """

        frame = cat(norm_edges, normal, binormal, dims=3)
        kb = Array{Float64}(undef, n, 3)
        len = sum([norm(edges[i,:]) for i in 2:n])

        #adding J (counterclockwise pi/2 rotation matrix)
        J = [0 -1;1 0]
        new(n, X, nTwist, len, midp, edges, vor, kb, zeros(Float64, n), frame, J)

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
    rod.edges .= rod.X[2:end,:] .- rod.X[1:end-1,:]
end # function

"""
    vDom(cRod)
updates the voronoi domains' lengths of a closed Rod.
See voronoi(oRod) for equivalent function
"""
function vDom(rod::cRod)
    rod.voronoi[1] = norm(rod.edges[1,:]) + norm(rod.edges[end,:])
    for i in 2:rod.n
        rod.voronoi[i] = norm(rod.edges[i,:]) + norm(rod.edges[i-1,:])
    end
end # function

"""
    vDom(oRod)
updates the voronoi domains' lengths of an open Rod.
See voronoi(cRod) for equivalent function
"""
function vDom(rod::oRod)
    for i in 1:(rod.n-1)
        rod.voronoi[i] = norm(rod.edges[i+1,:]) + norm(rod.edges[i,:])
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
    kb(cRod)
Computes the discrete curvature binormal (kappa b) from Bergou et al.
"""
function kb(rod::cRod)
    rod.kb[1,:] = (2. * cross(rod.frame[end, 1, :], rod.frame[1,1,:])
            /(norm(rod.frame[end,1,:])*norm(rod.frame[1,1,:])
            + dot(rod.frame[end,1, :], rod.frame[1,1,:])))
    for i in 2:rod.n
        rod.kb[i,:] = (2. * cross(rod.frame[i-1,1, :], rod.frame[i,1,:])
                /(norm(rod.frame[i-1,1,:])*norm(rod.frame[i,1,:])
                + dot(rod.frame[i-1,1,:], rod.frame[i,1,:])))
    end
end # function

function kb(rod::oRod)
    for i in 2:rod.n
        rod.kb[i-1,:] = (2. * cross(rod.edges[i-1, :], rod.edges[i,:])
                /(norm(rod.edges[i-1,:])*norm(rod.edges[i,:])
                + dot(rod.edges[i-1, :], rod.edges[i,:])))
    end
end # function

"""
    matcurve(rod::cRod)

Computes the material curvatures of the rod
"""
function matcurve(rod::cRod)  ## omega_i^j in Bergou 2008
    omega = Array{Float64}(undef, rod.n, 2, 2)

    omega[1,1,1] = dot(rod.kb[1,:], rod.frame[end,3,:])
    omega[1,1,2] = - dot(rod.kb[1,:], rod.frame[end,2,:])

    omega[1,2,1] = dot(rod.kb[1,:], rod.frame[1,3,:])
    omega[1,2,2] = - dot(rod.kb[1,:], rod.frame[1,2,:])

    for i in 2:rod.n
        omega[i,1,1] = dot(rod.kb[i,:], rod.frame[i-1,3,:])
        omega[i,1,2] = - dot(rod.kb[i,:], rod.frame[i-1,2,:])

        omega[i,2,1] = dot(rod.kb[i,:], rod.frame[i,3,:])
        omega[i,2,2] = - dot(rod.kb[i,:], rod.frame[i,2,:])

    end
    return omega
end # function


"""
    skewmat(a::Array{Float64})

Returns the skew-symmetric matrix such as for a and b 3-vectors, cross(a,b) = skewmat(a) * b
"""
function skewmat(a::Float64)
    return [[0, -a[3], a[2]],[a[3], 0, -a[1]],[-a[2], a[1], 0]]
end # function

"""
    bEnergy(rod::cRod)

Computes the bending energy of the rod
"""
function bEnergy(rod::cRod, alpha::Float64)
    E = 0.
    omega = matcurve(rod)
    for i in 1:rod.n
        for j in 1:2
            E += dot(omega[i,j,:], alpha*omega[i,j,:] )/(2. * rod.voronoi[i])
        end
    end
    return E
end # function

"""
    bForce(rod::cRod)

Computes the bending energy of the rod, for an naturally isotropic straight rod
"""
function bForce(rod::cRod)
    kb = rod.kb
    edges = rod.edges
    norm_edges = Array{Float64}(undef, rod.n)
    for i in 1:rod.n
        norm_edges[i] = norm(edges[i,:])
    end
    Fb = Array{Float64}(undef, rod.n, 3)
    for i in 2:rod.n-1 ### TODO: handle edge cases i=2, i=n-1
        ui = -((-2.* cross(edges[i,:], kb[i,:]) + (edges[i,:]*transpose(kb[i,:]) * kb[i,:]) +
            (-2.* cross(edges[i-1,:], kb[i,:]) + (edges[i-1,:]*transpose(kb[i,:]) * kb[i,:])))/
            (norm_edges[i-1]*norm_edges[i] + dot(edges[i,:], edges[i-1,:])))/rod.voronoi[i]

        v = (-2.* cross(edges[i-2,:], kb[i-1,:]) + (edges[i-2,:]*transpose(kb[i-1,:]) * kb[i-1,:]))/(norm_edges[i-2]*norm_edges[i-1] + dot(edges[i-1,:], edges[i-2,:]))/rod.voronoi[i-1]

        w = (-2.* cross(edges[i+1,:], kb[i+1,:]) + (edges[i+1,:]*transpose(kb[i+1,:]) * kb[i+1,:]))/(norm_edges[i]*norm_edges[i+1] + dot(edges[i,:], edges[i+1,:]))/rod.voronoi[i+1]

        clamp_contrib = .5 * (rod.nTwist/rod.len) * (-kb[i-1,:]/norm_edges[i] + kb[i+1,:]/norm_edges[i] + (-1./norm_edges[i-1] + 1./norm_edges[i+1]) * kb[i,:])

        # Fb[i,:] = (combination of ui, v, w, clamp_contrib...)

    end
end


"""
    bEnergy(rod::oRod)

Computes the bending energy of the rod
"""
function bEnergy(rod::oRod)
    E = 0.
    omega = rod.matcurve
    for i in 1:rod.n
        for j in 1:2
            E += dot(omega[i,j,:], rod.B*omega[i,j,:])) / (2. * rod.voronoi[i])
        end
    end
    return E
end # function


"""
    bForce(rod::oRod)
Computes the forces due to bending and twist elasticity

"""
function bForce(rod::oRod)

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
