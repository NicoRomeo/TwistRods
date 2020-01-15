
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
        vor[1] = (norm(edges[1,:]) + norm(edges[end,:])) / 2
        for i in 2:n
            vor[i] = (norm(edges[i,:]) + norm(edges[i-1,:])) / 2
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
        kb = Array{Float64}(undef, n-2, 3)
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
    chi::Array{Float64}
    ttilda::Array{Float64,2}
    dtilda::Array{Float64,3}
    theta::Array{Float64,1}
    frame::Array{Float64,3}
    J::Matrix{Int8,2}
    function oRod(X::Array{Float64,2}, nTwist::Float64)
        n = size(X)[1]
        midp = (X[[1:end-1],:] + X[[2:end],:]) / 2.
        edges = X[[2:end],:] - X[[1:end-1],:] # only n - 1 edges
        vor = Array{Float64}(undef,n)
        vor[1] = norm(edges[1,:]) / 2
        vor[n] = nor(edges[n,:]) / 2
        for i in 2:n-1
            vor[i] = (norm(edges[i,:]) + norm(edges[i-1,:])) / 2
        end
        normal = edges[[2:end],:] - edges[[1:end-1],:]
        for i in 1:n-1
            normalize!(normal[i,:])
        end
        binormal = Array{Float64}(undef,n-1,3)
        for i in 1:n-1
            binormal[i,:] = normalize!(cross(edges[i,:],normal[i,:]))
        end
        norm_edges = Array{Float64}(undef,n-1,3)
        for i in 1:n-1
            norm_edges[i,:] = normalize(edges[i,:])
        end
        frame = cat(norm_edges, normal, binormal, dims=3)
        chi = Array{Float64}(undef, n-2)
        for i in 1:n-2
            chi[i] = 1 .+ dot(edges[i,:], edges[i+1,:])
        end
        ttilda = (frame[1:end-1,1,:] + frame[2:end,1,:]) ./ chi
        dtilda = Array{Float64}(undef, n-2, 3,2)
        dtilda[:,:,1] = (frame[1:end-1,2,:] + frame[2:end,2,:]) ./ chi
        dtilda[:,:,1] = (frame[1:end-1,3,:] + frame[2:end,3,:]) ./ chi
        kb = Array{Float64}(undef, n-2, 3)
        len = sum([norm(edges[i,:]) for i in 1:n-1])

        #adding J (counterclockwise pi/2 rotation matrix)
<<<<<<< HEAD
        J = [0 -1;1 0]
        new(n, X, nTwist, len, midp, edges, vor, kb, chi, ttilda, dtilda, zeros(Float64, n), frame, J)
=======
        new(n, X, nTwist, len, midp, edges, vor, kb, zeros(Float64, n), frame, J)
>>>>>>> 6e96fbd824dfca8124f2441996368edd8c9ee244
    end # function
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

edge quanitity
"""
function vDom(rod::cRod)
    rod.voronoi[1] = norm(rod.edges[1,:]) + norm(rod.edges[end,:]) / 2
    for i in 2:rod.n
        rod.voronoi[i] = (norm(rod.edges[i,:]) + norm(rod.edges[i-1,:])) / 2
    end
end # function

"""
    vDom(oRod)
updates the voronoi domains' lengths of an open Rod.
See voronoi(cRod) for equivalent function

vertex quantity
"""
function vDom(rod::oRod)
    rod.voronoi[1] = norm(rod.edges[1,:]) / 2
    rod.voronoi[rod.n] = norm(rod.edges[rod.n - 1,:]) / 2
    for i in 2:(rod.n - 1)
        rod.voronoi[i] = (norm(rod.edges[i-1,:]) + norm(rod.edges[i,:])) / 2
    end
end # function


"""
    midpoints(rod)

returns array of midpoints of each edge  of the rod.
Output shape differs for closed or open rod

edge quanitity
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

interior quantity
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
    for i in 2:(rod.n-1)
        rod.kb[i-1,:] = (2. * cross(rod.edges[i-1, :], rod.edges[i,:])
                /(norm(rod.edges[i-1,:])*norm(rod.edges[i,:])
                + dot(rod.edges[i-1, :], rod.edges[i,:])))
    end
end # function

"""
    matcurve(rod::cRod)
Computes the material curvatures of the rod

interior quanitity
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
    chi(rod::oRod)

Computes the intermediate scalar quantity chi from "Discrete Viscous Thread", Bergou et al 2010 (appendix A)
Interior quantity
"""
function chi(rod::oRod)
    res = Array{Float64}(undef, rod.n-2)
    for i in 1:rod.n-2
        res[i] = 1 + dot(rod.edges[i,:], rod.edges[i+1,:])
    end
    return res
end # function


"""
    ttilda(rod::oRod)

Updates the quantity t tilda
Interior Quantity
"""
function update_tilda(rod::oRod)
    rod.ttilda = (rod.frame[1:end-1,1,:]+rod.frame[2:end,1,:]) ./ chi  #To test!
end # function


"""
    dtilda(rod::oRod)

Computes the abreviation d tilda. Array of d_1 tilda and d_2 tilda, each being an
Interior Quantity
"""
function dtilda(rod::oRod)
    dtilda[:,:,1] = (rod.frame[1:end-1,2,:] + rod.frame[2:end,2,:]) ./ chi
    dtilda[:,:,2] = (rod.frame[1:end-1,3,:] + rod.frame[2:end,3,:]) ./ chi
end # function


"""
    skewmat(a::Array{Float64})

Returns the skew-symmetric matrix such as for a and b 3-vectors, cross(a,b) = skewmat(a) * b

"""
function skewmat(a::Array{Float64})
    return [0 -a[3] a[2];a[3] 0 -a[1];-a[2] a[1] 0]
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

vertex quanitity
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
    twistgrad(rod::oRod)

returns the twist gradient ∂m_i/∂e^j
"""
function twistgrad(rod::oRod, i::Int64, j::Int64)
    if i == j
        return rod.kb[i,:]/(2*norm(rod.edges[i, :]))
    elseif (j == i - 1 && j >= 1)
        return rod.kb[i,:]/(2*norm(rod.edges[i-1, :]))
    else
        return [0.,0.,0.]
end # function

"""
    matcurvegrad(rod::oRod, i::Int64, j::Int64)

Returns the gradients of material curvatures ∂κ_i/∂e^j
"""
function matcurvegrad(rod::oRod, i::Int64, j::Int64)
    if i == j
        return cat((-rod.matcurv[i,1]*rod.ttilda[i,:] - cross(rod.frame[i-1,1,:], rod.dtilda[i,:,2]))/norm(rod.edges[i-1, :]),
                (rod.matcurv[i,2]*rod.ttilda[i,:]+cross(rod.frame[i-1,1,:], rod.dtilda[i,:,1]))/norm(rod.edges[i-1, :])
                ,dims=2)
    elseif (j == i - 1 && j >= 1)
        return cat((-rod.matcurv[i,1]*rod.ttilda[i,:]+cross(rod.frame[i,1,:], rod.dtilda[i,:,2]))/norm(rod.edges[i-1, :]),
                (rod.matcurv[i,2]*rod.ttilda[i,:]-cross(rod.frame[i,1,:], rod.dtilda[i,:,1]))/norm(rod.edges[i-1, :])
                ,dims=2)
    else
        return [0.,0.,0.]
end # function


"""
    bForce(rod::oRod)
Computes the forces due to bending and twist elasticity

 vertex quanitity
"""
function bForce(rod::oRod)

end # function


###### collision handling ####
"""
    dist(aRod)

returns matrix of distances between points of the rod

edge quanitity
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
