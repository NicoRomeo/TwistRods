
using DifferentialEquations
using Flux
using LinearAlgebra

##########
# Draft code for non-linear implicit euler solver
#
#
#
##
#n = 10

#pos = zeros(4, n)
#pos = rand(4, n)

function initialize()


end # func

function curvebnom(x, frames)

end

function energy(X, theta, u0, p, t)
    n = p[1]
    B = [1 0; 0 1]

    edges = diff(X, dims=2)
    m = diff(theta)
    print(size(edges))

    tangent = zeros(Float64, (3, n-1))
    
    for i in 1:(n-1)
        tangent[:,i] = normalize(edges[:,i])
    end

    kb = zeros(Float64, (3, n-2))
    phi = zeros(Float64, n-2)
    for i in 1:(n-2)
        kb[:,i] = 2 .* cross(tangent[:,i], tangent[:,i+1])/(1 + dot(tangent[:,i], tangent[:,i+1]))
        phi[i] = 2*atan(norm(kb[:,i])/2)
    end

    # Bishop frame
    u = zeros(Float64, (3, n-1))
    u[:,1] = normalize(u0)
    for i in 1:(n-2)
        ax = normalize(kb[:,i])
        u[:,i+1] = dot(ax, u[:,i]) * ax + cos(phi[i]) *cross(cross(ax, u[:,i]), ax) + sin(phi[i]) * cross(ax, u[:,i])
    end
    v = zeros(Float64, (3, n-1))
    for i in 1:(n-1)
        v[:,i] = cross(tangent[:,i], u[:,i])
    end
    m1 = zeros(Float64, (3, n-1))
    m2 = zeros(Float64, (3, n-1))
    for i in 1:(n-1)
        m1[:,i] = cos.(theta[i]) * u[:,i] + sin.(theta[i]) * v[:,i]
        m2[:,i] = -sin.(theta[i]) * u[:,i] + cos.(theta[i]) * v[:,i]
    end

    kappa = zeros(Float64, (2, n-2))
    for i in 1:(n-2)
        kappa[:,i] = .5 * [dot(kb[:,i], m2[:,i] + m2[:,i+1]), -dot(kb[:,i], m1[:,i] + m1[:,i+1])]
    end

    ell = zeros(Float64, n)
    ell[1] = .5 * norm(edges[:,1])
    ell[n] = .5 * norm(edges[:,n-1])
    for i in 2:(n-1)
        ell[i] = .5 *(norm(edges[:,i-1]) + norm(edges[:,i]))
    end
    Ebend = 0
    for i in 1:(n-2)
        Ebend += kappa[:,i]' * B * kappa[:,i] / ell[i+1]
    end
    Ebend = .5 * Ebend

    Etwist = sum(m .^2 ./ ell[2:n-1])



    return Ebend + Etwist
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
