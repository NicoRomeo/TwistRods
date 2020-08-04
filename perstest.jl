##
## Test for the DiffEq solver
##
##

##
## ISSUES
# 1. Fix x2 gradient checks
#
##
## CODE
#
#

using DifferentialEquations
using Flux
using LinearAlgebra
using Plots

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

function normd(x)
    return x / sqrt(x'x)
end

function energy(X, theta, u0, p, t)
    n = p[1]
    B = [1 0; 0 1]

    edges = diff(X, dims = 2)
    m = diff(theta)

    tangent = zeros(Float64, (3, n - 1))

    for i = 1:(n-1)
        tangent[:, i] = normalize(edges[:, i])
    end

    kb = zeros(Float64, (3, n - 2))
    phi = zeros(Float64, n - 2)
    for i = 1:(n-2)
        kb[:, i] =
            2 .* cross(tangent[:, i], tangent[:, i+1]) /
            (1 + dot(tangent[:, i], tangent[:, i+1]))
        phi[i] = 2 * atan(norm(kb[:, i]) / 2)
    end

    # println("this is kb:")
    # println(kb)

    # Bishop frame
    u = zeros(Float64, (3, n - 1))
    u[:, 1] = normalize(u0)
    for i = 1:(n-2)
        ax = normalize(kb[:, i])
        u[:, i+1] =
            dot(ax, u[:, i]) * ax +
            cos(phi[i]) * cross(cross(ax, u[:, i]), ax) +
            sin(phi[i]) * cross(ax, u[:, i])
    end
    v = zeros(Float64, (3, n - 1))
    for i = 1:(n-1)
        v[:, i] = cross(tangent[:, i], u[:, i])
    end
    m1 = zeros(Float64, (3, n - 1))
    m2 = zeros(Float64, (3, n - 1))
    for i = 1:(n-1)
        m1[:, i] = cos.(theta[i]) * u[:, i] + sin.(theta[i]) * v[:, i]
        m2[:, i] = -sin.(theta[i]) * u[:, i] + cos.(theta[i]) * v[:, i]
    end

    kappa = zeros(Float64, (2, n - 2))
    for i = 1:(n-2)
        kappa[:, i] =
            0.5 * [
                dot(kb[:, i], m2[:, i] + m2[:, i+1]),
                -dot(kb[:, i], m1[:, i] + m1[:, i+1]),
            ]
    end

    ell = zeros(Float64, n)
    ell[1] = 0.5 * norm(edges[:, 1])
    ell[n] = 0.5 * norm(edges[:, n-1])
    for i = 2:(n-1)
        ell[i] = 0.5 * (norm(edges[:, i-1]) + norm(edges[:, i]))
    end
    Ebend = 0
    for i = 1:(n-2)
        Ebend += kappa[:, i]' * B * kappa[:, i] / ell[i+1]
    end
    Ebend = 0.5 * Ebend

    Etwist = sum(m .^ 2 ./ ell[2:n-1])

    return Ebend + Etwist
end

# function energydict(
#     X::AbstractArray,
#     theta::AbstractArray,
#     u0::AbstractArray,
#     p,
# )
#     n = p[1]
#     l0 = p[2]
#     B = [1 0; 0 1]
#
#     # edges, tangent, kb, phi
#     edges = Dict{Int32,Array{Float64,1}}()
#     tangent = Dict{Int32,Array{Float64,1}}()
#     kb = Dict{Int32,Array{Float64,1}}()
#     phi = Dict{Int32,Float64}()
#
#     # voronoi domain
#     ell = Dict{Int32,Float64}()
#
#     edges[1] = X[:, 2] - X[:, 1]
#     tangent[1] = normd(edges[1])
#     ell[1] = 0.5 * sqrt(edges[1]' * edges[1])
#
#     m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)
#
#     # Bishop frame
#     u = Dict{Int32,Array{Float64,1}}()
#     v = Dict{Int32,Array{Float64,1}}()
#     # Material frame
#     m1 = Dict{Int32,Array{Float64,1}}()
#     m2 = Dict{Int32,Array{Float64,1}}()
#
#     u[1] = normd(u0)
#     v[1] = cross(tangent[1], u[1])
#     m1[1] = cos(theta[1]) * u[1] + sin(theta[1]) * v[1]
#     m2[1] = -sin(theta[1]) * u[1] + cos(theta[1]) * v[1]
#     Ebend = 0.0
#     Etwist = 0.0
#     Estretch = (edges[1] .- l0)' * (edges[1] .- l0)
#     for i = 1:(n-2)
#         edges[i+1] = X[:, i+2] - X[:, i+1]
#         tangent[i+1] = normd(edges[i+1])
#         kb[i] =
#             2 .* cross(tangent[i], tangent[i+1]) /
#             (1 + tangent[i]' * tangent[i+1])
#         kbn = sqrt(kb[i]' * kb[i])
#         phi[i] = 2 * atan(kbn / 2)
#
#         ax = kb[i] / kbn
#
#         ell[i+1] =
#             0.5 * (sqrt(edges[i]' * edges[i]) + sqrt(edges[i+1]' * edges[i+1]))
#
#         u[i+1] =
#             dot(ax, u[i]) * ax +
#             cos(phi[i]) * cross(cross(ax, u[i]), ax) +
#             sin(phi[i]) * cross(ax, u[i])
#         v[i+1] = cross(tangent[i+1], u[i+1])
#         m1[i+1] = cos(theta[i+1]) * u[i+1] + sin(theta[i+1]) * v[i+1]
#         m2[i+1] = -sin(theta[i+1]) * u[i+1] + cos(theta[i+1]) * v[i+1]
#         k = 0.5 * [dot(kb[i], m2[i] + m2[i+1]), -dot(kb[i], m1[i] + m1[i+1])]
#
#         Ebend += k' * B * k / ell[i+1]
#         Etwist += m[i] .^ 2 / ell[i+1]
#         s = edges[i+1] .- l0
#         Estretch += s's
#     end
#     Ebend = 0.5 * Ebend
#     Estretch = 0.5 * Estretch
#     return Ebend + Etwist + Estretch
# end # function


function energy_clean(
    X::AbstractArray,
    theta::AbstractArray,
    u0::AbstractArray,
    p,
)
    n = trunc(Int,p[1])
    l0 = p[2]
    B = [1 0; 0 1]

    #fix β, used in twist calculations (see notes)
    β = 1
    # edges, tangent, kb, phi

    edges = X[:,2] - X[:,1]
    # println("these are edges")
    # println(edges)

    tangent = normd(edges)
    ell = 0.5 * sqrt(edges'edges)
    # println("this is ell")
    # println(ell)

    # m = zeros(Float64, n - 1)
    # print(m)
    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)
    # for i = 1:n-1
    #     m[i] = theta[i+1] - theta[i]
    # end #for loop

    # println("this is m")
    # println(m)
    # println(size(m))
    u = normd(u0)
    # println("this is u, tangent")
    # println(u, tangent)

    v = cross(tangent, u)
    m1 = cos(theta[1]) * u + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u + cos(theta[1]) * v
    Ebend = 0.0
    # Etwist = 0.0, I believe this is incorrect
    Etwist = m[n-1]^2 / ell

    s = edges'edges - l0
    Estretch = s's
    for i = 1:(n-2)
        edges_1 = X[:, i+2] - X[:, i+1]
        tangent_1 = normd(edges_1)
        kb = 2 .* cross(tangent, tangent_1) / (1 + tangent'tangent_1)

        # println("this is kb")
        # println(kb)

        kbn = sqrt(kb'kb)
        ell = 0.5 * (sqrt(edges'edges) + sqrt(edges_1'edges_1))
        phi = 2 * atan(kbn / 2)

        if !isapprox(kbn, 0.0)
            ax = kb / kbn
            u =
                dot(ax, u) * ax +
                cos(phi) * cross(cross(ax, u), ax) +
                sin(phi) * cross(ax, u)
        end
        v = cross(tangent_1, u)
        m1_1 = cos(theta[i+1]) * u + sin(theta[i+1]) * v
        m2_1 = -sin(theta[i+1]) * u + cos(theta[i+1]) * v
        k = 0.5 * [dot(kb, m2 + m2_1), -dot(kb, m1 + m1_1)]

        Ebend += k' * B * k / ell
        Etwist += m[i] ^ 2 / ell
        s = edges_1'edges_1 .- l0
        Estretch += s*s
        # update for next vertex
        edges = edges_1
        tangent = tangent_1
        m1 = m1_1
        m2 = m2_1
    end
    Ebend = 0.5 * Ebend
    Estretch = 0.5 * Estretch
    Etwist = 0.5 * Etwist
    # println("Below are twist and bend")
    # println(Etwist)
    # println(Ebend)
    # println(Etwist + Ebend)
    return Ebend + Etwist #right?!
end # function


#not sure why below is still in code?

# function state2vars(state::Array{Float64,2})
#     x = state[1:3, 2:end]
#     theta = state[4, 2:end-1]
#     u0 = state[1:3, 1]
#     return (x, theta, u0)
# end

function vars2state(
    x::Array{Float64},
    theta::Array{Float64},
    u0::Array{Float64},
)
    return vcat(u0, vec(x), theta)
end

function state2vars(state::Array{Float64,1}, n::Integer)
    pos = reshape(state[4:3*(n+1)], (3, n))
    theta = state[3*n+4:end]
    u0 = state[1:3]
    return (u0, pos, theta)
end

function rotatea2b(a::Array{Float64,1}, b::Array{Float64,1})
    return 2.0 * (a + b) * (a + b)' / ((a + b)'* (a + b)) - [1 0 0; 0 1 0; 0 0 1]
end

function force(ds, state::Array{Float64,1}, param, t)
    # unpack state variables
    n = param[1]
    pos = reshape(state[4:3*(n+1)], (3, n))
    # println("this is pos in 262")
    # println(pos)
    theta = state[3*n+4:end]
    u0 = state[1:3]
    # define fucntions for energy/gradient
    Ex = x -> energy_clean(x, theta, u0, param)
    Et = t -> energy_clean(pos, t, u0, param)

    # print(typeof(Ex))
    # print(typeof(Et))
    # println("Hello")
    fx = -1.0 * Flux.gradient(Ex, pos)[1]
    ft = -1.0 * Flux.gradient(Et, theta)[1]

    # update u0...
    fu0 = fx[:, 2] - fx[:, 1]

    ds = vars2state(fx, ft, fu0)
    return fx,ft
end


"""
NO TWIST, 45° BEND
"""

"""
GRAPH 1: Straight bend
"""

println("%%%%%%%%%%%%%%%%%%%%%% STRAIGHT BEND %%%%%%%%%%%%%%%%%%%%%%%%%")

#note that theta is in angles ==> CHANGE TO RADIANS

energy_bend = plot()
en(theta) = energy_clean([0. 1. 1+cos(theta*pi/180);
        0. 0. sin(theta*pi/180);0. 0. 0.],[0,0,0],[0.;1.;0.],[3,1])
en_hand(theta) = 2*(sin(theta*pi/180)/(1 + cos(theta*pi/180)))^2

plot!(en,0,90, label = "code")
plot!(en_hand,0,90, label = "hand")
xlabel!("Angle of bend (°)")
ylabel!("Energy")
title!("1: Angle of bend vs. energy")

display(energy_bend)

p0 = [0. 1. 1;0. 0. 1.;0. 0. 0.]

"""
TESTING FLUX.GRADIENT
"""

println("%%%%%%%%%%%%%%%%%%%%%% FLUX GRADIENT%%%%%%%%%%%%%%%%%%%%%%%%%")

Ex = x -> energy_clean(x, [0,0,0],[0.,1.,0.], [3,1])
Et = t -> energy_clean(p0, t, [0.;1.;0.], [3,1])

fx = -1.0 * Flux.gradient(Ex, p0)[1]
ft = -1.0 * Flux.gradient(Et, [0,0,0])[1]

println("this is (fx,ft), bent 90°")
println(fx,ft)

println("this is bend Energy, bent 90°")
println(Ex(p0))

"""
TESTING FLUX.GRADIENT, SERIES
"""

println("%%%%%%%%%%%%%%%%%%%%%% FLUX GRADIENT, SERIES%%%%%%%%%%%%%%%%%%%%%%%%%")

#everything in radians

function force_b(ϕ, theta, norm, param)

    p0 = [0 1 1+cos(ϕ); 0 0 sin(ϕ); 0 0 0]

    Ex = x -> energy_clean(x, theta,norm, param)
    fx = -1.0 * Flux.gradient(Ex, p0)[1]

    f_12 = -4*(sin(ϕ)/(1 + cos(ϕ))^2)
    f_11 = -(sin(ϕ)/(1 + cos(ϕ)))^2
    f_13 = 0

    return [f_11,f_12,f_13],fx[:,1]
end #function

# function force_b2(ϕ, theta, norm, param)
#
#     p0 = [0 1 1+cos(ϕ); 0 0 sin(ϕ); 0 0 0]
#
#     Ex = x -> energy_clean(x, theta,norm, param)
#     fx = -1.0 * Flux.gradient(Ex, p0)[1]
#
#     f_22 = (4sin(ϕ)/(1 + cos(ϕ))^2) * ((sin(ϕ)/2) - 2) * 0.5 #!!! fix !!!
#     f_21 = (4sin(ϕ)/(1 + cos(ϕ))^2) * ((sin(ϕ)/2) - 2) * -0.5 #!!! fix !!!
#     f_23 = 0
#
#     return [f_21,f_22,f_23],fx[:,2]
# end #function

function force_b2(ϕ, theta, norm, param)

    p0 = [0 1 1+cos(ϕ); 0 0 sin(ϕ); 0 0 0]

    Ex = x -> energy_clean(x, theta,norm, param)
    fx = -1.0 * Flux.gradient(Ex, p0)[1]

    f_22 = 0
    f_21 = -(((4*(sin(ϕ))^2)/(1 + cos(ϕ))^2) - (((sin(ϕ))^2)/(1 + cos(ϕ))^2)*(1 - cos(ϕ)))
    f_23 = 0

    return f_21,fx[1,2]
end #function

println("this is hand calc vs. fx, bent ϕ_1, x1 + x3")

for i = 1:90

    println(force_b(i * pi/180, [0.,0.,0.], [0.,1.,0.],[3,1]))

end #for

#requires lots of fixing
println("this is hand calc vs. fx, bent ϕ_1, x2")

for i = 1:90
    println(force_b2(i * pi/180, [0.,0.,0.], [0.,0.,1.],[3,1]))
end #for


"""
GRAPH 2: Smaller  ==> bigger loops
"""

println("%%%%%%%%%%%%%%%%%%%%%% SMALLER ==> BIGGER LOOPS%%%%%%%%%%%%%%%%%%%%%%")
# x_circle = Array{Float64}()

energy_bend = plot()
# en(theta) = energy_clean([0. 1. 1+cos(theta*pi/180);
#         0. 0. sin(theta*pi/180);0. 0. 0.],[0,0,0],[0.;1.;0.],[3,1])

X = zeros(Float64,3,10)
function en(rad)
    X = zeros(Float64,3,10)
    for i = 1:10
        X[1,i] = rad * cos(2.0*pi*(i-1)/10)
        X[2,i] = rad * sin(2.0*pi*(i-1)/10)
    end #for loop

    # println("this is positions")
    # println(X)
    n1 = -cos(pi/10)
    n2 = -sin(pi/10)
    energy = energy_clean(X,zeros(10,1),[n1;n2;0.],[10,sqrt(2*rad^2*(1-cos(2*pi/10)))])
    return energy
end #function

sin_i = sin(2.0*pi/10)
cos_i = cos(2.0*pi/10)

en_hand(rad) = (2.0/ (sqrt(2*rad^2*(1-cos(2*pi/10))))) * (10 - 2) * (sin_i/(1+cos_i))^2

# plot!(en,1,10, xaxis=:log, yaxis=:log)
plot!(en,1,10, label = "code")
plot!(en_hand,1,10, label = "hand")
xlabel!("Radius of loop")
ylabel!("Bend + twist energy")
title!("2: Radius of loop vs. energy")

display(energy_bend)


"""
GRAPH 3: Straight rod w/ twist
"""

println("%%%%%%%%%%%%%%%%%%%%%% STRAIGHT ROD W/ TWIST%%%%%%%%%%%%%%%%%%%%%%%%%")

energy_tot = plot()
# en(theta) = energy_clean([0. 1. 1+cos(theta);
#         0. 0. sin(theta);0. 0. 0.],[0.,pi/6,0.],[0.;1.;0.],[3,1])

function en(rad)
    X = [0 1 2; 0 0 0; 0 0 0]
    energy = energy_clean(X,[0,rad,rad],[0.;1.;0.],[3,1])
    return energy
end #function

# plot!(en,10^-6,pi,xaxis=:log, yaxis=:log)
plot!(en,10^-6,pi)
xlabel!("Angle of twist (rad)")
ylabel!("Energy")
title!("3: Angle twist in straight rod vs. energy")

display(energy_tot)

"""
GRAPH 4: Straight rod w/ twist throughout
"""

println("%%%%%%%%%%%%%%%%%%%%%% STRAIGHT ROD W/ TWIST THROUGHOUT%%%%%%%%%%%%%")

energy_tot = plot()
# en(theta) = energy_clean([0. 1. 1+cos(theta);
#         0. 0. sin(theta);0. 0. 0.],[0.,pi/6,0.],[0.;1.;0.],[3,1])

# vertices = 8
# function en(theta)
#     X = [0 1 2 3 4 5 6 7; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0]
#     energy = energy_clean(X,[0,theta,theta*2, theta*3, theta * 4, theta*5, theta*6, theta*7],
#                             [0.;1.;0.],[8,1])
#     return energy
# end #function

vertices = 7
function en(theta)
    X = [0 1 2 3 4 5 6; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0]
    energy = energy_clean(X,[0,theta,theta*2, theta*3, theta * 4, theta*5, theta*6],
                            [0.;1.;0.],[7,1])
    return energy
end #function

#twist throughout
#note the OFF-BY-ONE error
println("num of vertices")
println(vertices)
en_hand(theta) = 0.5 * theta^2 * (vertices)

# plot!(en,10^-6,pi,xaxis=:log, yaxis=:log)
plot!(en,10^-6,pi)
plot!(en_hand,10^-6,pi,label = "hand")
xlabel!("Angle of twist (rad)")
ylabel!("Energy")
title!("3: Angle twist in straight rod vs. energy")

display(energy_tot)

# println("twist energy")
# println(energy_clean([0. 1. 1+cos(0);
#         0. 0. sin(0);0. 0. 0.],[0.,pi/6,0.],[0.;1.;0.],[3,1]))
