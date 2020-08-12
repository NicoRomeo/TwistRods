"""
pseudocode / steps

goal: specify endpoints, twist, and axial shortening

"""
## Buckling test
## ISSUES
# 1.
##


using Plots
using ColorSchemes
#using Makie
#using Flux
using Zygote
using LinearAlgebra
using NLsolve
using DelimitedFiles

include("newrod.jl")
## Functions to access states and back
# function state2vars(state::Array{Float64,1}, n::Integer)
# function vars2state(x::Array{Float64},theta::Array{Float64},u0::Array{Float64})
##

function energy_q(q, u0::Array{Float64,1}, p)
    n = Int(p[1])
    l0 = p[2]
    B = [1 0; 0 1]

    # edges, tangent, kb, phi
    #X = reshape(q[1:3*n], (3,n))
    X = q[1:3*n]
    theta = q[3*n+1:end]

    #edges = X[:, 2] - X[:, 1]
    edges = q[4:6] - q[1:3]
    tangent = normd(edges)
    ell = 0.5 * sqrt(edges'edges)

    m = diff(theta, dims = 1) #Dict(i => theta[i+1] - theta[i] for i in 1:n-1)

    u = normd(u0)
    v = cross(tangent, u)
    m1 = cos(theta[1]) * u + sin(theta[1]) * v
    m2 = -sin(theta[1]) * u + cos(theta[1]) * v
    Ebend = 0.0
    Etwist = 0.0
    s = sqrt(edges'edges) - l0
    Estretch = s's
    i= 1
    while i < n - 1
    #for i = 1:(n-2)
        #edges_1 = X[:, i+2] - X[:, i+1]
        edges_1 = q[3*(i+1)+1:3*(i+2)] - q[3*i+1:3*(i+1)]
        tangent_1 = normd(edges_1)
        kb = 2 .* cross(tangent, tangent_1) / (1 + tangent'tangent_1)
        kbn = sqrt(kb'kb)
        ell = 0.5 * (sqrt(edges'edges) + sqrt(edges_1'edges_1))
        if !isapprox(kbn, 0.0)
            phi = 2 * atan(dot(kb, kb / kbn) / 2)
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
        Etwist += m[i] .^ 2 / ell
        s = sqrt(edges_1'edges_1) .- l0
        Estretch += s * s
        # update for next vertex
        edges = edges_1
        tangent = tangent_1
        m1 = m1_1
        m2 = m2_1
        i += 1
    end
    ell = 0.5 * sqrt(edges'edges)
    Etwist += m[end] .^ 2 / ell
    Ebend = 0.5 * Ebend
    Estretch = 0.5 * Estretch
    Etwist = 0.5 * Etwist
    return Ebend + Etwist #+ Estretch
end # function



function skewmat(a::Array{Float64})
    return [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
end # function

"""
    rotab(a, b, c)

rotation matrix sending unit vectors a to b, applied to c
"""
function rotab(a::Array{Float64,1}, b::Array{Float64,1}, c::Array{Float64,1})
    axb = cross(a, b)
    V = skewmat(axb)
    u = c + cross(axb, c) + V*V * c / (1 + dot(a, b))
    return u
end # function



function F!(f::Array{Float64,1}, q::Array{Float64,1}, u0, param)
    E = mu -> energy_q(mu, u0, param)
    f[:] = -1.0 * Zygote.gradient(E, q)[1]
end

"""
vor_length(x_inp)
"""

function vor_length(x_inp)
    x_inp = vec(x_inp)
    n = size(x_inp)[1] / 3
    n = Int(n)

    x_inp = reshape(x_inp,(3,n))
    edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]

    vl = zeros(n-1) #1 per edge

    #note major assumption: vor length determines undeformed config
    for i=1:n-1
        vl[i] = sqrt(dot(edges[:,i],edges[:,i]))
        # println(C[i])
    end #loop

    vl_fin = vec(vl)
    return vl_fin
end #function

"""
    timestep(args)

function that runs a single timestep using an RK4 time step.
    args: F force function
    state

"""

# timestep(F!, f, state, dt, param, i, txt_array, err_tol)
function timestep(
    F!,
    f::Array{Float64,1},
    state::Array{Float64,1},
    dt::Float64,
    param,
    txt_array,
    err_tol,
    fixed_ends,
    constraint_func,
    gen_t,
)
    n = Int(param[1])
    l0 = param[2]
    x, theta, u0 = state2vars(state, n)
    init_state = state[4:end]
    # println("init_state, start of timestep: ", init_state)

    q_i = vcat(vec(x), theta)
    tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])

    # Compute explicit initial guess for X, theta at t+dt

    #RK 4
    F!(f, q_i, u0, param)
    # println("first f: ", f)
    q_g = (dt / 2) * f + q_i
    e1 = LinearAlgebra.normalize!(q_g[4:6] - q_g[1:3])
    u1 = rotab(tangent0, e1, u0)

    F!(f, q_g, u1, param)

    q = dt * f + q_i
    e1 = LinearAlgebra.normalize!(q[4:6] - q[1:3])
    u1 = rotab(tangent0, e1, u0)

    #RK4
    q0 = vcat(vec(x), theta)
    tan0 = LinearAlgebra.normalize!(q0[4:6] - q0[1:3])

    F!(f, q0, u0, param)
    k1 = f
    q1 = (dt / 2) * f + q0
    e1 = LinearAlgebra.normalize!(q1[4:6] - q1[1:3])
    # u1 = rotab(tangent0, e1, u0)
    # println("last f: ", f)

    F!(f, q1, u1, param)
    k2 = f
    q2 = (dt / 2) * f + q1
    e2 = LinearAlgebra.normalize!(q2[4:6] - q2[1:3])
    u2 = rotab(tangent0, e2, u0)
    # println("last f: ", f)

    F!(f, q2, u2, param)
    k3 = f
    q3 = (dt / 2) * f + q2
    e3 = LinearAlgebra.normalize!(q3[4:6] - q3[1:3])
    u3 = rotab(tangent0, e3, u0)

    F!(f, q3, u3, param)
    k4 = f
    q4 = (dt) * f + q3
    e4 = LinearAlgebra.normalize!(q4[4:6] - q4[1:3])
    u4 = rotab(tangent0, e4, u0)

    # println("last f: ", f)
    x_fin = q_i + (1/6 * dt * (k1 + 2k2 + 2k3 + k4))
    e_fin = LinearAlgebra.normalize!(x_fin[4:6] - x_fin[1:3])
    u_fin = rotab(tangent0, e_fin, u0)

    # println("this is ufin, xfin pre-projection")
    init = vcat(u_fin,x_fin)
    # println(vcat(u_fin,x_fin))
    """
    CONSTRAINTS
    """

    j =  0

    #lumped model
    #init mass
    ℳ = n #total scalar mass
    m_k = ℳ/n #assuming uniform rod

    x = reshape(x_fin[1:3n], (3,n)) #reshape x, twist last row
    theta_proj = x_fin[3n+1:end]
    # println("theta_proj: ", theta_proj)
    edges = x[:, 2:end] - x[:, 1:end-1] #defining edges

    #mass matrix: uniform, lumped rod, CHECK THISß

    # #init generalized mass matrix
    M = zeros(4n, 4n)
    # M = zeros(3n,3n)
    mass_dens = 1.
    rad_edge = 1.

    M[1:3n,1:3n] = m_k * Matrix(1.0I, 3n, 3n)

    #moment of inertia defined on edges, cylindrical rod
    #this is assuming i know how to integrate
    M[3n+1:4n, 3n+1:4n] = Matrix(1.0I,n,n) #(mass_dens * rad_edge^3 * 2*pi)/3 *

    #computing inverse
    #check if faster way to do this
    M_inv = inv(M)
    # println(M)
    #init C
    #(x_inp, vor1, start1, end1, t_cur)

    C = constraint_func(x, theta_proj, l0, fixed_ends[1], fixed_ends[2], gen_t)
    C_abs = zeros(size(C)[1])

    for i= 1:size(C)[1]
        C_abs[i] = abs(C[i])
    end #for
    maxC = maximum(C_abs)

    vor_len = vor_length(x)

    io = open(txt_array[1], "a")
    writedlm(io, C)
    write(io, "* \n")
    close(io)

    io = open(txt_array[2], "a")
    writedlm(io, "* ", vor_len)
    close(io)

    iteration = 1

    # while maxC >= err_tol || maxC <= -err_tol
    while maxC >= err_tol
        #vectorize
        x = vec(x)

        #initializing gradC
        #n-1 ==> pos
        #2 ==> boundary
        #2 ==> boundary theta

        gradC = zeros(n-1 + 2 + 2,4*n)

        #calculating gradient, must be changed depending on constraint
        #4:n+2

        for r = 1:n-1
            gradC[r, 3*(r-1)+1:3*(r-1)+3] = -2 * edges[:,r]
            gradC[r, 3*(r-1)+4:3*(r-1)+6] = 2 * edges[:,r]
        end #for

        # for r = 1:n
        #     gradC[r, 3*(r-4)+1:3*(r-4)+3] = -2 * edges[:,r-3]
        #     gradC[r, 3*(r-4)+4:3*(r-4)+6] = 2 * edges[:,r-3]
        # end #for

        #2
        gradC[n, 1:3] = 2 * (x[1:3] - fixed_ends[1])
        gradC[n+1,3n-2:3n] = 2 * (x[3n-2:3n] - fixed_ends[2])

        #theta
        gradC[n+2,3n+1] = 2 * (theta_proj[1] - 0.)
        gradC[n+3,end] = 2 * (theta_proj[end] - 0.9)

        # gradC = Zygote.gradient(constraint, x)

        #solving for lagrange multipliers
        # println("gradC: ", gradC)
        gradC_t = transpose(gradC)
        G = gradC * M_inv * gradC_t

        # println("G: ", G)
        #note: G is sparse. Not quite banded (note staggered rows)
        fin_det = det(G)

        # println("fin_det: ", fin_det)
        G_inv = pinv(G) #check runtime

        δλ = G_inv * C
        δλ_next = δλ / dt^2
        δx_next = -dt^2 * (M_inv * gradC_t * δλ_next)

        #x_next_fp is vec
        # δx_next[1:3] = zeros(3)
        # δx_next[end-2:end] = zeros(3)
        x_next_fp = x_fin + δx_next
        j += 1

        iteration += 1
        #______________re_init___________________
        #updating x, edges, C
        x_fin = x_next_fp
        x_arr = reshape(x_fin[1:3n],(3,n))
        x = x_fin[1:3n]
        theta_proj = x_fin[3n+1:end]

        # x_arr[:,1] = fixed_ends[1] + [(gen_t * dt), 0, 0]
        # x_arr[:,end] = fixed_ends[2] - [(gen_t * dt), 0, 0]

        edges = x_arr[:, 2:end] - x_arr[:, 1:end-1] #defining edges

        M[1:3n,1:3n] = m_k * Matrix(1.0I, 3n, 3n)

        #moment of inertia defined on edges, cylindrical rod
        #this is assuming i know how to integrate
        M[3n+1:4n, 3n+1:4n] = Matrix(1.0I,n,n) #(mass_dens * rad_edge^3 * 2pi)/3 *

        #computing inverse
        M_inv = inv(M)

        #initializing C
        C = constraint_func(x, theta_proj, l0, fixed_ends[1], fixed_ends[2], gen_t)
        C_abs = zeros(size(C)[1])

        for i= 1:size(C)[1]
            C_abs[i] = abs(C[i])
        end #for

        maxC = maximum(C_abs)

        #initializing vor length
        vor_len = vor_length(x)

        io = open(txt_array[1], "a")
        writedlm(io, C)
        write(io, "* \n")
        close(io)

        io = open(txt_array[2], "a")
        writedlm(io, "* ", vor_len)
        close(io)

    end #while

    """

    insert velocity update

    vvv ISSUE BELOW vvv
    """
    #h^2(∇C(x_j) M^-1 ∇C(x_j)^T)δλ_{j+1} = C(x_j)

    # package things use, and return the new state
    x = reshape(x,(3,n))

    e_fin_proj = LinearAlgebra.normalize!(x[:,2] - x[:,1])
    u_fin_proj = rotab(tangent0, e_fin_proj, u0)

    x_fin_proj = vec(x)
    u_fin_proj = vec(u_fin_proj)

    fin_proj = Vector{Float64}(undef, 4n+3)
    fin_proj[1:3] = u_fin_proj
    fin_proj[4:3n+3] = x_fin_proj
    fin_proj[3n+4:end] = theta_proj

    #calculating final f
    f_prop = (fin_proj[4:end] - init_state)/dt

    #writing to text file
    io = open(txt_array[1], "a")
    # write(io, maxC, "\n")
    # C = constraint(x)
    write(io, "______ \n")
    write(io, "______ \n")
    close(io)

    io = open(txt_array[2], "a")
    write(io, "fin: ")
    writedlm(io, vor_len)
    write(io, "______ \n")
    close(io)

    io = open(txt_array[3], "a")
    writedlm(io, sum(f_prop))
    write(io, "________\n")
    close(io)

    return fin_proj #vcat(u4, q4),
end # function

"""
timestep_axis
"""

function timestep_axis(
    F!,
    f::Array{Float64,1},
    state::Array{Float64,1},
    dt::Float64,
    param,
    txt_array,
    err_tol,
    fixed_ends,
    constraint_func,
    gen_t,
)
    n = Int(param[1])
    l0 = param[2]
    x, theta, u0 = state2vars(state, n)
    init_state = state[4:end]
    # println("init_state, start of timestep: ", init_state)

    q_i = vcat(vec(x), theta)
    tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])

    # Compute explicit initial guess for X, theta at t+dt

    #RK 4
    F!(f, q_i, u0, param)
    # println("first f: ", f)
    q_g = (dt / 2) * f + q_i
    e1 = LinearAlgebra.normalize!(q_g[4:6] - q_g[1:3])
    u1 = rotab(tangent0, e1, u0)

    F!(f, q_g, u1, param)

    q = dt * f + q_i
    e1 = LinearAlgebra.normalize!(q[4:6] - q[1:3])
    u1 = rotab(tangent0, e1, u0)

    #RK4
    q0 = vcat(vec(x), theta)
    tan0 = LinearAlgebra.normalize!(q0[4:6] - q0[1:3])

    F!(f, q0, u0, param)
    k1 = f
    q1 = (dt / 2) * f + q0
    e1 = LinearAlgebra.normalize!(q1[4:6] - q1[1:3])
    # u1 = rotab(tangent0, e1, u0)
    # println("last f: ", f)

    F!(f, q1, u1, param)
    k2 = f
    q2 = (dt / 2) * f + q1
    e2 = LinearAlgebra.normalize!(q2[4:6] - q2[1:3])
    u2 = rotab(tangent0, e2, u0)
    # println("last f: ", f)

    F!(f, q2, u2, param)
    k3 = f
    q3 = (dt / 2) * f + q2
    e3 = LinearAlgebra.normalize!(q3[4:6] - q3[1:3])
    u3 = rotab(tangent0, e3, u0)

    F!(f, q3, u3, param)
    k4 = f
    q4 = (dt) * f + q3
    e4 = LinearAlgebra.normalize!(q4[4:6] - q4[1:3])
    u4 = rotab(tangent0, e4, u0)

    # println("last f: ", f)
    x_fin = q_i + (1/6 * dt * (k1 + 2k2 + 2k3 + k4))
    e_fin = LinearAlgebra.normalize!(x_fin[4:6] - x_fin[1:3])
    u_fin = rotab(tangent0, e_fin, u0)

    # println("this is ufin, xfin pre-projection")
    init = vcat(u_fin,x_fin)
    # println(vcat(u_fin,x_fin))
    """
    CONSTRAINTS
    """

    j =  0

    #lumped model
    #init mass
    ℳ = n #total scalar mass
    m_k = ℳ/n #assuming uniform rod

    x = reshape(x_fin[1:3n], (3,n)) #reshape x, twist last row
    theta_proj = x_fin[3n+1:end]
    # println("theta_proj: ", theta_proj)
    edges = x[:, 2:end] - x[:, 1:end-1] #defining edges

    #mass matrix: uniform, lumped rod, CHECK THISß

    # #init generalized mass matrix
    M = zeros(4n, 4n)
    # M = zeros(3n,3n)
    mass_dens = 1.
    rad_edge = 1.

    M[1:3n,1:3n] = m_k * Matrix(1.0I, 3n, 3n)

    #moment of inertia defined on edges, cylindrical rod
    #this is assuming i know how to integrate
    M[3n+1:4n, 3n+1:4n] = Matrix(1.0I,n,n) #(mass_dens * rad_edge^3 * 2*pi)/3 *

    #computing inverse
    #check if faster way to do this
    M_inv = inv(M)
    # println(M)
    #init C
    #(x_inp, vor1, start1, end1, t_cur)

    C = constraint_func(x, theta_proj, l0, fixed_ends[1], fixed_ends[2], gen_t)
    C_abs = zeros(size(C)[1])

    for i= 1:size(C)[1]
        C_abs[i] = abs(C[i])
    end #for
    maxC = maximum(C_abs)

    vor_len = vor_length(x)

    io = open(txt_array[1], "a")
    writedlm(io, C)
    write(io, "* \n")
    close(io)

    io = open(txt_array[2], "a")
    writedlm(io, "* ", vor_len)
    close(io)

    iteration = 1

    # while maxC >= err_tol || maxC <= -err_tol
    while maxC >= err_tol
        #vectorize
        x = vec(x)

        #initializing gradC
        #n-1 ==> pos
        #2 ==> boundary
        #2 ==> boundary theta

        gradC = zeros(n-1 + 2 + 2,4*n)

        #calculating gradient, must be changed depending on constraint
        #4:n+2

        for r = 1:n-1
            gradC[r, 3*(r-1)+1:3*(r-1)+3] = -2 * edges[:,r]
            gradC[r, 3*(r-1)+4:3*(r-1)+6] = 2 * edges[:,r]
        end #for

        # for r = 1:n
        #     gradC[r, 3*(r-4)+1:3*(r-4)+3] = -2 * edges[:,r-3]
        #     gradC[r, 3*(r-4)+4:3*(r-4)+6] = 2 * edges[:,r-3]
        # end #for

        v = l0/10 #v = velocity
        # adj_t = gen_t * 10000
#         C_fin[n] = dot(x_inp[:,1] - (start1  + (adj_t * v) * [1;0;0]),x_inp[:,1] - (start1  + (adj_t * v)* [1;0;0])) #note first vertex initializes @ origin
#         C_fin[n+1] = dot(x_inp[:,end] - (end1 - (adj_t * v) * [1;0;0]), x_inp[:,end] - (end1 - (adj_t * v) * [1;0;0]))
#         C_fin[n+2] = (theta_inp[1])^2
#         C_fin[n+3] = (theta_inp[end] - 0.9)^2
#
# fixed_ends[1], fixed_ends[2]

        fin_first= fixed_ends[1]  + (gen_t * v) * [1.;0.;0.]
        fin_last = fixed_ends[2] - (gen_t * v) * [1.;0.;0.]

        println("fin_first, fin_last", fin_first, fin_last)

        #2
        gradC[n, 1:3] = 2 * (x[1:3] - fin_first)
        gradC[n+1,3n-2:3n] = 2 * (x[3n-2:3n] - fin_last)

        #theta ==> changed
        gradC[n+2,3n+1] = 2 * (theta_proj[1] - 0.)
        gradC[n+3,end] = 2 * (theta_proj[end] - 0.9)

        # gradC = Zygote.gradient(constraint, x)

        #solving for lagrange multipliers
        # println("gradC: ", gradC)
        gradC_t = transpose(gradC)
        G = gradC * M_inv * gradC_t

        # println("G: ", G)
        #note: G is sparse. Not quite banded (note staggered rows)
        fin_det = det(G)

        # println("fin_det: ", fin_det)
        G_inv = pinv(G) #check runtime

        δλ = G_inv * C
        δλ_next = δλ / dt^2
        δx_next = -dt^2 * (M_inv * gradC_t * δλ_next)

        #x_next_fp is vec
        # δx_next[1:3] = zeros(3)
        # δx_next[end-2:end] = zeros(3)
        x_next_fp = x_fin + δx_next
        j += 1

        iteration += 1
        #______________re_init___________________
        #updating x, edges, C
        x_fin = x_next_fp
        x_arr = reshape(x_fin[1:3n],(3,n))
        x = x_fin[1:3n]
        theta_proj = x_fin[3n+1:end]

        # x_arr[:,1] = fixed_ends[1] + [(gen_t * dt), 0, 0]
        # x_arr[:,end] = fixed_ends[2] - [(gen_t * dt), 0, 0]

        edges = x_arr[:, 2:end] - x_arr[:, 1:end-1] #defining edges

        M[1:3n,1:3n] = m_k * Matrix(1.0I, 3n, 3n) #m_k *

        #moment of inertia defined on edges, cylindrical rod
        #this is assuming i know how to integrate
        M[3n+1:4n, 3n+1:4n] = Matrix(1.0I,n,n) #(mass_dens * rad_edge^3 * 2pi)/3 *

        #computing inverse
        M_inv = inv(M)

        #initializing C
        C = constraint_func(x, theta_proj, l0, fixed_ends[1], fixed_ends[2], gen_t)
        C_abs = zeros(size(C)[1])

        for i= 1:size(C)[1]
            C_abs[i] = abs(C[i])
        end #for

        maxC = maximum(C_abs)

        #initializing vor length
        vor_len = vor_length(x)

        io = open(txt_array[1], "a")
        writedlm(io, C)
        write(io, "* \n")
        close(io)

        io = open(txt_array[2], "a")
        writedlm(io, "* ", vor_len)
        close(io)

    end #while

    """

    insert velocity update

    vvv ISSUE BELOW vvv
    """
    #h^2(∇C(x_j) M^-1 ∇C(x_j)^T)δλ_{j+1} = C(x_j)

    # package things use, and return the new state
    x = reshape(x,(3,n))

    e_fin_proj = LinearAlgebra.normalize!(x[:,2] - x[:,1])
    u_fin_proj = rotab(tangent0, e_fin_proj, u0)

    x_fin_proj = vec(x)
    u_fin_proj = vec(u_fin_proj)

    fin_proj = Vector{Float64}(undef, 4n+3)
    fin_proj[1:3] = u_fin_proj
    fin_proj[4:3n+3] = x_fin_proj
    fin_proj[3n+4:end] = theta_proj

    #calculating final f
    f_prop = (fin_proj[4:end] - init_state)/dt

    #writing to text file
    io = open(txt_array[1], "a")
    # write(io, maxC, "\n")
    # C = constraint(x)
    write(io, "______ \n")
    write(io, "______ \n")
    close(io)

    io = open(txt_array[2], "a")
    write(io, "fin: ")
    writedlm(io, vor_len)
    write(io, "______ \n")
    close(io)

    io = open(txt_array[3], "a")
    writedlm(io, sum(f_prop))
    write(io, "________\n")
    close(io)

    return fin_proj #vcat(u4, q4),
end # function

function twist_color(inp_theta)
    fin_col = zeros(size(inp_theta))

    for i=1:size(inp_theta)[1]
        col = mod(inp_theta[i], 2*pi)
        col = col / (2*pi)
        fin_col[i] = col
    end #for

    # println("fin_col: ", fin_col)
    return fin_col
end #function

function runsim_twist(
    title,
    dims,
    err_tol,
    tspan,
    n_t,
    N,
    l0,
    pos_0,
    theta_0,
    u_0,
    constraint_type,
    limits
)
    println("%%%%%%%%%%%%", title, "%%%%%%%%%%%%%%%")

    #initializing text files
    max_txt = string(title, "_maxc", ".txt")
    vor_txt = string(title, "_vor", ".txt")
    sum_f  = string(title, "_sum_f", ".txt")

    txt_array = [max_txt, vor_txt, sum_f]

    #re-initializing files
    io = open(max_txt, "a")
    close(io)
    rm(max_txt)

    io = open(vor_txt, "a")
    close(io)
    rm(vor_txt)

    io = open(sum_f, "a")
    close(io)
    rm(sum_f)

    #opening files

    io = open(max_txt, "a")
    io = open(vor_txt, "a")
    io = open(sum_f, "a")

    # setting txt_switch
    txt_switch = title

    println("%%%%%%%%%%%% start of calcs ==> change below %%%%%%%%%%%%%%%")
    #initializing sim
    dt = (tspan[2] - tspan[1]) / n_t
    param = [N, l0]  #parameter vector

    state_0 = vars2state(pos_0, theta_0, u_0)
    x, theta, u_0 = state2vars(state_0, N)
    println("x, theta, u_0: ", x, theta, u_0)

    f = zeros(Float64, 4 * N)

    plt = plot(aspect_ratio=:equal)
    xlims!(limits[1])
    ylims!(limits[2])
    zlims!(limits[3])
    # zlims!((-1.,10.))
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1, x_2, x_3 = x_cur[1, :], x_cur[2, :], x_cur[3,:]
    twist_weights = twist_color(theta_0)

    # println(twist_weights)
    #initializing plot
    for i = 1:N-1
        scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                label = legend = false, aspect_ratio=:equal)
        plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                label = legend = false, aspect_ratio=:equal,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]])
    end #for loop

    title!(title)
    display(plt)
    #final plot / animation
#     F!,
#     f::Array{Float64,1},
#     state::Array{Float64,1},
#     dt::Float64,
#     param,
#     txt_array,
#     err_tol,
#     fixed_ends,
#     constraint_func,
#     gen_t
# )
    function step!(gen_t)
        state = timestep(F!, f, state, dt, param, txt_array, err_tol,
                        [pos_0[:,1],pos_0[:,end]], constraint_type, gen_t)
        t += dt

        # state_0 = vars2state(pos_0, theta_0, u_0)
        # x, theta, u_0 = state2vars(state_0, N)

        x_cur = reshape(state[4:3*(N+1)], (3, N))
        # println("theta =", state[3N+4:end])

        theta_cur = state[3*(N+1) + 1:end]
        twist_weights = twist_color(theta_cur)

        println("dt: ", t)
        println("this is theta, timsetp 1: ", theta_cur)
        println("this is x, timsetp 1: ", x_cur)
        println("************** * * * BOINK *8*** *8**88   8 ")

        # println(twist_weights)

        plt = plot(aspect_ratio=:equal)
        xlims!(limits[1])
        ylims!(limits[2])
        zlims!(limits[3])
        for i = 1:N-1
            scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                    label = legend = false)
            plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                    label = legend = false,
                    linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]])
        end #for

        title!(title)
    end #function

    t = 0.
    anim = @animate for i = 1:n_t
        step!(t)
    end every 1
    gif(anim, string(title,".gif"), fps = (Int(floor(n_t/5))))

    #force/torque checks
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f_bend) = ", sum(f[1:3N]))
    println("sum(f_twist) = ", sum(f[3N+1:4N]))

    display(plt)
    println("u =", state[1:3])
    png(title)

    return state
end #function

function runsim_axis(
    title,
    dims,
    err_tol,
    tspan,
    n_t,
    N,
    l0,
    pos_0,
    theta_0,
    u_0,
    constraint_type,
    limits
)
    println("%%%%%%%%%%%%", title, "%%%%%%%%%%%%%%%")

    #initializing text files
    max_txt = string(title, "_maxc", ".txt")
    vor_txt = string(title, "_vor", ".txt")
    sum_f  = string(title, "_sum_f", ".txt")

    txt_array = [max_txt, vor_txt, sum_f]

    #re-initializing files
    io = open(max_txt, "a")
    close(io)
    rm(max_txt)

    io = open(vor_txt, "a")
    close(io)
    rm(vor_txt)

    io = open(sum_f, "a")
    close(io)
    rm(sum_f)

    #opening files

    io = open(max_txt, "a")
    io = open(vor_txt, "a")
    io = open(sum_f, "a")

    # setting txt_switch
    txt_switch = title

    println("%%%%%%%%%%%% start of calcs ==> change below %%%%%%%%%%%%%%%")
    #initializing sim
    dt = (tspan[2] - tspan[1]) / n_t
    param = [N, l0]  #parameter vector

    state_0 = vars2state(pos_0, theta_0, u_0)
    x, theta, u_0 = state2vars(state_0, N)
    println("x, theta, u_0: ", x, theta, u_0)

    f = zeros(Float64, 4 * N)

    plt = plot(aspect_ratio=:equal)
    xlims!(limits[1])
    ylims!(limits[2])
    zlims!(limits[3])
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1, x_2, x_3 = x_cur[1, :], x_cur[2, :], x_cur[3,:]
    twist_weights = twist_color(theta_0)

    # println(twist_weights)
    #initializing plot
    for i = 1:N-1
        scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                label = legend = false)
        plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                label = legend = false,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]])
    end #for loop

    title!(title)
    display(plt)
    #final plot / animation
    #     F!,
    #     f::Array{Float64,1},
    #     state::Array{Float64,1},
    #     dt::Float64,
    #     param,
    #     txt_array,
    #     err_tol,
    #     fixed_ends,
    #     constraint_func,
    #     gen_t
    # )
    function step!(gen_t)
        state = timestep_axis(F!, f, state, dt, param, txt_array, err_tol,
                        [pos_0[:,1],pos_0[:,end]], constraint_type, gen_t)
        t += dt
        println(t)
        # state_0 = vars2state(pos_0, theta_0, u_0)
        # x, theta, u_0 = state2vars(state_0, N)

        x_cur = reshape(state[4:3*(N+1)], (3, N))
        # println("theta =", state[3N+4:end])

        theta_cur = state[3*(N+1) + 1:end]
        twist_weights = twist_color(theta_cur)

        println("dt: ", t)
        println("this is theta, timsetp 1: ", theta_cur)
        println("this is x, timsetp 1: ", x_cur)
        println("************** * * * BOINK *8*** *8**88   8 ")

        # println(twist_weights)

        plt = plot(aspect_ratio=:equal)
        xlims!(limits[1])
        ylims!(limits[2])
        zlims!(limits[3])
        for i = 1:N-1
            scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                    label = legend = false)
            plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1], x_cur[3,i:i+1],
                    label = legend = false,
                    linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[i]])
        end #for

        title!(title)
    end #function

    t = 0.
    anim = @animate for i = 1:n_t
        step!(t)
    end every 1
    gif(anim, string(title,".gif"), fps = (Int(floor(n_t/5))))

    #force/torque checks
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f_bend) = ", sum(f[1:3N]))
    println("sum(f_twist) = ", sum(f[3N+1:4N]))

    display(plt)
    println("u =", state[1:3])
    png(title)

    return state
end #function

"""
    constraints(pos)

function that generates constraints array as a function of position*
    *may add additional args,
"""

function constraint_twist(x_inp, theta_inp, vor1, start1, end1, t_cur)
    n = size(vec(x_inp))[1]/3
    n = Int(n)

    # println(theta_inp)
    x_inp = reshape(x_inp,(3,n))
    edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]

    C = zeros(n-1) #1 per edge, 2 for endpoints
    undeformed_config = vor1^2 #vor length, for now

    # CONSTRAINT 1: inextensibility
    for i=1:n-1
        C[i] = dot(edges[:,i],edges[:,i]) - undeformed_config
        # println(C[i])
    end #loop

    # println(C)
    # C[n:n+2] = x_inp[:,1] - start1 #note first vertex initializes @ origin
    # C[n+3:n+5] = x_inp[:,end] - end1
    # C_fin[4:end-3] = C

    C_fin = zeros(n - 1 + 4)

    # CONSTRAINT 2: fixed ends
    # C_fin[1:3] = x_inp[:,1] - start1 #note first vertex initializes @ origin
    # C_fin[end-2:end] = x_inp[:,end] - end1
    # C_fin[1:n-1] = C

    C_fin[n] = dot(x_inp[:,1] - start1,x_inp[:,1] - start1) #note first vertex initializes @ origin
    C_fin[n+1] = dot(x_inp[:,end] - end1, x_inp[:,end] - end1)
    C_fin[n+2] = (theta_inp[1])^2
    C_fin[n+3] = (theta_inp[end] - 0.9)^2

    C_fin[1:n-1] = C

    C_fin = vec(C_fin)
    # println("Cfin = BOINK ", C_fin)
    return C_fin
end #function
# (x_inp, theta_inp, vor1, start1, end1, t_cur)

function constraint_axis(x_inp, theta_inp, vor1, start1, end1, t_cur)
    n = size(vec(x_inp))[1]/3
    n = Int(n)

    # println(theta_inp)
    x_inp = reshape(x_inp,(3,n))
    edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]

    C = zeros(n-1) #1 per edge, 2 for endpoints
    undeformed_config = vor1^2 #vor length, for now

    # CONSTRAINT 1: inextensibility
    for i=1:n-1
        C[i] = (dot(edges[:,i],edges[:,i]) - undeformed_config)
        # println(C[i])
    end #loop

    # println(C)
    # C[n:n+2] = x_inp[:,1] - start1 #note first vertex initializes @ origin
    # C[n+3:n+5] = x_inp[:,end] - end1
    # C_fin[4:end-3] = C

    C_fin = zeros(n - 1 + 4)

    # CONSTRAINT 2: fixed ends
    # C_fin[1:3] = x_inp[:,1] - start1 #note first vertex initializes @ origin
    # C_fin[end-2:end] = x_inp[:,end] - end1
    # C_fin[1:n-1] = C

    v = vor1/10 #v = velocity

    C_fin[n] = dot(x_inp[:,1] - (start1  + (v * t_cur) * [1.;0.;0.]),x_inp[:,1] - (start1  + (v * t_cur)* [1.;0.;0.])) #note first vertex initializes @ origin
    C_fin[n+1] = dot(x_inp[:,end] - (end1 - (v * t_cur) * [1.;0.;0.]), x_inp[:,end] - (end1 - (v * t_cur) * [1.;0.;0.]))
    C_fin[n+2] = (theta_inp[1])^2
    C_fin[n+3] = (theta_inp[end] - 0.9)^2

    C_fin[1:n-1] = C

    C_fin = vec(C_fin)
    # println("Cfin = BOINK ", C_fin)

    # n = size(vec(x_inp))[1]/3
    # n = Int(n)
    #
    # x_inp = reshape(x_inp,(3,n))
    # edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]
    #
    # C = zeros(n-1) #1 per edge, 2 for endpoints
    # undeformed_config = vor1^2 #vor length, for now
    #
    # # CONSTRAINT 1: inextensibility
    # for i=1:n-1
    #     C[i] = dot(edges[:,i],edges[:,i]) - undeformed_config
    #     # println(C[i])
    # end #loop
    #
    # v = vor1 / 10 #v = velocity
    # #CONSTRAINT 2: moving
    # C[n:n+2] = x_inp[:,1] - (start1 + (v*t)) #note first vertex initializes @ origin
    # C[n+1:n+3] = x_inp[:,end] - (end1 - (v*t))
    #
    # C_fin = vec(C)
    return C_fin
end #function

"""
main() function
"""

function main()

    buff = 1.
    #twisting rod

    name = "buckle_1_twisting_rod"
    dim = 3
    error_tolerance_C = 10^-6
    timespan = (0.,1.)
    num_tstep = 10
    N_v = 10
    vor = 1.
    init_pos = [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.; 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.;
                0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # init_theta = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    init_theta = [0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]
    init_norm = [0.0, 1.0, 0.0]
    lims = [(minimum(init_pos[1,:]) - buff, maximum(init_pos[1,:]) + buff),
            (minimum(init_pos[2,:]) - buff, maximum(init_pos[2,:]) + buff),
            (minimum(init_pos[3,:]) - buff, maximum(init_pos[3,:]) + buff)]

    new_state = runsim_twist(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
            init_pos, init_theta, init_norm,
            constraint_twist, lims)

    #bringing ends together
    name = "buckle_2_axial_t"
    dim = 3
    error_tolerance_C = 10^-3
    timespan = (0.,1.)
    num_tstep = 10^2
    N_v = 10
    vor = 1.
    init_pos, init_theta, init_norm = state2vars(new_state, N_v)

    runsim_axis(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
            init_pos, init_theta, init_norm,
            constraint_axis, lims)

end #func

main()
