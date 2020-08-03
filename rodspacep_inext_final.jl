##
## Test for the DiffEq solver
##
##

##
## ISSUES
# 1. Fix coloring
#
##
## CURRENTLY
# check "Resolving Issues.txt"
#
## CODE BELOW
#
#

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
    timestep(args)

function that runs a single timestep using a RK2 time step.
    args: F force function
    state

"""
timestep(F!, f, state, dt, param, i, txt_array, err_tol)
function timestep(
    F!,
    f::Array{Float64,1},
    state::Array{Float64,1},
    dt::Float64,
    param,
    txt_array,
    err_tol
)
    # unpack state
    n = Int(param[1])
    l0 = param[2]
    x, theta, u0 = state2vars(state, n)
    init_state = state

    q_i = vcat(vec(x), theta)
    tangent0 = LinearAlgebra.normalize!(q_i[4:6] - q_i[1:3])

    # Compute explicit initial guess for X, theta at t+dt

    #RK 2
    F!(f, q_i, u0, param)

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

    F!(f, q1, u1, param)
    k2 = f
    q2 = (dt / 2) * f + q1
    e2 = LinearAlgebra.normalize!(q2[4:6] - q2[1:3])
    u2 = rotab(tangent0, e2, u0)

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

    x_fin = q_i + (1/6 * dt * (k1 + 2k2 + 2k3 + k4))
    e_fin = LinearAlgebra.normalize!(x_fin[4:6] - x_fin[1:3])
    u_fin = rotab(tangent0, e_fin, u0)

    # println("this is ufin, xfin pre-projection")
    init = vcat(u_fin,x_fin)
    # println(vcat(u_fin,x_fin))

    """
    PSEUDOCODE

    pt. 1: unconstrained step
    - normal step (DONE)

    pt. 2: projecting onto constraint manifold
    - initialize M, mass matrix ==> (3n + 6) x (3n + 6)
    - initialize C(x_j), constraint  matrix
    - initialize C(x_j)^T, transpose of constraint matrix

    - check h (time-step size)
    - solve linear system (ODE solver? do this tomorrow, 7/1)

    """
    j =  0
    # x = x_fin

    #lumped model
    #init mass
    ℳ = n #total scalar mass
    m_k = ℳ/n #assuming uniform rod

    x = reshape(x_fin[1:3n], (3,n)) #reshape x, twist last row
    # println("this is x")
    # println(x)

    theta_proj = x_fin[3n+1:end]

    edges = x[:, 2:end] - x[:, 1:end-1] #defining edges

    #mass matrix: uniform, lumped rod, CHECK THISß
    M = m_k * Matrix(1.0I, 3n, 3n)

    # #init generalized mass matrix
    # M = zeros(3n + 6, 3n + 6)

    #computing inverse
    #check if faster way to do this
    M_inv = inv(M)

    #initializing C as func

    function constraint(x_inp)
        x_inp = reshape(x_inp,(3,n))
        edges = x_inp[:, 2:end] - x_inp[:, 1:end-1]

        C = zeros(n-1) #1 per edge
        undeformed_config = l0^2 #vor length, for now

        #note major assumption: vor length determines undeformed config
        for i=1:n-1
            C[i] = dot(edges[:,i],edges[:,i]) - undeformed_config
            # println(C[i])
        end #loop

        C_fin = vec(C)
        return C_fin
    end #function

    function vor_length(x_inp)
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

    #initializing y (keep track of velocity), q, CoM
    # CoM_x = zeros(1,4)
    #
    # for i=1:n
    #     CoM_x += x[:,i]
    # end #loop

    #CoM should not change

    # CoM_x = (1/n) * CoM_x #center of mass
    #
    # r = CoM_x[1:3] - ref[1:3]
    # dqdt = Zygote.gradient(q, t)[1]
    # drdt = Zygote.gradient(r, t)[1]
    # dxdt = Zygote.gradient(x, t)[1]

    #initializing v and y
    # v = [dqdt; drdt; dxdt]
    # y = [q_inv * dqdt; drdt; dxdt]
    #minimizing the objective function W
    #in Bergou, 2008:
    #yM̃y - C^Tλ using Goldenthal, 2007 notation

    #init C
    C = constraint(x)
    maxC = maximum(C)

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
    while maxC >= 10^-4 || maxC <= -10^-4
        #vectorize
        x = vec(x)

        #initializing gradC
        gradC = zeros(n-1,3*n)

        #calculating gradient
        for r = 1:n-1
            gradC[r, 3*(r-1)+1:3*(r-1)+3] = -2 * edges[:,r]
            gradC[r, 3*(r-1)+4:3*(r-1)+6] = 2 * edges[:,r]
        end #for

        # gradC = Zygote.gradient(constraint, x)

        #solving for lagrange multipliers
        # println("gradC")
        # println(gradC)
        gradC_t = transpose(gradC)
        G = gradC * M_inv * gradC_t

        # println("G", G)
        #note: G is sparse. Not quite banded (note staggered rows)
        G_inv = inv(G) #check runtime

        δλ = G_inv * C
        δλ_next = δλ / dt^2
        δx_next = -dt^2 * (M_inv * gradC_t * δλ_next)

        #x_next_fp is vec
        x_next_fp = x + δx_next
        j += 1

        iteration += 1
        #_______________update_y/v__________________
        # q0 = ones(3,3) #no rotation
        # r0 = r0 #check timestep func!!!
        # x0 = x0 #check timestep func!!!
        #
        # v -= (1/dt)*[2q0*q_inv, r0 - r, x0 - x]
        #how to update y?

        #______________re_init___________________
        #updating x, edges, C
        x = x_next_fp
        x_arr = reshape(x,(3,n))

        edges = x_arr[:, 2:end] - x_arr[:, 1:end-1] #defining edges

        #mass matrix: uniform, lumped rod
        M = m_k * Matrix(1.0I, 3n, 3n)
        #computing inverse
        M_inv = inv(M)

        #initializing C
        C = constraint(x)
        maxC = maximum(C)

        #initializing vor length
        vor_len = vor_length(x)

        #writing to file
        # println("maxC: ", maxC)

        #writing to text file

        io = open(txt_array[1], "a")
        writedlm(io, C)
        write(io, "* \n")
        close(io)

        io = open(txt_array[2], "a")
        writedlm(io, "* ", vor_len)
        close(io)

        # println("this is C:", C)
        # println("this is max C:", maximum(C))

        #initializing y, q, CoM
        # CoM_x = zeros(1,4)
        #
        # for i=1:n
        #     CoM_x += x[:,i]
        # end #loop
        #
        # CoM_x = (1/n) * CoM_x #center of mass
        # q = zero(3,3) #not correct, fix to rotation matrix
        # q_inv = inv(q)
        #
        # r = CoM_x[1:3] - ref[1:3]
        # dqdt = Zygote.gradient(q, t)[1]
        # drdt = Zygote.gradient(r, t)[1]
        # dxdt = Zygote.gradient(x, t)[1]

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
    #
    # println("u_fin dim")
    # println(size(u_fin_proj))
    #
    # println("x_fin dim")
    # println(size(x_fin_proj))

    # println("fin_proj", fin_proj)
    # println("delta", fin_proj - init)

    #calculating final f
    f_prop = (fin_proj - init_state)/dt

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

    return fin_col
end #function

function runsim(
    title,
    dims,
    err_tol,
    tspan,
    n_t,
    N,
    l0,
    pos_0,
    theta_0,
    u_0
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
    # txt_switch = title

    #initializing sim
    dt = (tspan[2] - tspan[1]) / n_t
    #param = [3, 1]

    param = [N, l0]  #parameter vector

    println("this is pos_0: ")
    println(pos_0)

    state_0 = vars2state(pos_0, theta_0, u_0)

    println("state 0")
    println(state_0)
    x, theta, u_0 = state2vars(state_0, N)
    println("x, theta, u_0")
    println(x)
    println(theta)
    println(u_0)

    f = zeros(Float64, 4 * N)

    plt = plot()
    state = state_0[:]
    x_cur = reshape(state[4:3*(N+1)], (3, N))
    x_1 = x_cur[1, :]
    x_2 = x_cur[2, :]
    twist_weights = twist_color(theta_0)
    fin_twist_weights = zeros(N-1)

    for i = 1:N-1
        hand_switch = 0

        avg1 = (twist_weights[i] + twist_weights[i+1])  / 2
        avg2 = mod(avg1 + 0.5,1)

        delta_avg1 = abs(avg1 - (twist_weights[i+1]))
        delta_avg2 = abs(avg2 - (twist_weights[i+1]))

        if delta_avg1 < delta_avg2
            fin_twist_weights[i] = avg1
        else
            fin_twist_weights[i] = avg2
        end #conditional

    end #loop

    #initializing plot
    for i = 1:N-1
        scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false)
        plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                label = legend = false,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[fin_twist_weights[i]])
    end #for loop

    title!(title)
    # display(plt)

    #final plot / animation ==> ISSUE ==> make animation
    function step!()
        state = timestep(F!, f, state, dt, param, txt_array, err_tol)

        x_cur = reshape(state[4:3*(N+1)], (3, N))
        # println("theta =", state[3N+4:end])

        theta_cur = state[3*(N+1) + 1:end]
        twist_weights = twist_color(theta_cur)

        for i = 1:N-1
            hand_switch = 0

            avg1 = (twist_weights[i] + twist_weights[i+1])  / 2
            avg2 = mod(avg1 + 0.5,1)

            #FIX BELOW
            delta_avg1 = abs(avg1 - (twist_weights[i+1]))
            delta_avg2 = abs(avg2 - (twist_weights[i+1]))

            if delta_avg1 < delta_avg2
                fin_twist_weights[i] = avg1
            else
                fin_twist_weights[i] = avg2
            end #conditional
        end #for

        plt = plot()

        scatter!(plt, x_cur[1,1], x_cur[2,1],
                label = legend = false,
                linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[twist_weights[1]])

        for i = 1:N-1
            scatter!(plt, x_cur[1,i+1], x_cur[2,i+1],
                    label = legend = false)
            plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
                    label = legend = false,
                    linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[fin_twist_weights[i]])
        end #for

        title!(title)
    end #function

    anim = @animate for i = 1:n_t
        step!()
    end every 1
    gif(anim, string(title,".gif"), fps = 25)

    println(f)
    println("check sum ", isapprox(sum(f), 0., atol=1e-4))
    println("sum(f) = ", sum(f))
    println("sum(f_bend) = ", sum(f[1:3N]))
    println("sum(f_twist) = ", sum(f[3N+1:4N]))

    display(plt)
    println("u =", state[1:3])
    png(title)

    # for i = 1:n_t
    #     state = timestep(F!, f, state, dt, param, i, txt_array, err_tol)
    #
    #     x_cur = reshape(state[4:3*(N+1)], (3, N))
    #     # println("theta =", state[3N+4:end])
    #
    #     theta_cur = state[3*(N+1) + 1:end]
    #     twist_weights = twist_color(theta_cur)
    #
    #     for i = 1:N-1
    #         hand_switch = 0
    #
    #         avg1 = (twist_weights[i] + twist_weights[i+1])  / 2
    #         avg2 = mod(avg1 + 0.5,1)
    #
    #         #FIX BELOW
    #         delta_avg1 = abs(avg1 - (twist_weights[i+1]))
    #         delta_avg2 = abs(avg2 - (twist_weights[i+1]))
    #
    #         if delta_avg1 < delta_avg2
    #             fin_twist_weights[i] = avg1
    #         else
    #             fin_twist_weights[i] = avg2
    #         end #conditional
    #
    #     end #loop
    #
    #     for i = 1:N-1
    #         scatter!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
    #                 label = legend = false)
    #         plot!(plt, x_cur[1,i:i+1], x_cur[2,i:i+1],
    #                 label = legend = false,
    #                 linecolor = ColorSchemes.cyclic_mygbm_30_95_c78_n256_s25[fin_twist_weights[i]])
    #     end #for
    # end #for loop

    #force/torque checks



end #function

function main()

    #twist_straight
    name = "twist_straight"
    dim = 2
    error_tolerance_C = 10^-6
    timespan = (0.,5.)
    num_tstep = 200
    N_v = 5
    vor = 1.
    init_pos = [0. 1. 2. 3. 4.; 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0.]
    init_theta = [0., 1.0, 2.0, 3., 4.]
    init_norm = [0.0, 1.0, 0.0]

    runsim(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
            init_pos, init_theta, init_norm)

    #bend w/o twist
    name = "bend_wo_twist"
    dim = 2
    error_tolerance_C = 10^-6
    timespan = (0.,5.)
    num_tstep = 1000
    N_v = 3
    vor = 1.
    init_pos = [1. 0. 0.; 0. 0. 1.; 0. 0. 0.]
    init_theta = [0., 0., 0.]
    init_norm = [0.0, 1.0, 0.0]

    runsim(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
            init_pos, init_theta, init_norm)

    #bend w/ twist
    name = "bend_w_twist"
    dim = 2
    error_tolerance_C = 10^-6
    timespan = (0.,5.)
    num_tstep = 1000
    N_v = 3
    vor = 1.
    init_pos = [1. 0. 0.; 0. 0. 1.; 0. 0. 0.]
    init_theta = [0., 1.0, 2.0]
    init_norm = [0.0, 1.0, 0.0]

    runsim(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
            init_pos, init_theta, init_norm)

    #circle w/o twist
    name = "circle_wo_twist"
    dim = 2
    error_tolerance_C = 10^-6
    timespan = (0.,10.)
    num_tstep = 100
    N_v = 10

    init_pos = zeros(Float64, 3, 10)
    for i = 1:10
        init_pos[1, i] = 10 * cos(-2.0 * pi * (i - 1) / 10)
        init_pos[2, i] = 10 * sin(-2.0 * pi * (i - 1) / 10)
    end #for loop

    e_0 = normalize(init_pos[:, 2] - init_pos[:, 1])
    l_i = init_pos[:, 2] - init_pos[:, 1]
    vor = sqrt(l_i[1]^2 + l_i[2]^2)

    init_theta = zeros(Float64, N_v)
    init_norm = [e_0[2], -e_0[1], 0.0]

    runsim(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
            init_pos, init_theta, init_norm)

    #circle w/ twist
    name = "circle_w_twist"
    dim = 2
    error_tolerance_C = 10^-6
    timespan = (0.,10.)
    num_tstep = 100
    N_v = 10

    init_pos = zeros(Float64, 3, 10)
    for i = 1:10
        init_pos[1, i] = 10 * cos(-2.0 * pi * (i - 1) / 10)
        init_pos[2, i] = 10 * sin(-2.0 * pi * (i - 1) / 10)
    end #for loop

    e_0 = normalize(init_pos[:, 2] - init_pos[:, 1])
    l_i = init_pos[:, 2] - init_pos[:, 1]
    vor = sqrt(l_i[1]^2 + l_i[2]^2)

    init_theta = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]
    init_norm = [e_0[2], -e_0[1], 0.0]

    runsim(name, dim, error_tolerance_C, timespan, num_tstep, N_v, vor,
            init_pos, init_theta, init_norm)

end

main()
