"""
function sandbox for fast projection / projection

mass_matrix()
    - returns mass matrix of rod given config

constraint_ends_fixed()
    - returns constraint matrix given config + fixed ends

fast_project()
    - 1 fast projection step

fp_WRAP()
    - multiple fast_projection steps w/ condition (i.e. error tolerance)
"""

#NOTE: FIX UNITS!!!! WIP WIP WIP WIP WIP
function mass_matrix(
    uq, #current config, integrator
    n ## of vertices
    # l0, #vor length
    # ℳ, #total mass
    # rad_e #radius
)
    q_thet = @view uq[end-(n-1):end]
    q_int = @view uq[7:end-(n-1)-4]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3:end-(n-1)-1]

    # m_k = ℳ/n #assuming uniform rod
    # x = reshape(x_fin[1:3n], (3,n)) #reshape x, twist last row
    # theta_proj = x_fin[3n+1:end]
    # # println("theta_proj: ", theta_proj)
    # edges = x[:, 2:end] - x[:, 1:end-1] #defining edges
    # #init generalized mass matrix
    MM = zeros(4n, 4n)
    # # mass_dens = 1.
    # # rad_edge = 1.
    m_k = 1. #FIX
    MM[1:3n,1:3n] = m_k * Matrix(1.0I, 3n, 3n)
    # #moment of inertia defined on edges, cylindrical rod
    # #this is assuming i know how to integrate
    MM[3n+1:4n, 3n+1:4n] = Matrix(1.0I,n,n) #(mass_dens * rad_edge^3 * 2*pi)/3 *
    MM[1:3,1:3] .= 0.
    MM[5:6,5:6] .= 0.
    MM[3*(n-1) + 1:3*n,3*(n-1) + 1:3*n] .= 0.
    MM[3*(n-2) + 2:3*(n-1),3*(n-2) + 2:3*(n-1)] .= 0.
    return MM
end #func

function constraint_ends_fixed(
    uq, #current config, intg
    n, ## oof vertices
    l0, #vor length
    fends_bool, #whether there  are ends
    fends #position of ends
    )

    u = @view uq[1:3]
    q_thet = @view uq[end-(n-1):end]
    q_pos = @view uq[4:end-(n-1)-1]
    q_int = @view uq[7:end-(n-1)-4]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3:end-(n-1)-1]

    e_pos = q_pos[4:end] - q_pos[1:end-3]
    e_pos = reshape(e_pos,(3,n-1))
    C_fin = zeros(n - 1 + 4)
    # CONSTRAINT 1: inextensibility
    for i = 1:n-1
        eC = e_pos[:, i]
        C_fin[i] = dot(eC,eC) - l0^2
    end #loop
    # CONSTRAINT 2: fixed ends
    if fends == true
        d1st = q_1st - fends[1]
        d2nd = q_2nd - fends[2]

        C_fin[n-1+1] = 0. #fixed orientation
        C_fin[n-1+2] = 0. #fixed orientation
        C_fin[n-1+3] = (qthet[1] - fends[3])^2
        C_fin[n-1+4] = (qthet[end] - fends[4])^2
    end #cond
    return C_fin
end #func

function fast_init!(
    C, #Cvec
    MM, #massmatrix
    uq, #config, intg
    n, ##off vertices
    l0, #vor length
    cfunc, #constraint function
    fends_bool,
    fends,
    dt
    )
    #SETTINGUP / computing inverse

    u = @view uq[1:3]
    q_thet = @view uq[end-(n-1):end]
    q_pos = @view uq[4:end-(n-1)-1]
    q_int = @view uq[7:end-(n-1)-4]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3:end-(n-1)-1]

    epos = q_pos[4:end] - q_pos[1:end-3]
    epos = reshape(epos,(3,n-1))
    MMi = pinv(MM)
    C[:] = constraint_ends_fixed(uq,n,l0,fends_bool,fends)
    maxC = maximum(abs.(C))
    #SETTINGUP GRADC
    gradC = zeros(n-1 + 2 + 2,4*n)
    for r = 1:n-1
        gradC[r, 3*(r-1)+1:3*(r-1)+3] = -2 * epos[:,r]
        gradC[r, 3*(r-1)+4:3*(r-1)+6] = 2 * epos[:,r]
    end #for
    #fixed ends
    # gradC[1, 1:3] = 2 * (epos[:, 1])
    # gradC[n-1, 3*((n-1)-2)+1:3*((n-1)-2)+3] = -2 * (epos[:, n-1])
    #theta
    gradC[n+2,3n+1] = 2 * (q_thet[1] - fends[3])
    gradC[n+3,end] = 2 * (q_thet[end] - fends[4])

    gradC_t = transpose(gradC)
    G = gradC * MMi * gradC_t

    δλ = IterativeSolvers.lsmr(G, C, atol = btol = 10^-12)
    δλn = δλ / dt^2
    δxn = -dt^2 * (MMi * gradC_t * δλn)

    #x_next_fp is vec
    # println("this is δxn: ", δxn)
    # println(size(δxn))

    q_int[:] += δxn[4:3n-6+3]
    q_thet[:] += δxn[3n-6+7:end]
end #func

function fp_WRAP!(
    uq, #config, intg
    n, ##off vertices
    l0, #vor length
    cfunc, #constraint function
    fends_bool,
    fends,
    etol,
    dt
    )

    u = @view uq[1:3]
    q_thet = @view uq[end-(n-1):end]
    q_pos = @view uq[4:end-(n-1)-1]
    q_int = @view uq[7:end-(n-1)-4]
    q_1st = @view uq[4:6]
    q_2nd = @view uq[end-(n-1)-3:end-(n-1)-1]

    epos = q_pos[4:end] - q_pos[1:end-3]
    epos = reshape(epos,(3,n-1))

    MM = mass_matrix(uq,n)
    MMi = pinv(MM)
    C = constraint_ends_fixed(uq,n,l0,fends_bool,fends)
    maxC = maximum(abs.(C))

    #running alg
    it = 0
    while maxC >= etol && it < 10^3
        # println(maxC)
        fast_init!(C, MM, uq, n, l0, cfunc, fends_bool,fends,dt)
        # println(C)
        maxC = maximum(abs.(C))
        MM = mass_matrix(uq,n)
        it += 1
    end  #cond
    # println("%%%%%%%%%%%NEWTIMESTEP%%%%%%%%%%%")
    # println(maxC)
    if it > 10^3 - 1
        println("%%%%%% MAX HIT, exit %%%%%")
    end #cond
end #func
