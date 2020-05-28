##
## Test for the DiffEq solver
##
##


using Plots
using NLsolve

include("newrod.jl")

## Functions to access states and back
# function state2vars(state::Array{Float64,1}, n::Integer)
# function vars2state(x::Array{Float64},theta::Array{Float64},u0::Array{Float64})
##



"""
    impl(args)

implicit solver for a timestep
"""
function impl(args)

end # function

"""
    timestep(args)

function that runs a single timestep
    args: F force function
    state
"""
function timestep(F, state::Array{Float64,1})

    # Compute explicit initial guess for X, theta at t+dt

    # Using the initial gues, use a NL solver to get (X, theta) at t+dt

    # update u_0

    # package things use, and return the new state
end # function



function main()
    tspan = (0.0, 3.0)
    param = [3, 1]
    N = 4
    l0 = 1
    param = [N, l0]  #parameter vector
    pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 1.0 1.0 0.0; 4.0 1.0 2.0], (2,1))

    #straight line
    #pos_0 = permutedims([0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0], (2, 1))

    println("this is pos_0: ")
    println(pos_0)

    #pos_0 = [0.0 0.0 0.0; 0.0 1.0 4.0; 0.0 0.0 0.0]

    theta_0 = [0.0, 0.0, 0.0]
    u_0 = [1.0, 0.0, 0.0]

    state_0 = vars2state(pos_0, theta_0, u_0)
end


main()
