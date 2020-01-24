# Stochastic Dynamics of Elastic Rods
# builds on the Discrete Elastic rods framework

using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Statistics
using NLsolve
using Random


include("rodbase.jl")


function runstep(rod::aRod) where {T<:Number}
    # update Geometry and associated quantitites: midpoints, voronoi domains, ..

    midpoints(rod)
    vDom(rod)

    #below is stuff I added
    #chi only uses rod.edges
    chi(rod)

    #kb uses chi for calcs
    kb(rod)

    #ttilda + dtilda uses chi for calcs
    ttilda(rod)
    dtilda(rod)

    #uses dtilda, chi, kb
    matcurve(rod)

    # bending force
    E = bEnergy(rod)
    println(E)
    # compute new frame, compute twisting force
    #end of stuff I added

end


function datapoint(rod::aRod, measure::Measure)

end #function


function runsim(sim::Simulation)
    rod = sim.rod
    param = sim.param

    param_array = Array{Any}(undef, 5)
    N = get(sim.param, "N")
    for i in range(N)
        runstep(rod, param_array)
        if i%n_measure == 0
            datapoint(rod, sim.measure)
        end #if
    end #for
end #function


function main()
    print("\n___Rod1___\n")
    X = rand(5, 3)

    nTwist = 0.0
    B = [1.0 0.0;0.0 1.0]
    rod1 = oRod(X, B, nTwist)

    runstep(rod1)
    print("\n___Rod2___\n")
    X = zeros(Float64, 50, 3)
    for i = 1:50
        X[i, 1] = cos(2.0 * pi * i / 50.0)
        X[i, 2] = sin(2.0 * pi * i / 50.0)
    end

    nTwist = 0.0
    B = [1.0 0.0;0.0 1.0]
    rod2 = oRod(X, B, nTwist)

    println("energy: ")
    runstep(rod2)
    print(rod2.kb[1])
    print(rod2.kb[2])
    print(rod2.kb[3])

    print("\n___Rod3___\n")
    X = zeros(Float64, 50, 3)
    for i = 1:50
        X[i, 1] = 8.0 * cos(2.0 * pi * i / 50.0)
        X[i, 2] = 8.0 * sin(2.0 * pi * i / 50.0)
    end
    nTwist = 0.0
    B = [1.0 0.0;0.0 1.0]
    rod3 = oRod(X, B, nTwist)
    runstep(rod3)
    print(rod3.kb[1])
    print(rod3.kb[2])
    print(rod3.kb[3])
    matcg = matcurvegrad(rod3)
    print(size(matcg))
    println("Matcg")
    println(matcg[1,1,:,:])
    println(matcg[1, 2, :, :])
    println("End matcg")
    f = bForce(rod3, matcg)
    println(f)

    print("\n___Rod4___\n")
    X = zeros(Float64, 3, 3)
    phi = pi/2
    radius = 1.0

    X[1, 1] = radius
    X[1, 2] = 0

    X[2, 1] = 0
    X[2, 2] = 0

    X[3, 1] = radius * cos(phi)
    X[3, 2] = radius * sin(phi)


    nTwist = 0.0
    B = [1.0 0.0;0.0 1.0]
    rod4 = oRod(X, B, nTwist)
    runstep(rod4)

    println(rod4.frame[:,:,1])
    println(rod4.kb)
    println(rod4.matcurves)
    matcg = matcurvegrad(rod4)
    print(size(matcg))
    println("Matcg")
    println(matcg[1,1,:,:])
    println(matcg[1, 2, :, :])
    println("End matcg")
    f = bForce(rod4, matcg)
    println(f)

    print("\n___end test___\n")

end
