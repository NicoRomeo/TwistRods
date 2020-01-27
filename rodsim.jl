# Stochastic Dynamics of Elastic Rods
# builds on the Discrete Elastic rods framework

using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Statistics
using NLsolve
using Random
using Plots

include("rodbase.jl")


function runstep(rod::aRod, tstep::Float64) where {T<:Number}
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

    matcg = matcurvegrad(rod)
    bf = bForce(rod, matcg)

    E = bEnergy(rod)

    #updating vertices
    ex_euler(rod, tstep, bf)

    E = bEnergy(rod)

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
    #print("\n___Rod1___\n")
    X = rand(5, 3)

    nTwist = 0.0
    B = [1.0 0.0;0.0 1.0]
    rod1 = oRod(X, B, nTwist)

    runstep(rod1)
    #print("\n___Rod2___\n")
    X = zeros(Float64, 50, 3)
    for i = 1:50
        X[i, 1] = cos(2.0 * pi * i / 50.0)
        X[i, 2] = sin(2.0 * pi * i / 50.0)
    end

    nTwist = 0.0
    B = [1.0 0.0;0.0 1.0]
    rod2 = oRod(X, B, nTwist)

    #println("energy: ")
    runstep(rod2)
    #print(rod2.kb[1])
    #print(rod2.kb[2])
    #print(rod2.kb[3])

    #print("\n___Rod3___\n")
    X = zeros(Float64, 50, 3)
    for i = 1:50
        X[i, 1] = 8.0 * cos(2.0 * pi * i / 50.0)
        X[i, 2] = 8.0 * sin(2.0 * pi * i / 50.0)
    end
    nTwist = 0.0
    B = [1.0 0.0;0.0 1.0]
    rod3 = oRod(X, B, nTwist)
    # runstep(rod3)
    # #print(rod3.kb[1])
    # #print(rod3.kb[2])
    # #print(rod3.kb[3])
    # matcg = matcurvegrad(rod3)
    # #print(size(matcg))
    # #println("Matcg")
    # #println(matcg[1,1,:,:])
    # #println(matcg[1, 2, :, :])
    # #println("End matcg")
    # f = bForce(rod3, matcg)
    # #println(f)

    gui();

    @gif for i in 1:10000
        plt = plot3d(1, xlim=(-10,10), ylim=(-10,10), zlim=(-5,5),
                    title = "rod", marker = 2)
        runstep(rod3,0.01)
        for j in 1:rod3.n
            push!(plt, rod3.X[j,1],rod3.X[j,2],rod3.X[j,3])
        end
    end every 100

    #print("\n___Rod4___\n")
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

    ln(rod4.frame[:,:,1])
    #println(rod4.kb)
    #println(rod4.matcurves)
    matcg = matcurvegrad(rod4)
    #print(size(matcg))
    #println("Matcg")
    #println(matcg[1,1,:,:])
    #println(matcg[1, 2, :, :])
    #println("End matcg")
    f = bForce(rod4, matcg)
    #println(f)

    #print("\n___Rod5___\n")
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
    rod5 = oRod(X, B, nTwist)
    # runstep(rod5)

    # #println(rod5.frame[:,:,1])
    # #println(rod5.kb)
    # #println(rod5.matcurves)
    # matcg = matcurvegrad(rod5)
    # #print(size(matcg))
    # #println("Matcg")
    # #println(matcg[1,1,:,:])
    # #println(matcg[1, 2, :, :])
    # #println("End matcg")
    # f = bForce(rod5, matcg)
    # #println(f)

    # @gif for i in 1:10
    #     plt = plot3d(1, xlim=(-2,2), ylim=(-2,2), zlim=(-2,2),
    #                 title = "rod", marker = 2)
    #     runstep(rod5,0.01)
    #     for j in 1:rod5.n
    #         push!(plt, rod5.X[j,1],rod5.X[j,2],rod5.X[j,3])
    #     end
    # end every 1

    # #println("updated vertices")
    # #println(rod5.X)
    #
    # x = X[:,1]
    # y = X[:,2]
    # z = X[:,3]
    #
    # display(plot(x,y,z, title = "test plot"))
    # #print("\n___end test___\n")

end
