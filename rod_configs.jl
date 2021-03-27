"""
ROD CONFIGS
    returns q and vor length
    - spiral(): open spiral, configed. to be open circle
    - twist_rod(): twisted rod
    - closed_loop(): closed loop, WIP
"""

function spiral(v;rad = 1.)
    X = zeros(Float64,3,v)
    ne = 2 * pi/v
    for i = 1:v
        X[1,i] = rad * sin(i * ne)
        X[2,i] = rad * cos(i * ne)
        X[3,i] = 0.
    end #loop
    vor = sqrt(dot(X[:,2] - X[:,1],X[:,2] - X[:,1]))
    return vec(X), vor
end #func

function twist_rod(v,rad_tot)
    X = zeros(Float64,3,v)
    rad = rad_tot/(v-1)
    for i = 1:v
        X[1,i] = rad * (i-1)
    end #loop
    return vec(X), rad
end #func

function closed_loop()
end  #func
