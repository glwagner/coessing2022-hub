using Oceananigans
using Oceananigans.Units
using JLD2
using DataDeps
using GLMakie

include("surface_simulation_utilities.jl")

# Download bathymetry data using DataDeps
bathymetry_path = datadep"quarter_degree_data/bathymetry-1440x600.jld2"
bathymetry = jldopen(bathymetry_path)["bathymetry"]
bathymetry = Float64.(bathymetry .>= 0)

filenames = ["near_global_nonturbulent_particle_transport.jld2",
             "near_global_particle_transport.jld2"]

set_theme!(Theme(fontsize=24))
fig = Figure(resolution=(1500, 1200))
n = Observable(1)

for (i, filename) in enumerate(filenames)
    ax = Axis(fig[i, 1], xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
    xlims!(ax, -160, 100)
    ylims!(ax, -75, 25)

    # Load speed
    st = FieldTimeSeries(filename, "s")
    times = st.times
    global Nt = length(times)
    λ, φ, z = nodes(st)

    @info "Masking..."
    mask = bathymetry .> 0
    for n = 1:Nt
        ss = interior(st[n], :, :, 1)
        ss[mask] .= NaN
    end

    @info "Opening particles..."
    # Load particles
    file = jldopen(filename)
    iterations = parse.(Int, keys(file["timeseries/t"]))
    x = [file["timeseries/particles/$i"].x for i in iterations]
    y = [file["timeseries/particles/$i"].y for i in iterations]
    close(file)

    x = hcat(x...)
    y = hcat(y...)

    @info "Making figure..."

    s = @lift interior(st[$n], :, :, 1)
    heatmap!(ax, λ, φ, s, colorrange=(-0.1, 1.0), colormap=:viridis, nan_color=:black)

    Ntail = 100
    chunk = 10

    function chunked_range(n, δ, c)
        nr = UnitRange(n - δ - c, n - δ - 1)
        println(length(nr))
        return nr
    end

    for δ in Ntail-1:-4:1
        xδ = @lift $n - δ > 1 ? x[:, $n - δ] : x[:, 1]
        yδ = @lift $n - δ > 1 ? y[:, $n - δ] : y[:, 1]

        # yδ = @lift begin
        #     nr = chunked_range($n, δ, chunk)
        #     nr[1] > 1 ? y[p, nr] : y[p, 1] * ones(chunk)
        # end

        ϵ = 1 - δ / Ntail
        scatter!(ax, xδ, yδ, color=(:red, ϵ), markersize=12ϵ)
    end

    xn = @lift x[:, $n]
    yn = @lift y[:, $n]
    scatter!(ax, xn, yn, color=(:white, 0.8), markersize=15)

    timestr = @lift string("time = ", prettytime(times[$n]))
    text!(-8, 15, text=timestr, color=:white, textsize=32)
end

display(fig)

@info "Recording movie..."

record(fig, "nonturbulent_turbulent_buoyant_particle_advection.mp4", 1:Nt, framerate=24) do nn
    @info "Plotting frame $nn of $Nt..."
    n[] = nn
end

