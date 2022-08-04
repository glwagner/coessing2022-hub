using Oceananigans
using Oceananigans.Units
using JLD2
using DataDeps
using GLMakie

using Oceananigans: arch_array

include("surface_simulation_utilities.jl")
using .SurfaceSimulationUtilities: PrescribedVelocityTimeSeries, Progress

# 0.25 degree resolution
arch = CPU()
Nx = 1440
Ny = 600
Nz = 1

underlying_grid = LatitudeLongitudeGrid(arch,
                                        size = (Nx, Ny, Nz),
                                        longitude = (-180, 180),
                                        latitude = (-75, 75),
                                        halo = (4, 4, 4),
                                        z = (0, 1),
                                        precompute_metrics = true)

# Download bathymetry data using DataDeps
bathymetry_path = datadep"quarter_degree_data/bathymetry-1440x600.jld2"
bathymetry = jldopen(bathymetry_path)["bathymetry"]
bathymetry = Float64.(bathymetry .>= 0)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(bathymetry))

U = XFaceField(grid)
V = YFaceField(grid)
velocities = PrescribedVelocityFields(u=U, v=V)
                 
function instantiate_particle_positions(λ₀, φ₀;
                                        N = 10,
                                        Δ = (4, 4))

    Δλ, Δφ = Δ
    # Distribute particles in a square of dimension (Δλ, Δφ)
    λᵢ = λ₀ .+ Δλ .* (2rand(N) .- 1)
    φᵢ = φ₀ .+ Δφ .* (2rand(N) .- 1)
    zᵢ = ones(N) / 2

    return λᵢ, φᵢ, zᵢ
end

λᵢ, φᵢ, zᵢ = [], [], []

# Guinea
n = 200
λ₁ = -15 * ones(n)
φ₁ = range(start=-5, stop=5, length=n)
z₁ = zeros(n)

# Drake
n = 200
λ₂ = -100 * ones(n)
φ₂ = range(start=-70, stop=-55, length=n)
z₂ = zeros(n)

# West Pacific
#n = 40
#λ₃ = 150 * ones(n)
#φ₃ = range(start=-5, stop=20, length=n)
#z₃ = zeros(n)
λ₃, φ₃, z₃ = instantiate_particle_positions(126, 26, N=40, Δ=(2, 2))

# Southern Ocean
n = 500
λ₄ = 70 * ones(n)
φ₄ = range(start=-65, stop=-40, length=n)
z₄ = zeros(n)

# Horn of Africa
λ₅, φ₅, z₅ = instantiate_particle_positions(55, 5)

# Pacific garbage patch
λ₆, φ₆, z₆ = instantiate_particle_positions(-125, -25, N=100, Δ=(15, 15))

append!(λᵢ, λ₁, λ₂, λ₃, λ₆)
append!(φᵢ, φ₁, φ₂, φ₃, φ₆)
append!(zᵢ, z₁, z₂, z₃, z₆)

particles = LagrangianParticles(x=λᵢ, y=φᵢ, z=zᵢ)

# Note: we have to eliminate tracers to avoid time-stepping temperature and salinity.
model = HydrostaticFreeSurfaceModel(; grid, velocities, particles, tracers=(), buoyancy=nothing)

# Simulation
simulation = Simulation(model; Δt=6hours, stop_time=7years)
simulation.callbacks[:progress] = Callback(Progress(), IterationInterval(10))

# Prepare a callback that updates the prescribed velocities
velocities_file = jldopen("quarter_degree_velocity_field.jld2")

u = velocities_file["u̅"]
v = velocities_file["v̅"]

# Uncomment to use fluctuating velocities rather than mean velocities
#u = velocities_file["u"][360:720]
#v = velocities_file["v"][360:720]

velocities_time_series = PrescribedVelocityTimeSeries(u, v, δt=day)
simulation.callbacks[:update_velocities] = Callback(velocities_time_series)

u = model.velocities.u
v = model.velocities.v
s = sqrt(u^2 + v^2)
outputs = (; s, particles=model.particles)
#filename = "near_global_nonturbulent_particle_transport.jld2"
filename = "near_global_particle_transport.jld2"

#=
simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs; filename,
                                                      schedule = TimeInterval(1day),
                                                      overwrite_existing = true)

# Let's goo!
run!(simulation)

@info """
    Simulation took $(prettytime(simulation.run_wall_time))
"""
=#

# Load speed
st = FieldTimeSeries(filename, "s")
times = st.times
Nt = length(times)
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

set_theme!(Theme(fontsize=24))
fig = Figure(resolution=(1800, 1200))
ax = Axis(fig[1, 1], xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
xlims!(ax, -160, 100)
#slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = Observable(1)

s = @lift interior(st[$n], :, :, 1)
heatmap!(ax, λ, φ, s, colorrange=(-0.1, 1.0), colormap=:viridis, nan_color=:black)

#=
Visualization:

* Data is the position of the particle at each moment in time
* Marker is current time index
* Obtain vector as set of particles 
=#

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

display(fig)

@info "Recording movie..."

record(fig, "turbulent_buoyant_particle_advection.mp4", 1:Nt, framerate=24) do nn
    @info "Plotting frame $nn of $Nt..."
    n[] = nn
end
