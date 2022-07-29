using Oceananigans
using Oceananigans.Units
using JLD2
using DataDeps
using GLMakie

using Oceananigans: arch_array

include("surface_simulation_utilities.jl")
using .SurfaceSimulationUtilities: PrescribedVelocityTimeSeries, progress

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
                 
# Particles
λ₀ = -90
φ₀ = -60
Nparticles = 1000

Δλ = 4
Δφ = 4

# Distribute particles in a square of dimension (Δλ, Δφ)
λₚ = λ₀ .+ Δλ .* (2rand(Nparticles) .- 1)
φₚ = φ₀ .+ Δφ .* (2rand(Nparticles) .- 1)

λₚ = arch_array(arch, λₚ)
φₚ = arch_array(arch, φₚ)
zₚ = arch_array(arch, ones(Nparticles) / 2)

particles = LagrangianParticles(x=λₚ, y=φₚ, z=zₚ)

# Note: we have to eliminate tracers to avoid time-stepping temperature and salinity.
model = HydrostaticFreeSurfaceModel(; grid, velocities, particles, tracers=(), buoyancy=nothing)
@show model

# Simulation
simulation = Simulation(model; Δt=6hours, stop_time=35year)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# Prepare a callback that updates the prescribed velocities
velocities_file = jldopen("quarter_degree_velocity_field.jld2")

u = velocities_file["u̅"]
v = velocities_file["v̅"]

# Uncomment to use fluctuating velocities rather than mean velocities
u = velocities_file["u"][1:1800]
v = velocities_file["v"][1:1800]

velocities_time_series = PrescribedVelocityTimeSeries(u, v, δt=day)
simulation.callbacks[:update_velocities] = Callback(velocities_time_series)

u = model.velocities.u
v = model.velocities.v
s = sqrt(u^2 + v^2)
outputs = (; s, particles=model.particles)
filename = "near_global_particle_transport.jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs; filename,
                                                      schedule = TimeInterval(30days),
                                                      overwrite_existing = true)

# Let's goo!
run!(simulation)

@info """
    Simulation took $(prettytime(simulation.run_wall_time))
"""

st = FieldTimeSeries(filename, "s")

s = interior(st[1], :, :, 1)
mask = bathymetry .> 0
s[mask] .= NaN
λ, φ, z = nodes(st)

file = jldopen(filename)
iterations = parse.(Int, keys(file["timeseries/t"]))
x = [file["timeseries/particles/$i"].x for i in iterations]
y = [file["timeseries/particles/$i"].y for i in iterations]
close(file)

x = hcat(x...)
y = hcat(y...)

fig = Figure(resolution=(2400, 1200))
ax = Axis(fig[1, 1])

heatmap!(ax, λ, φ, s, colorrange=(-0.5, 0.5), colormap=:solar, nan_color=:black)

for p in 1:Nparticles
    xp = x[p, :]
    yp = y[p, :]
    lines!(ax, xp, yp, color=(:blue, 0.2))
end

display(fig)

