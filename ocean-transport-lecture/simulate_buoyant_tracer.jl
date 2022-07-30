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
tracer_advection = WENO()
closure = ScalarBiharmonicDiffusivity(κ=100)
                 
# Note: we have to eliminate tracers to avoid time-stepping temperature and salinity.
model = HydrostaticFreeSurfaceModel(; grid, velocities, closure, buoyancy=nothing,
                                    tracers=:c, tracer_advection)
@show model

cᵢ(λ, φ, z) = φ
set!(model, c=cᵢ)

# Simulation
simulation = Simulation(model; Δt=2hours, stop_time=1year)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# Prepare a callback that updates the prescribed velocities
velocities_file = jldopen("quarter_degree_velocity_field.jld2")

u = velocities_file["u̅"]
v = velocities_file["v̅"]

# Uncomment to use fluctuating velocities rather than mean velocities
#u = velocities_file["u"][1:1800]
#v = velocities_file["v"][1:1800]

velocities_time_series = PrescribedVelocityTimeSeries(u, v, δt=day)
simulation.callbacks[:update_velocities] = Callback(velocities_time_series)

c = model.tracers.c
u = model.velocities.u
v = model.velocities.v
s = sqrt(u^2 + v^2)
outputs = (; s, c)
filename = "near_global_tracer_transport.jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs; filename,
                                                      schedule = TimeInterval(1day),
                                                      overwrite_existing = true)

# Let's goo!
run!(simulation)

@info """
    Simulation took $(prettytime(simulation.run_wall_time))
"""

fig = Figure(resolution=(2400, 1200))
ax = Axis(fig[2, 1])

st = FieldTimeSeries(filename, "s")
ct = FieldTimeSeries(filename, "c")

times = ct.times
Nt = length(times)

# Mask
mask = bathymetry .> 0
for n in 1:Nt
    cc = interior(ct[n], :, :, 1)
    cc[mask] .= NaN
end

slider = Slider(fig[3, 1], range=1:Nt, startvalue=1)
n = slider.value

cn = @lift interior(ct[$n], :, :, 1)
λ, φ, z = nodes(ct)
heatmap!(ax, λ, φ, cn, colorrange=(-75, 75), colormap=:redblue, nan_color=:black)

title = @lift string("Tracer mixing after t = ", prettytime(times[$n]))
Label(fig[1, 1], title, tellwidth=false)

display(fig)

record(fig, "buoyant_tracer.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end

