import Pkg
Pkg.activate("/home/ssilvest/stable_oceananigans/Oceananigans.jl/")

using Statistics
using JLD2
using Printf
using GLMakie
using Oceananigans
using Oceananigans.Units   
using Oceananigans.Utils 
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures: arch_array, device
using Oceananigans.ImmersedBoundaries
using Oceananigans.Operators
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Grids: inactive_cell

const tot_time_in_days = 10 * 360

@inline current_time_index(time, tot_time)       = mod(unsafe_trunc(Int32, time / 1days),     tot_time) + 1
@inline next_time_index(time, tot_time)          = mod(unsafe_trunc(Int32, time / 1days) + 1, tot_time) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / 1days, 1) * (u₂ - u₁)

#####
##### Grid
#####

arch = CPU()
reference_density = 1029

latitude = (-75, 75)

const Nyears  = 5
mean_climatology = true

#####  The regions for the simulation can be:
#####
#####
#####
#####

# 0.25 degree resolution
Nx = 1440
Ny = 600
Nz = 1

output_prefix = "near_global_tracer_transport_$(Nx)_$(Ny)"

#####
##### Load forcing files and inital conditions from ECCO version 4
##### https://ecco.jpl.nasa.gov/drive/files
##### Bathymetry is interpolated from ETOPO1 https://www.ngdc.noaa.gov/mgg/global/
#####

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/ss/new_hydrostatic_data_after_cleared_bugs/quarter_degree_near_global_input_data/"

datanames = ["bathymetry-1440x600"]

dh = DataDep("quarter_degree_near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    [path * data * ".jld2" for data in datanames]
)

DataDeps.register(dh)

datadep"quarter_degree_near_global_lat_lon"

files = [:file_bathymetry]
for (data, file) in zip(datanames, files)
    datadep_path = @datadep_str "quarter_degree_near_global_lat_lon/" * data * ".jld2"
    @eval $file = jldopen($datadep_path)
end

bathymetry = file_bathymetry["bathymetry"]

# Remember the convention!! On the surface a negative flux increases a positive decreases
bathymetry = arch_array(arch, bathymetry)
bathymetry_mask = Float64.(bathymetry .>= 0)

## Smoot out isolated cells


long1 = -180
long2 =  180

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 1),
                                              longitude = (long1, long2),
                                              latitude = latitude,
                                              halo = (4, 4, 4),
                                              z = (0, 1),
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(bathymetry_mask))
                 
#####
##### Load Mean prescribed velocity fields
#####

file_velocities = jldopen("prescribed_velocity_field_smoothed.jld2")

U = XFaceField(grid)
V = YFaceField(grid)

if mean_climatology
    set!(U, file_velocities["u̅"])
    set!(V, file_velocities["v̅"])
else
    @kernel function _set_velocities!(U, u₁, u₂, v₁, v₂, time)
        i, j, k = @index(Global, NTuple)
        U.u[i, j, k] = cyclic_interpolate(u₁[i, j, k], u₂[i, j, k], time)
        V.v[i, j, k] = cyclic_interpolate(v₁[i, j, k], v₂[i, j, k], time)
    end

    @inline function velocity_function(simulation)
        time = simulation.model.clock.time

        grid = simulation.model.grid
        n₁ = current_time_index(time, tot_time_in_days)
        n₂ = next_time_index(time, tot_time_in_days)

        u₁ = arch_array(arch, file_velocities["u"][n₁])
        u₂ = arch_array(arch, file_velocities["u"][n₂])
        
        v₁ = arch_array(arch, file_velocities["v"][n₁])
        v₂ = arch_array(arch, file_velocities["v"][n₂])
        
        event = launch!(arch, grid, :xyz, _set_velocities!, simulation.model.velocities, u₁, u₂, v₁, v₂, time)
        wait(device(arch), event)

        return nothing
    end
end

#####
##### Include dissipation for P?
#####

using Oceananigans.Operators: Δx, Δy
using Oceananigans.Fields: @compute
using Oceananigans.Operators: div_xyᶜᶜᶜ

@inline function κhb_dyn_new(i, j, k, grid, clock, fields, p)
    δ = div_xyᶜᶜᶜ(i, j, k, grid, fields.u, fields.v)

    return p.background_κ + p.multip * abs(δ) * max(0, fields.P[i, j, k] - 1.0)
end

closure = ScalarDiffusivity(κ = κhb_dyn_new, discrete_form = true, parameters = (background_κ = 100.0, multip = 1e8))

#####
##### Add a forcing to contrast divergence
#####

using Oceananigans.Grids: xnode, ynode

τ  = 1/1days
τδ = 1/2hours

@inline atlantic(λ, φ, z)        = exp(-((λ + 75)^2 + (φ - 30)^2) / 2^2)
@inline guinea(λ, φ, z)          = exp(-((λ + 2)^2 + φ^2) / 2^2)
@inline pacific(λ, φ, z)         = exp(-((λ + 120)^2 + φ^2) / 2^2)
@inline southern_ocean(λ, φ, z)  = exp(-((λ + 12)^2 + (φ + 53)^2) / 2^2)

@inline initial_p(λ, φ, z) = atlantic(λ, φ, z) + guinea(λ, φ, z) + pacific(λ, φ, z) + southern_ocean(λ, φ, z)

@inline function p_forcing(i, j, k, grid, clock, fields, p) 
    
    δ = div_xyᶜᶜᶜ(i, j, k, grid, fields.u, fields.v)

    λ = xnode(Center(), i, grid) 
    φ = ynode(Center(), j, grid) 

    mask = Int(p.init(λ, φ, 0) > p.limit)

    return  p.τ * mask * (p.init(λ, φ, 0) - fields.P[i, j, k]) - p.τδ * fields.P[i, j, k] * abs(δ)
end

FP = Forcing(p_final_forcing_limited_newh, discrete_form=true, parameters = (; τ, τδ,  init = initial_p, limit = 0.6))

#####
##### Setup model:
#####

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    velocities = PrescribedVelocityFields(u = U, v = V),
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                    tracers = :P,
                                    closure = nothing,
                                    forcing = (; P = FP),
                                    tracer_advection = WENO())
@info "model initialized"

#####
##### Initial condition forcing
##### 

set!(model, P = initial_p)

#####
##### Simulation setup
#####
Δt    = 30minutes  # for initialization, then we can go up to 6 minutes?

simulation = Simulation(model, Δt = Δt, stop_time = 10years)

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    @info @sprintf("Time: % 12s, iteration: %d, max(|P|): %.2e m, max(|u|): %.2e ms⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration,
                    maximum(abs, P), maximum(abs, model.velocities.u),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

if !mean_climatology
    simulation.callbacks[:interp_vel] = Callback(velocity_function, IterationInterval(1))
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

P = model.tracers.P

save_interval = 10days

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; P),
                                                              schedule = TimeInterval(save_interval),
                                                              filename = output_prefix * "_tracer",
                                                              overwrite_existing = true)

# Let's goo!
@info "Running with Δt = $(prettytime(simulation.Δt))"

run!(simulation)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Time step: $(prettytime(Δt))
"""

####
#### Plotting results
####

bat = deepcopy(arch_array(CPU(), bathymetry_mask))

using GLMakie, JLD2

surface_file = jldopen(output_prefix * "_tracer.jld2")

bat = Float64.(bat)
bat[bat .== 1] .= NaN

iterations = parse.(Int, keys(surface_file["timeseries/t"]))

iter = Observable(0)

Pi(iter) = surface_file["timeseries/P/" * string(iter)][:, :, 1]
ti(iter) = string(surface_file["timeseries/t/" * string(iter)] / day)

P = @lift Pi($iter) .+ bat

max_P = 2.0
min_P = 0

fig = Figure(resolution = (1400, 900))

ax = Axis(fig[1, 1], title="Tracer concentration (m)")
hm = GLMakie.heatmap!(ax, P, colorrange=(min_P, max_P), colormap = :solar, nan_color = :black)
cb = Colorbar(fig[1, 2], hm)

title_str = @lift "Earth day = " * ti($iter)
ax_t = fig[0, :] = Label(fig, title_str)

GLMakie.record(fig, output_prefix * ".mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

display(fig)

close(surface_file)
