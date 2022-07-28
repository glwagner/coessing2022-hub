using Oceananigans
using Oceananigans.Units   
using Oceananigans.Utils 

using JLD2
using GLMakie
using Statistics
using Printf

using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: arch_array, device
using Oceananigans.ImmersedBoundaries
using Oceananigans.Operators

@inline function progress(sim)
    @info @sprintf("Time: % 12s, iteration: %d, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration,
                    prettytime(sim.run_wall_time))

    return nothing
end

@inline current_time_index(time, tot_time)       = mod(unsafe_trunc(Int32, time / 1days),     tot_time) + 1
@inline next_time_index(time, tot_time)          = mod(unsafe_trunc(Int32, time / 1days) + 1, tot_time) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / 1days, 1) * (u₂ - u₁)


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

#####
##### Grid
#####

bathymetry_path = @datadep_str "quarter_degree_near_global_lat_lon/bathymetry-1440x600.jld2"
bathymetry = jldopen(datadep_path)["bathymetry"]
bathymetry = arch_array(arch, bathymetry)
bathymetry_mask = Float64.(bathymetry .>= 0)

# 0.25 degree resolution
arch = CPU()
Nx = 1440
Ny = 600
Nz = 1

@show underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = (-75, 75),
                                              halo = (4, 4, 4),
                                              z = (0, 1),
                                              precompute_metrics = true)

output_prefix = "near_global_particle_transport_$(Nx)_$(Ny)"

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(bathymetry_mask))
                 
#####
##### Load Mean prescribed velocity fields
#####

velocities_file = jldopen("prescribed_velocity_field.jld2")

const Ū = velocities_file["u̅"]
const V̄ = velocities_file["v̅"]

const u = velocities_file["u"][1:1800]
const v = velocities_file["v"][1:1800]

const vel_idx  = length(u)
const tot_time = length(u)

U = XFaceField(grid)
V = YFaceField(grid)

#####
##### Include Particles
#####

function run_particle_simulation!(λ₀, φ₀, n_particles; degree_spread_λ=1, degree_spread_φ=1, δ_turb=0, Nyears=1)
    
    ##### 
    ##### Setting particle's initial position
    #####

    λₚ = λ₀ .+ degree_spread_λ .* (rand(n_particles) .- 0.5);
    φₚ = φ₀ .+ degree_spread_φ .* (rand(n_particles) .- 0.5);

    λₚ = arch_array(arch, λₚ)
    φₚ = arch_array(arch, φₚ)

    zₚ = arch_array(arch, 0.5 .* ones(n_particles));
    lagrangian_particles = LagrangianParticles(x=λₚ, y=φₚ, z=zₚ)

    #####
    ##### Velocity depending on δ_turb
    #####

    if δ_turb == 0
        set!(U, Ū)
        set!(V, V̄)
    else
        uₙ = deepcopy(u)
        vₙ = deepcopy(v)

        for i in 1:vel_idx
            @info "$i of $vel_idx"
            uₙ[i] = δ_turb .* u[i] .+ (1 - δ_turb) .* Ū
            vₙ[i] = δ_turb .* v[i] .+ (1 - δ_turb) .* V̄
        end

        @kernel function _set_velocities!(u_mod, v_mod, u₁, u₂, v₁, v₂, time)
            i, j, k = @index(Global, NTuple)
            u_mod[i, j, k] = cyclic_interpolate(u₁[i, j, k], u₂[i, j, k], time)
            v_mod[i, j, k] = cyclic_interpolate(v₁[i, j, k], v₂[i, j, k], time)
        end

        @inline function velocity_function(simulation)
            time = simulation.model.clock.time

            grid = simulation.model.grid
            n₁ = current_time_index(time, tot_time)
            n₂ = next_time_index(time, tot_time)

            u₁ = arch_array(arch, uₙ[n₁])
            u₂ = arch_array(arch, uₙ[n₂])
            
            v₁ = arch_array(arch, vₙ[n₁])
            v₂ = arch_array(arch, vₙ[n₂])
            
            u_mod = simulation.model.velocities.u
            v_mod = simulation.model.velocities.v

            event = launch!(arch, grid, :xyz, _set_velocities!, u_mod, v_mod, u₁, u₂, v₁, v₂, time)
            wait(device(arch), event)

            return nothing
        end
    end

    #####
    ##### Setup model:
    #####

    model = HydrostaticFreeSurfaceModel(; grid,
                                        velocities = PrescribedVelocityFields(u = U, v = V),
                                        coriolis = nothing,
                                        tracers = (),
                                        buoyancy = nothing,
                                        closure = nothing,
                                        particles = lagrangian_particles)
    @show model

    #####
    ##### Simulation setup
    #####

    simulation = Simulation(model; Δt=6hours, stop_time=Nyears*years)

    if δ_turb != 0 
        simulation.callbacks[:interp_vel] = Callback(velocity_function, IterationInterval(1))
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    u, v, w = model.velocities
    simulation.output_writers[:surface_fields] = JLD2OutputWriter(model,  (; u, v, particles=model.particles),
                                                                schedule = TimeInterval(10days),
                                                                filename = output_prefix * "_tracer",
                                                                overwrite_existing = true)

    # Let's goo!
    @info "Running with Δt = $(prettytime(simulation.Δt))"

    run!(simulation)

    @info """

        Simulation took $(prettytime(simulation.run_wall_time))
        Time step: $(prettytime(Δt))
    """
    return model
end

####
#### Plotting results
####

function visualize_results(output_prefix)
    bat = deepcopy(arch_array(CPU(), bathymetry_mask))

    surface_file = jldopen(output_prefix * "_tracer.jld2")

    bat = Float64.(bat)
    bat[bat .== 1] .= NaN

    iterations = parse.(Int, keys(surface_file["timeseries/t"]))

    iter = Observable(0)

    xi(iter) = surface_file["timeseries/particles/" * string(iter)].x
    yi(iter) = surface_file["timeseries/particles/" * string(iter)].y
    ti(iter) = string(surface_file["timeseries/t/" * string(iter)] / day)

    ui(iter) = surface_file["timeseries/u/" * string(iter)][:, :,       1]
    vi(iter) = surface_file["timeseries/v/" * string(iter)][:, 1:end-1, 1]
    
    Px = @lift Array(xi($iter))
    Py = @lift Array(yi($iter))

    x₀ = []
    y₀ = []
    for j in 1:length(iterations)
        push!(x₀, xi(iterations[j]))
        push!(y₀, yi(iterations[j]))
    end

    speed = @lift (Array(ui($iter)).^2 .+ Array(vi($iter)).^2).^(0.5) .+ bat

    fig = Figure(resolution = (1400, 900))

    λ = range(-179.75, 179.75, length = 1440)
    φ = range(-74.75, 74.75, length = 600)

    ax = Axis(fig[1, 1], title="Tracer concentration (m)")
    hm = GLMakie.heatmap!(ax, λ, φ, speed, colorrange=(-0.5, 0.5), colormap = :solar, nan_color = :black)
    GLMakie.scatter!(ax, Px, Py, color = :red)

    GLMakie.record(fig, output_prefix * ".mp4", 1:length(iterations), framerate=8) do i
        @info "Plotting iteration $i of $(length(iterations))..."
        if i > 1
            GLMakie.scatter!(ax, x₀[i-1], y₀[i-1], color = :white, markersize = 4)
        end
        iter[] = iterations[i]
    end

    display(fig)

    close(surface_file)
end
