module SurfaceSimulationUtilities

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

# Interpolation utilities
@inline current_time_index(t, T, δt)    = mod(unsafe_trunc(Int32, t / δt), T) + 1
@inline    next_time_index(t, T, δt)    = mod(unsafe_trunc(Int32, t / δt) + 1, T) + 1
@inline cyclic_interpolate(a, b, t, δt) = a + mod(t / δt, 1) * (b - a)

@kernel function _interpolate_velocities!(u, v, u₁, v₁, u₂, v₂, t, δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u[i, j, k] = cyclic_interpolate(u₁[i, j, k], u₂[i, j, k], t, δt)
        v[i, j, k] = cyclic_interpolate(v₁[i, j, k], v₂[i, j, k], t, δt)
    end
end

struct PrescribedVelocityTimeSeries{U, V, T}
    u_time_series :: U
    v_time_series :: V
    times :: T
    time_interval :: Float64
    total_time :: Float64
end

"""
    PrescribedVelocityTimeSeries(u, v, δt=0.0)

Return a callable object that can be used to prescribe a time-series of velocities.
If `u, v` are  `Vector` and therefore time series, `δt` is the time interval.
the time points of the time series are then assumed to span from 0 to `(length(u)-1) * δt`.

if `u, v` are `Array`, they are assumed to represent stationary / non-evolving
velocity fields, and `δt` is irrelevant.

Example
=======

```julia
# Create `u, v`, then build a time series with interval of 1 day
velocities_time_series = PrescribedVelocityTimeSeries(u, v, δt=day)

# Add the time-series as a `Callback` to update `simulation.model.velocities`
# every iteration.
simulation.callbacks[:update_velocities] = Callback(velocities_time_series)
```
"""
function PrescribedVelocityTimeSeries(u, v; δt=0.0)
    if u isa Vector
        Nu = length(u)
        T = (Nu - 1) * δt
        times = range(0, step=δt, length=Nu)
    else
        T = 0.0
        times = nothing
    end

    return PrescribedVelocityTimeSeries(u, v, times, δt, T)
end

# functor to be used as a Callback
function (time_series::PrescribedVelocityTimeSeries)(simulation)
    model = simulation.model
    grid = model.grid
    arch = grid.architecture

    ut = time_series.u_time_series
    vt = time_series.v_time_series

    u = model.velocities.u
    v = model.velocities.v

    if ut isa Vector # otherwise, "ut" is not a time series
        t = time(simulation)
        T = time_series.total_time
        δt = time_series.time_interval
        
        # Extract velocity data on either side of the time interval
        n₁ = current_time_index(t, T, δt)
        n₂ =    next_time_index(t, T, δt)

        n₁ = Int(n₁)
        n₂ = Int(n₂)

        u₁ = ut[n₁]
        u₂ = ut[n₂]
        v₁ = vt[n₁]
        v₂ = vt[n₂]

        event = launch!(arch, grid, :xyz, _interpolate_velocities!,
                        u, v, u₁, v₁, u₂, v₂, t, δt)

        wait(device(arch), event)

    elseif iteration(simulation) == 0
        set!(u, ut)
        set!(v, vt)
    end

    return nothing
end


#####
##### Forcing files and inital conditions from ECCO version 4
##### https://ecco.jpl.nasa.gov/drive/files
##### Bathymetry is interpolated from ETOPO1 https://www.ngdc.noaa.gov/mgg/global/
#####

using DataDeps

bathymetry_url = "https://github.com/CliMA/OceananigansArtifacts.jl" *
                 "/raw/ss/new_hydrostatic_data_after_cleared_bugs" *
                 "/quarter_degree_near_global_input_data/bathymetry-1440x600.jld2"

data_name = "quarter_degree_data"

dd = DataDep(data_name,
             "Data for simulations between 75S and 75N at 1/4 degree resolution",
             [bathymetry_url])

if !(data_name ∈ keys(DataDeps.registry))
    DataDeps.register(dd)
end

end # module
