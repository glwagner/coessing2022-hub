using JLD2

file = jldopen("near_global_lat_lon_1440_600_48_fine_surface.jld2")
key  = keys(file["timeseries/u"])

key = key[2:end] # removing serialized

# now just selecting the timeseries 1-day spaced
new_key = []
u = []
v = []
T = []
for i in 1:40:length(key)
    push!(new_key, key[i])
    push!(u, file["timeseries/u/$(key[i])"])
    push!(v, file["timeseries/v/$(key[i])"])
    push!(T, file["timeseries/T/$(key[i])"])
end

u = u |> Vector{Array{Float32, 3}};
v = v |> Vector{Array{Float32, 3}};
T = T |> Vector{Array{Float32, 3}};

u̅ = zeros(size(u[1]))
v̅ = zeros(size(v[1]))
T̅ = zeros(size(T[1]))

# calculating averages
for i in 1:length(key)
    u̅ .+= file["timeseries/u/$(key[i])"] 
    v̅ .+= file["timeseries/v/$(key[i])"] 
    T̅ .+= file["timeseries/T/$(key[i])"] 
    @info "adding field $i of $(length(key))"
end

u̅ ./= length(key)
v̅ ./= length(key)
T̅ ./= length(key)


@inline function horizonthal_filter(vec, iᵢ, iₑ, jᵢ, jₑ, Nz_mine)
    vec2 = deepcopy(vec)
    for k in 1:Nz_mine
        for i in iᵢ:iₑ, j in jᵢ:jₑ
            neigbours = [vec[i, j, k], vec[i + 1, j, k], vec[i - 1, j, k], vec[i, j + 1, k], vec[i, j - 1, k]]
            non_null  = Int.(neigbours .!= 0)
            if sum(non_null) > 0
                vec2[i, j, k] = sum(neigbours) / 5
            end
        end
    end
    return vec2
end

# ū = deepcopy(u̅)
# v̄ = deepcopy(v̅)
# for passes in 1:1
#     ū = horizonthal_filter(ū, 2, 1438, 2, 598, 1)
#     v̄ = horizonthal_filter(v̄, 2, 1438, 2, 598, 1)
# end

# jldsave("prescribed_velocity_field_smoothed.jld2", u̅=ū, v̅=v̄)


jldsave("prescribed_surface_fields.jld2", u=u, v=v, um=u̅, vm=v̅) #, T̅=T̅)




