const STATUS_TO_MESSAGE = Dict(
    :OK => ":heavy_check_mark:",
    :NOT_OK => ":x:",
    :NO_RRULE => ":grey_question:",
    :NO_CUDA => "NO CUDA",
    true => ":heavy_check_mark:",
    false => ":x:"
)


function format_time(t::Real)
    if t >= 1
        return @sprintf "%.1f s" t
    elseif t >= 1e-3
        return @sprintf "%.1f ms" (t * 1e3)
    elseif t >= 1e-6
        return @sprintf "%.1f μs" (t * 1e6)
    elseif t >= 1e-9
        return @sprintf "%.1f ns" (t * 1e9)
    else
        return @sprintf "%.1e" t
    end
end
format_time(t) = t


function format_status(status::Symbol)
    return get(STATUS_TO_MESSAGE, status, status)
end


format_eltype(::Type{Float16}) = "f16"
format_eltype(::Type{Float32}) = "f32"
format_eltype(::Type{Float64}) = "f64"