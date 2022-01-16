using Random
using Logging
using DataFrames
using ChainRules: rrule, unthunk
using ChainRulesTestUtils
using NNlib
using CUDA
using NNlibCUDA
using BenchmarkTools
using MDTable
using Printf


const REPORTS = Dict{Symbol, Any}[]

# const UNKNOWN = ":grey_question:"
# const OK =
# const NOT_OK = ":x:"

const STATUS_TO_MESSAGE = Dict(
    :OK => ":heavy_check_mark:",
    true => ":heavy_check_mark:",
    :NOT_OK => ":x:",
    :NO_RRULE => ":grey_question:",
    :NO_CUDA => "NO CUDA"
)


###############################################################################
#                                 CallSpec                                    #
###############################################################################

struct Shape
    value::Tuple
end

split_qualified_name(name::Symbol) = [name]
split_qualified_name(name::Expr) = vcat(split_qualified_name(name.args[1]), name.args[2].value)

resolve_name(m::Module, name::Symbol) = getfield(m, name)
function resolve_name(m::Module, name::Expr)
    obj = m
    parts = split_qualified_name(name)
    for part in parts
        obj = getfield(obj, part)
    end
    return obj
end


function argument_spec(arg_ex)
    if Meta.isexpr(arg_ex, :call) && arg_ex.args[1] in (:r, :rand)
        # e.g. r(5, 10)
        value = arg_ex.args[2:end]
        @assert all(n -> n isa Number, value)
        return Shape(Tuple(value))
    elseif arg_ex isa Expr
        error("Cannot parse argument expression $arg_ex")
    else
        return arg_ex
    end
end


struct CallSpec
    fn::Any
    args::Vector{Any}
end


function CallSpec(m::Module, ex::Expr)
    if Meta.isexpr(ex, :call)
        # e.g. softmax(r(5, 10))
        fn = resolve_name(m, ex.args[1])
        args = map(argument_spec, ex.args[2:end])
        return CallSpec(fn, args)
    elseif Meta.isexpr(ex, :.)
        # e.g. :(NNlib.relu.(r(5, 10)))
        bcast_fn = resolve_name(m, ex.args[1])
        bcast_args = map(argument_spec, ex.args[2].args)
        return CallSpec(Broadcast.broadcasted, [bcast_fn, bcast_args...])
    else
        error("@inspect expects call or broadcasting expression as " *
                "an argument, but got $ex")
    end
end

function Base.show(io::IO, call::CallSpec)
    if call.fn == Broadcast.broadcasted
        arg_str = join([a isa Shape ? "r$(a.value)" : a for a in call.args[2:end]], ", ")
        print(io, "$(call.args[1]).($arg_str)")
    else
        arg_str = join([a isa Shape ? "r$(a.value)" : a for a in call.args], ", ")
        print(io, "$(call.fn)($arg_str)")
    end
end


###############################################################################
#                                 Benchmarking                                #
###############################################################################

function random_array(T::Type, sz::Tuple)
    A = T(undef, sz)
    return rand!(A)
end

function make_fn_args(call::CallSpec, T::Type)
    args = [a isa Shape ? random_array(T, a.value) : a for a in call.args]
    return call.fn, args
end


"""
    benchmark_call(call::CallSpec, T::Type)

Benchmark call specification using array type T. Return timing
of the forward and backward pass or one of the special values:

* :NO_RRULE - if the function doesn't have rrule() defined
* :ERROR - if the call (forward or backward) resulted in an error

Examples:
=========

    call = CallSpec(@__MODULE__, :(NNlib.softmax(r(5, 10))))
    benchmark_call(call, CuArray{Float32})


"""
function benchmark_call(call::CallSpec, T::Type)
    fn, args = make_fn_args(call, T)
    fwd_time = nothing
    try
        fwd_time = @belapsed $fn($args...) samples=100 seconds=1
    catch
        fwd_time = :ERROR
    end
    bwd_time = nothing
    rr = rrule(fn, args...)
    if rr === nothing
        bwd_time = :NO_RRULE
    else
        try
            y, pb = rr
            dy = y isa AbstractArray ? random_array(T, size(y)) : one(dy)
            bwd_time = @belapsed map($unthunk, $pb($dy)) samples=100 seconds=1
        catch
            bwd_time = :ERROR
        end
    end
    return fwd_time, bwd_time
end


function _measure(m::Module, ex::Expr)
    call = CallSpec(m, ex)
    # benchmark forward and backward passes
    @debug "  benchmarking"
    fwd_time_cpu, bwd_time_cpu = benchmark_call(call, Array{Float32})
    fwd_time_gpu, bwd_time_gpu = (CUDA.functional() ?
                                    benchmark_call(call, CuArray{Float32}) :
                                    (:NO_CUDA, :NO_CUDA))
    return Dict(
        :call => call,
        :fwd_time_cpu => fwd_time_cpu,
        :bwd_time_cpu => bwd_time_cpu,
        :fwd_time_gpu => fwd_time_gpu,
        :bwd_time_gpu => bwd_time_gpu
    )
end


macro measure(ex)
    :(_measure(@__MODULE__, $(QuoteNode(ex))))
end


###############################################################################
#                                 Documentation                               #
###############################################################################

function has_docstring(fn)
    md = Docs.doc(fn)
    return md.content[1].content[1] != "No documentation found."
end


###############################################################################
#                               Inspect/Analyze                               #
###############################################################################


function check_rrule_precision(call::CallSpec, AT::Type; atol=1e-3, rtol=1e-3)
    status = Dict{Type, Symbol}()
    for ET in (Float16, Float32, Float64)
        fn, args = make_fn_args(call, AT{ET})
        if rrule(fn, args...) === nothing
            status[ET] = :NO_RRULE
            continue
        end
        try
            test_rrule(fn, args...; check_inferred=false, atol=atol, rtol=rtol)
            status[ET] = :OK
        catch
            status[ET] = :NOT_OK
        end
    end
    return status
end


function _inspect(m::Module, ex::Expr; atol=1e-3, rtol=1e-3)
    @info "Inspecting $ex"
    call = CallSpec(m, ex)
    # check type coverage
    @debug "  checking on CPU"
    cpu_status = check_rrule_precision(call, Array; atol=atol, rtol=rtol)
    @debug "  checking on GPU"
    gpu_status = (CUDA.functional() ?
        check_rrule_precision(call, Array; atol=atol, rtol=rtol) :
        Dict(ET => :NO_CUDA for ET in (Float16, Float32, Float64))
    )
    # check docs
    docs_ok = has_docstring(
        call.fn == Broadcast.broadcasted ? call.args[1] : call.fn
    )
    return merge(
        Dict(
            :call => call,
            :docs_ok => docs_ok
        ),
        Dict(Symbol(lowercase("cpu_$(k)_ok")) => v for (k, v) in cpu_status),
        Dict(Symbol(lowercase("gpu_$(k)_ok")) => v for (k, v) in gpu_status),
    )
end


macro inspect(ex, kwargs...)
    kw = [esc(a) for a in kwargs]
    :(_inspect(@__MODULE__, $(QuoteNode(ex)); $(kw...)))
end


macro analyze(ex)
    quote
        df = _inspect(@__MODULE__, $(QuoteNode(ex)))
        push!(REPORTS, df)
    end
end

reset!() = empty!(REPORTS)


# function find_row(g::SubDataFrame, arrtyp, eltyp)
#     rows = filter(row -> row.arrtyp == arrtyp && row.eltyp == eltyp, g)
#     return first(rows)
# end


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


function report(path="src/ops/basic.jl", outpath="basic_output.md")
    reset!()
    include(path)
    df = DataFrame(REPORTS)
    out = DataFrame(
        :call => df.call,
        :cpu_f64 => map(format_status, df.cpu_float64_ok),
        :cpu_f32 => map(format_status, df.cpu_float32_ok),
        :cpu_f16 => map(format_status, df.cpu_float16_ok),
        :gpu_f64 => map(format_status, df.gpu_float64_ok),
        :gpu_f32 => map(format_status, df.gpu_float32_ok),
        :gpu_f16 => map(format_status, df.gpu_float16_ok),
        :docs_ok => map(format_status, df.docs_ok)
    )
    if outpath !== nothing
        MDTable.writeMDTable(outpath, out)
    end
    return out
end


function main()
    ENV["JULIA_DEBUG"] = Main

    fn = batched_mul
    default_args = (rand(3, 4, 10), rand(4, 3))
    @analyze batched_mul(rand(3, 4, 10), rand(4, 3))

    fn = (*)
    default_args = (rand(3, 4), rand(4, 3))
    @inspect *(rand(3, 4), rand(4, 3))
    @analyze *(rand(3, 4), rand(4, 3))
end