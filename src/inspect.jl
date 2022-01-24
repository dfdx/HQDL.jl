const REPORTS = Dict{Symbol, Any}[]
const FLOAT_TYPES = [Float64, Float32]


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


const SPECIAL_NAMES = Dict(
    :X => Shape((3, 4)),
    :Y => Shape((4, 3)),
)


function argument_spec(arg_ex)
    if arg_ex in (:X, :Y)
        # special value - replace with shape of a fixed size
        return SPECIAL_NAMES[arg_ex]
    elseif Meta.isexpr(arg_ex, :call) && arg_ex.args[1] in (:r, :rand)
        # e.g. r(5, 10)
        value = arg_ex.args[2:end]
        @assert all(n -> n isa Number, value)
        return Shape(Tuple(value))
    elseif Meta.isexpr(arg_ex, :vect) || Meta.isexpr(arg_ex, :tuple)
        # e.g. [1, 2, 3]
        return Base.eval(arg_ex)
    elseif arg_ex isa Expr
        error("Cannot parse argument expression $arg_ex")
    else
        return Base.eval(arg_ex)
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
        # return CallSpec(Broadcast.broadcasted, [bcast_fn, bcast_args...])
        return CallSpec(Broadcast.broadcast, [bcast_fn, bcast_args...])
    else
        error("@inspect expects call or broadcasting expression as " *
                "an argument, but got $ex")
    end
end

function CallSpec(ex::Expr)
    m = @__MODULE__
    @warn ("Creating CallSpec with module implicitely set to $m. This is error-prone" *
            "and thus not recommended. Pass module explicitely to avoid this message")
    return CallSpec(m, ex)
end


function Base.show(io::IO, call::CallSpec)
    # if call.fn == Broadcast.broadcasted
    if call.fn == Broadcast.broadcast
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
    args = []
    for a in call.args
        if a isa Shape
            push!(args, random_array(T, a.value))
        elseif a isa AbstractArray
            # change device, but not eltype
            push!(args, T <: CuArray ? cu(a) : a)
        else
            push!(args, a)
        end
    end
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
function benchmark_rrule(call::CallSpec, T::Type)
    fn, args = make_fn_args(call, T)
    fwd_time = nothing
    try
        fwd_time = @belapsed $fn($args...) samples=100 seconds=1
    catch
        fwd_time = :NOT_OK
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
            bwd_time = :NOT_OK
        end
    end
    return fwd_time, bwd_time
end


# workaround for Yota due to https://github.com/dfdx/Ghost.jl/issues/26
# should be fixed after migration from Ghost to Umlaut
sumfun(fn, a) = sum(fn(a))
sumfun(fn, a, b) = sum(fn(a, b))
sumfun(fn, a, b, c) = sum(fn(a, b, c))


# HAS_GPU = CUDA.func()


function benchmark_ad(grad_fn, call::CallSpec)
    fn, args = make_fn_args(call, Array{Float32})
    cuargs = map(cu, args)
    try
        # using large values for atol & rtol due to instability of numeric gradient
        # on large values; @inspect also checks AD for correctness on smaller values
        cpu_ok, gpu_ok = gradcheck_cpu_gpu(grad_fn, sumfun, fn, args...; atol=0.1, rtol=0.1)
        cpu_time = cpu_ok == :OK ? (@belapsed grad_fn(sumfun, $fn, $args...)) : cpu_ok
        gpu_time = gpu_ok == :OK  ? (@belapsed grad_fn(sumfun, $fn, $cuargs...)) : gpu_ok
        return cpu_time, gpu_time
    catch
    end
    return :NOT_OK, :NOT_OK
end


# function benchmark_ad(call::CallSpec, T::Type)
#     yota_correct, yota_time = benchmark_ad(Yota.grad, call, T)

# end


function _measure(m::Module, ex::Expr)
    call = CallSpec(m, ex)
    # benchmark forward and backward passes
    @debug "  benchmarking"
    fwd_time_cpu, bwd_time_cpu = benchmark_rrule(call, Array{Float32})
    fwd_time_gpu, bwd_time_gpu = (CUDA.functional() ?
                                    benchmark_rrule(call, CuArray{Float32}) :
                                    (:NO_CUDA, :NO_CUDA))
    zygote_cpu, zygote_gpu = benchmark_ad(Zygote.withgradient, call)
    yota_cpu, yota_gpu = benchmark_ad(Yota.grad, call)
    return Dict(
        :call => call,
        :fwd_time_cpu => fwd_time_cpu,
        :bwd_time_cpu => bwd_time_cpu,
        :fwd_time_gpu => fwd_time_gpu,
        :bwd_time_gpu => bwd_time_gpu,
        :zygote_cpu => zygote_cpu,
        :zygote_gpu => zygote_gpu,
        :yota_cpu => yota_cpu,
        :yota_gpu => yota_gpu,
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


# we need this due to https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/234
function ChainRulesCore.rrule(::typeof(broadcast), f::F, args...) where F
    rr = rrule(Broadcast.broadcasted, f, args...)
    rr === nothing && return nothing
    y, pb = rr
    Broadcast.materialize(Broadcast.instantiate(y)), pb
end


function check_rrule_precision(call::CallSpec, AT::Type; atol=1e-3, rtol=1e-3)
    status = Dict{Type, Symbol}()
    for ET in FLOAT_TYPES
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


macro try_or(ex, val)
    quote
        try
            $(esc(ex))
        catch
            $(esc(val))
        end
    end
end


macro no_exception(ex)
    quote
        try
            $(esc(ex))
            :OK
        catch
            :NOT_OK
        end
    end
end


function _inspect(m::Module, ex::Expr; atol=1e-3, rtol=1e-3)
    @info "Inspecting $ex"
    call = CallSpec(m, ex)
    # check invocation
    fn, args = make_fn_args(call, Array{Float32})
    invoke_ok = @try_or (fn(args...); :OK) :NOT_OK
    # check rrules for multiple arg types
    @debug "  checking rrule"
    cpu_status = check_rrule_precision(call, Array; atol=atol, rtol=rtol)
    gpu_status = (CUDA.functional() ?
        check_rrule_precision(call, Array; atol=atol, rtol=rtol) :
        Dict(ET => :NO_CUDA for ET in FLOAT_TYPES)
    )
    @debug "  checking AD engines"
    zygote_cpu, zygote_gpu = @try_or(gradcheck_cpu_gpu(Zygote.withgradient, sumfun, fn, args...), (:NOT_OK, :NOT_OK))
    yota_cpu, yota_gpu = @try_or(gradcheck_cpu_gpu(Yota.grad, sumfun, fn, args...), (:NOT_OK, :NOT_OK))
    # check type stability
    jet = @no_exception JET.@test_opt fn(args...)
    jet_rrule = @no_exception JET.@test_opt rrule(fn, args...)
    # check docs
    docs_ok = has_docstring(
        # call.fn == Broadcast.broadcasted ? call.args[1] : call.fn
        call.fn == Broadcast.broadcast ? call.args[1] : call.fn
    ) ? :OK : :NOT_OK
    return merge(
        Dict(
            :call => call,
            :invoke_ok => invoke_ok,
            :zygote_cpu => zygote_cpu,
            :zygote_gpu => zygote_gpu,
            :yota_cpu => yota_cpu,
            :yota_gpu => yota_gpu,
            :jet => jet,
            :jet_rrule => jet_rrule,
            :docs_ok => docs_ok,
        ),
        Dict(Symbol("cpu_" * format_eltype(k)) => v for (k, v) in cpu_status),
        Dict(Symbol("gpu_" * format_eltype(k)) => v for (k, v) in gpu_status),
    )
end


"""
    @inspect(ex, kwargs...)

Inspect a call or broadcasting expression, check forward and backward passes
on several precisions, docs, etc.

Expressions may include literal values (e.g. `1`), calls to `rand(...)`
(also aliased as `r(...)`) or one of the special short names:

* `X` - same as `r(3, 4)`
* `Y` - same as `r(4, 3)`

Macro kwargs are passed to appropriate steps, e.g. `atol` and `rtol` are
used in `test_rrule()`.

Examples:
=========

    @inspect softmax(r(5, 10))
    @inspect relu.(X)    # same as @inspect relu.(r(3, 4))
    @inspect X * Y
"""
macro inspect(ex, kwargs...)
    kw = [esc(a) for a in kwargs]
    :(_inspect(@__MODULE__, $(QuoteNode(ex)); $(kw...)))
end


"""
    @inspect!(ex, kwargs...)

Same as `@inspect`, but also saves the result to the global table
that `report()` then uses for report generation.
"""
macro inspect!(ex, kwargs...)
    kw = [esc(a) for a in kwargs]
    quote
        df = _inspect(@__MODULE__, $(QuoteNode(ex)); $(kw...))
        push!(REPORTS, df)
        df
    end
end


###############################################################################
#                                   Report                                    #
###############################################################################


reset!() = empty!(REPORTS)


# const REPORT_HEADER = """
# Call specification:

# * `r(...)` or `rand(...)` - random array of the specified size and tested precision
# * `X`, `Y` - aliases to `r(3, 4)` and `r(4, 3)` respectively

# Status meaning:

# * :heavy_check_mark: - check passed
# * :x: - there was an error during the check
# * :grey_question: - status is unclear (e.g. there's no rrule for the op, but an AD system may still be able to handle it)


# """

function collect_report(path)
    reset!()
    include(path)
    df = DataFrame(REPORTS)
    columns = [
        :invoke_ok, :cpu_f64, :cpu_f32, :gpu_f64, :gpu_f32,
        :zygote_cpu, :zygote_gpu, :yota_cpu, :yota_gpu,
        :jet, :jet_rrule, :docs_ok
    ]
    statuses = combine(df, columns .=> ByRow(format_status) .=> columns)
    out = hcat(
        DataFrame(:call => df.call),
        statuses
    )
    return out
end


function report(outpath="REPORT.md")
    basic = collect_report("src/ops/basic.jl")
    activations = collect_report("src/ops/activations.jl")
    # template engines? not invented here!
    tmpl = open("templates/REPORT_TEMPLATE.md") do f read(f, String) end
    tmpl = replace(tmpl,
        "{{ basic }}" => make_mdtable(basic),
        "{{ activations }}" => make_mdtable(activations),
    )
    open(outpath, "w") do io
        write(io, tmpl)
    end
    @info "All done! See $outpath for the result"
end