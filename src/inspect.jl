using DataFrames
using ChainRules: rrule, unthunk
using ChainRulesTestUtils
using NNlib
using CUDA
using NNlibCUDA
using BenchmarkTools
using MDTable
using Printf


const REPORTS = DataFrame[]


function is_valid_argspec(a)
    return a isa Number || (Meta.isexpr(a, :call) && a.args[1] == :rand)
end


# macro try_or_msg(ex, msg)
#     return quote
#         try
#             $ex
#         catch
#             $msg
#         end
#     end
# end


function has_docstring(fn)
    md = Docs.doc(fn)
    return md.content[1].content[1] != "No documentation found."
end


function _inspect(ex)
    @info "Inspecting $ex"
    @assert(
        Meta.isexpr(ex, :call),
        "@inspect expects call expression as an argument, but got $ex"
    )
    fn = eval(ex.args[1])
    default_args = []
    for a in ex.args[2:end]
        @assert(
            is_valid_argspec(a),
            "Arguments must be literals or calls to `rand()`, but encountered `$a`"
        )
        push!(default_args, eval(a))
    end
    df = DataFrame(
        :ex => [],
        :arrtyp => Type[], :eltyp => Type[],
        :fwd_time => [], :bwd_time => [],
        :has_docstring => Bool[]
    )
    for arrtyp in [Array, CuArray], eltyp in [Float64, Float32, Float16]
        @debug "  arrtyp = $arrtyp; eltyp = $eltyp"
        args = [arg isa AbstractArray ? convert(arrtyp{eltyp}, arg) : eltyp(arg)
                for arg in default_args]
        # measure inference time
        fwd_time = try
            @belapsed $fn($args...) seconds=1 samples=100
        catch
            "failed"
        end
        # measure backpropagation time
        bwd_time = try
            rr = rrule(fn, args...)
            if rr !== nothing
                y, pb = rr
                dy = y isa AbstractArray ? convert(arrtyp{eltyp}, ones(size(y))) : one(dy)
                @belapsed map($unthunk, $pb($dy)) seconds=1 samples=100
            else
                "no rrule"
            end
        catch
            "failed"
        end
        # combine the results
        push!(df, (
            ex,
            # fn, Any[size(arg) for arg in args],
            arrtyp, eltyp,
            fwd_time, bwd_time,
            has_docstring(fn)
        ))
    end
    return df
end


macro inspect(ex)
    :(_inspect($(QuoteNode(ex))))
end


macro analyze(ex)
    quote
        df = _inspect($(QuoteNode(ex)))
        push!(REPORTS, df)
    end
end

reset!() = empty!(REPORTS)


function find_row(g::SubDataFrame, arrtyp, eltyp)
    rows = filter(row -> row.arrtyp == Array && row.eltyp == Float32, g)
    return first(rows)
end


const MAYBE = ":grey_question:"
const YES = ":heavy_check_mark:"
const NO = ":x:"


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


function report!(summary="summary.md", detailed="detailed.md")
    rep = reduce(vcat, REPORTS)
    # rep = sort(rep, :ex, by=string)
    MDTable.writeMDTable(detailed, rep)
    sm = DataFrame(
        :func_name => [],
        :fwd_cpu_time => [], :fwd_gpu_time => [],
        :bwd_cpu_time => [], :bwd_gpu_time => [],
        :has_docstring => []
    )
    for g in groupby(rep, :ex)
        func_name = first(g.ex).args[1]
        fwd_cpu_time = find_row(g, Array, Float32).fwd_time
        fwd_gpu_time = find_row(g, CuArray, Float32).fwd_time
        bwd_cpu_time = find_row(g, Array, Float32).bwd_time
        bwd_gpu_time = find_row(g, CuArray, Float32).bwd_time
        has_doc = first(g.has_docstring)
        push!(sm, (
            func_name,
            fwd_cpu_time isa Real ? format_time(fwd_cpu_time) : NO,
            fwd_gpu_time isa Real ? format_time(fwd_gpu_time) : NO,
            bwd_cpu_time isa Real ? format_time(bwd_cpu_time) : NO,
            bwd_gpu_time isa Real ? format_time(bwd_gpu_time) : NO,
            has_doc ? YES : NO
        ))
    end
    MDTable.writeMDTable(summary, sm)
end


function run_report()
    @analyze *(rand(128, 512), rand(512, 64))
    @analyze batched_mul(rand(128, 512, 64), rand(512, 128))
    @analyze softmax(rand(512, 64))
    report!()
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