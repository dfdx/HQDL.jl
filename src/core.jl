using Random
using Logging
using Printf
using DataFrames
using ChainRulesCore
using ChainRules
using ChainRules: rrule, unthunk
using ChainRulesTestUtils
using NNlib
using CUDA
using NNlibCUDA
using BenchmarkTools
using JET
import Yota
import Zygote


CUDA.allowscalar(false)


include("format.jl")
include("markdown.jl")
include("gradcheck.jl")
include("inspect.jl")