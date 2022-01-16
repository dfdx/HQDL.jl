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


include("format.jl")
include("markdown.jl")
include("inspect.jl")