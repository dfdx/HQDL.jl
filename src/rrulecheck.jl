import FiniteDifferences

function ngradient(fn, args...)
    fdm = FiniteDifferences.central_fdm(5, 1)
    FiniteDifferences.grad(fdm, fn, args...)
end


"""
    check_rrule(f, args...; atol=1e-5, rtol=1e-5)

Quicker and less functional alternative to `test_rrule`.
"""
function check_rrule(fn, args...; atol=1e-5, rtol=1e-5)
    fns = (args...) -> sum(fn(args...))
    y, pb = rrule(fn, args...)
    dy = convert(typeof(y), ones(size(y))) # / reduce(*, size(y))
    rr_grad = map(unthunk, pb(dy)[2:end])
    n_grad = ngradient(fns, args...)
    results = []
    for n in 1:length(args)
        push!(results, isapprox(rr_grad[n], n_grad[n], rtol = rtol, atol = atol))
    end
    return all(results)
end