import Yota: ngradient


only_arg_gradients(::typeof(Yota.grad), ad_grads::Tuple) = ad_grads[2:end]
only_arg_gradients(::typeof(Zygote.withgradient), ad_grads::Tuple) = ad_grads


to_cpu(x::AbstractArray) = convert(Array, x)
to_cpu(x) = x

function gradcheck_cpu_gpu(grad_fn, f, args...; atol=1e-3, rtol=1e-3)
    n_grads = ngradient(f, args...)
    ad_grads_cpu = grad_fn(f, args...)[2]
    ad_grads_cpu = only_arg_gradients(grad_fn, ad_grads_cpu)
    results_cpu = []
    for n in 1:length(args)
        if ad_grads_cpu[n] isa NoTangent || ad_grads_cpu[n] === nothing
            push!(results_cpu, true)
        else
            push!(results_cpu, isapprox(ad_grads_cpu[n], n_grads[n], rtol=rtol, atol=atol))
        end
    end
    cpu_ok = all(results_cpu) ? :OK : :NOT_OK
    gpu_ok = if CUDA.functional()
        args_gpu = map(cu, args)
        ad_grads_gpu = only_arg_gradients(grad_fn, grad_fn(f, args_gpu...)[2])
        results_gpu = []
        for n in 1:length(args)
            if ad_grads_gpu[n] isa NoTangent || ad_grads_gpu[n] === nothing
                push!(results_gpu, true)
            else

                push!(results_gpu, isapprox(to_cpu(ad_grads_gpu[n]), n_grads[n], rtol=rtol, atol=atol))
            end
        end
        gpu_ok = all(results_gpu) ? :OK : :NOT_OK
    else
        :NO_CUDA
    end
    return cpu_ok, gpu_ok
end