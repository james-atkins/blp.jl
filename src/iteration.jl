import LinearAlgebra as LA

function within_tolerance(x, x_next, tolerance)
    for (a, b) in zip(x, x_next)
        if abs(a - b) >= tolerance
            return false
        end
    end
    
    return true
end

function numerical_issues(x)
    for elem in x
        if !isfinite(elem)
            return true
        end
    end

    return false
end

struct IterationResult
    delta::Vector{Float64}
    evaluations::Int32
    iterations::Int32
    error:::Union{String, Nothing}
end


struct SimpleFixedPointIteration
    tolerance::Float64
    max_iterations::Integer
end

function fixed_point_iteration(iteration::SimpleFixedPointIteration, x0::Vector{Float64}, contraction!::Function)
    x = copy(x0)
    x_next = Vector{Float64}(undef, length(x))
    iterations = 0

    while true
        if iterations > iteration.max_iterations
            return IterationResult(x, iterations, iterations, "Exceeded maximum iterations")
        end
        
        contraction!(x_next, x)
        iterations += 1

        if numerical_issues(x_next)
            return IterationResult(x, iterations, iterations, "Numerical issues")
        end

        if within_tolerance(x, x_next, iteration.tolerance)
            return IterationResult(x, iterations, iterations, nothing)
        end
    end
end
