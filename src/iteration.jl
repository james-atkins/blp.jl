export IterationResult, IterationStatus, ITERATION_CONVERGED, ITERATION_NUMERICAL_ISSUES, ITERATION_EXCEEDED_MAX_ITERATIONS, fixed_point_iteration

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

@enum IterationStatus begin
    ITERATION_CONVERGED
    ITERATION_NUMERICAL_ISSUES
    ITERATION_EXCEEDED_MAX_ITERATIONS
end

struct IterationResult
    status::IterationStatus
    delta::Vector{Float64}
    iterations::Int32
    evaluations::Int32
end

export SimpleFixedPointIteration

abstract type Iteration end

Base.@kwdef struct SimpleFixedPointIteration <: Iteration
    tolerance::Float64 = 1e-14
    max_iterations::Integer = 5_000
end

function fixed_point_iteration(iteration::SimpleFixedPointIteration, x0::AbstractVector, contraction!::Function)
    x = copy(x0)
    x_next = Vector{eltype(x)}(undef, length(x))
    iterations = 0

    while iterations < iteration.max_iterations
        contraction!(x_next, x)
        iterations += 1

        if numerical_issues(x_next)
            return IterationResult(ITERATION_NUMERICAL_ISSUES, x, iterations, iterations)
        end

        if within_tolerance(x, x_next, iteration.tolerance)
            return IterationResult(ITERATION_CONVERGED, x_next, iterations, iterations)
        end

        x .= x_next
    end

    return IterationResult(ITERATION_EXCEEDED_MAX_ITERATIONS, x, iterations, iterations)
end

export SQUAREMIteration, SQUAREM1, SQUAREM2, SQUAREM3

@enum SQUAREMScheme SQUAREM1=1 SQUAREM2=2 SQUAREM3=3

Base.@kwdef struct SQUAREMIteration <: Iteration
    max_iterations::Int = 1_000
    tolerance::Float64 = 1e-14
    scheme::SQUAREMScheme = SQUAREM3
    step_min::Float64 = 1.0
    step_max::Float64 = 1.0
    step_factor::Float64 = 4.0
end

function fixed_point_iteration(iteration::SQUAREMIteration, x_initial::AbstractVector, contraction!::Function)
    x0 = copy(x_initial)
    x1 = Vector{eltype(x0)}(undef, length(x0))
    x2 = Vector{eltype(x0)}(undef, length(x0))
    x_accel = Vector{eltype(x0)}(undef, length(x0))
    r = Vector{eltype(x0)}(undef, length(x0))
    v = Vector{eltype(x0)}(undef, length(x0))

    step_min = iteration.step_min
    step_max = iteration.step_max

    evaluations = 0
    iterations = 0

    while iterations < iteration.max_iterations
        # First step
        contraction!(x1, x0)
        evaluations += 1

        if numerical_issues(x1)
            return IterationResult(ITERATION_NUMERICAL_ISSUES, x0, iterations, evaluations)
        end

        if within_tolerance(x1, x0, iteration.tolerance)
            return IterationResult(ITERATION_CONVERGED, x1, iterations, evaluations)
        end

        # Second step
        contraction!(x2, x1)
        evaluations += 1

        if numerical_issues(x2)
            return IterationResult(ITERATION_NUMERICAL_ISSUES, x1, iterations, evaluations)
        end

        if within_tolerance(x2, x1, iteration.tolerance)
            return IterationResult(ITERATION_CONVERGED, x2, iterations, evaluations)
        end

        @. r = x1 - x0
        @. v = (x2 - x1) - r

        # Compute the step length
        if iteration.scheme == SQUAREM1
            alpha = (r ⋅ v) / (v ⋅ v)
        elseif iteration.scheme == SQUAREM2
            alpha = (r ⋅ r) / (r ⋅ v)
        else
            alpha = -sqrt((r ⋅ r) / (v ⋅ v))
        end

        # Bound the step length and update its bounds
        alpha = -max(step_min, min(step_max, -alpha))
        if -alpha == step_max
            step_max *= iteration.step_factor
        end
        if -alpha == step_min && step_min < 0
            step_min *= iteration.step_factor
        end

        # Acceleration step
        @. x_accel = x0 - (2 * alpha * r) + (alpha^2 * v)

        contraction!(x0, x_accel)
        evaluations += 1

        iterations += 1

        if numerical_issues(x0)
            return IterationResult(ITERATION_NUMERICAL_ISSUES, x1, iterations, evaluations)
        end

        if within_tolerance(x0, x_accel, iteration.tolerance)
            return IterationResult(ITERATION_CONVERGED, x2, iterations, evaluations)
        end
    end

    return IterationResult(ITERATION_EXCEEDED_MAX_ITERATIONS, x0, iterations, evaluations)
end
