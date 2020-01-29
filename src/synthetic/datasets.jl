using Base
using Random
using Distributions

abstract type AbstractSyntheticDataSet end

# MAML Sinusoid
struct Sinusoid <: AbstractSyntheticDataSet
    p_A::UnivariateDistribution
    p_ϕ::UnivariateDistribution
    p_x::UnivariateDistribution
end

Sinusoid() = Sinusoid(Uniform(0.1, 0.5), Uniform(0.0, π), Uniform(-5.0, 5.0))

function Base.rand(p::Sinusoid, k::Int)
    A = rand(p.p_A)
    ϕ = rand(p.p_ϕ)
    X = rand(p.p_x, k)
    Y = @. A * sin(X + ϕ)
    return X, Y
end
function Base.rand(p::Sinusoid)
    A= rand(p.p_A)
    ϕ = rand(p.p_ϕ)
    x = rand(p.p_x)
    y = A * sin(x + ϕ)
    return x, y
end

# PMAML Sinusoid+Line


# Uncertainty in Multitask Transfer learning, Harmonics
# NOTE: Unclear what sigma_y is from paper
struct Harmonic <: AbstractSyntheticDataSet
    p_A::UnivariateDistribution
    p_ω::UnivariateDistribution
    p_ϕ::UnivariateDistribution
    p_meta_x::UnivariateDistribution
    p_ϵ::UnivariateDistribution
end

Harmonic() = Harmonic(Uniform(0.0, 1.0), Uniform(5.0, 7.0), Uniform(0, 2π), Uniform(-4.0, 4.0), Normal(0.0, 0.1))

function Base.rand(p::Harmonic, k::Int)
    A = rand(p.p_A, 2)
    A₁, A₂ = A[1], A[2]
    ω = rand(p.p_ω)
    ϕ = rand(p.p_ϕ, 2)
    ϕ₁, ϕ₂ = ϕ[1], ϕ[2]
    μ_x = rand(p.p_meta_x)
    p_x = Normal(μ_x, 1.0)
    X = rand(p_x, k)
    ϵ = rand(p.p_ϵ, k)
    Y = @. A₁ * sin(ω * X + ϕ₁) + A₂ * sin(2ω * X + ϕ₂) + ϵ
    return X, Y
end
function Base.rand(p::Harmonic)
    A = rand(p.p_A, 2)
    A₁, A₂ = A[1], A[2]
    ω = rand(p.p_ω)
    ϕ = rand(p.p_ϕ, 2)
    ϕ₁, ϕ₂ = ϕ[1], ϕ[2]
    μ_x = rand(p.p_meta_x)
    p_x = Normal(μ_x, 1.0)
    X = rand(p_x)
    ϵ = rand(p.p_ϵ)
    y = A₁ * sin(ω * x + ϕ₁) + A₂ * sin(2ω * x + ϕ₂) + ϵ
    return x, y
end

using Plots; pyplot()
maml_sine = Sinusoid()
harmonic = Harmonic()

X, Y = rand(harmonic, 500)
sortind = sortperm(X)
X = X[sortind]
Y = Y[sortind]
p = plot(X, Y)
for i in 1:5
    X, Y = rand(harmonic, 500)
    sortind = sortperm(X)
    X = X[sortind]
    Y = Y[sortind]
    plot!(p, X, Y)
end
p
