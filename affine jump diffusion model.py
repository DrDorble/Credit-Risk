def affine_jump_diffusion(Paths, Steps, T, kappa, theta, sigma, lambda_jump, mu_jump, sigma_jump):
    dt = T / Steps
    time = np.linspace(0, T, Steps + 1)
    X = np.zeros((Paths, Steps + 1))
    X[:, 0] = theta  # Initialize X at the mean level theta

    # Generate Brownian motion increments
    dW = np.random.normal(0, np.sqrt(dt), (Paths, Steps))

    # Generate Poisson jumps
    poisson_paths = np.random.poisson(lambda_jump * dt, (Paths, Steps))  # Jump occurrences
    Z = poisson_paths  # Poisson process as jump occurrences

    for i in range(Steps):
        # Generate jump sizes, ensuring they are within reasonable bounds
        Jumps = Z[:, i] * np.random.normal(mu_jump, sigma_jump, Paths)

        X[:, i + 1] = X[:, i] + kappa * (theta - X[:, i]) * dt + sigma * np.sqrt(np.maximum(0, X[:, i])) * dW[:, i] + Jumps
        X[:, i + 1] = np.maximum(0, X[:, i + 1])  # Reflective barrier at zero

    return time, X
