## Symbols and VLA forward function

Let the vision-language-action model be denoted by

$$
\hat{A}^0 = \Phi_\theta(A_t^\tau, h_v, h_\ell, h_s, h_t),
$$

where $\theta$ are the model parameters and $\hat{A}^0$ is the predicted clean action (or equivalent action-space output of the policy head).

The model takes the following inputs:

$$
h_v = f_{\mathrm{vision}}(x_v), \qquad
h_\ell = f_{\mathrm{lang}}(x_\ell), \qquad
h_s = f_{\mathrm{state}}(x_s), \qquad
h_t = f_{\mathrm{time}}(\tau),
$$

where $x_v$ is the visual observation, $x_\ell$ is the language input, $x_s$ is the robot state, and $\tau \in (0,1]$ is the diffusion time.

The noisy action latent is defined as

$$
A_t^\tau = \tau \epsilon + (1-\tau) A^*,
$$

where $\epsilon \sim \mathcal{N}(0,I)$ and $A^*$ is the ground-truth action.

Let the training or analysis loss be

$$
L = \mathcal{L}\big(\Phi_\theta(A_t^\tau, h_v, h_\ell, h_s, h_t), A^*\big).
$$

Throughout, all inputs except the variable under analysis are held fixed.

## Vision Grad Norm: local gradient norm on the target variable

As an example, we consider the vision embedding and compute the gradient of the loss with respect to $h_v$ to measure local sensitivity:

$$
g_v := \nabla_{h_v} L.
$$

The scalar sensitivity score is defined as the gradient norm

$$
S_v := \|g_v\|_2.
$$

In practice, for each evaluation example, one performs a forward pass to compute $L$, backpropagates through the model while treating $h_v$ as the differentiation target, and records $\| \nabla_{h_v} L \|_2$.

This quantity measures the first-order sensitivity of the loss to infinitesimal perturbations of the vision embedding at the current point.


### Equivalence of $\nabla_{\delta_v} L(0)$ and $\nabla_{h_v} L$

We note that $\nabla_{\delta_v} L(0) = \nabla_{h_v} L$ since $\delta_v$ is an additive reparameterization of $h_v$.


## Special notes for gr00t implementation

During inference gr00t will have 4 steps. The model should calculate and save grad norm of all 4 inference steps with pure noise instead of 1 step with randomly sampled time and noise deducted by ground truth action.

Gr00t uses noisy = (1−t)·noise + t·actions. So pure noise starts from t=0.

Evaluate at the four inference-time latents (t = 0, 0.25, 0.5, 0.75 in GR00T's convention). "The first diffusion step" means t=0.

For coding, you should create delta = torch.zeros_like(h_v).requires_grad_(True), add it to the embedding, and call torch.autograd.grad(loss, delta). Since ∂L/∂δ = ∂L/∂h_v at δ=0 (the "equivalence"), this gives the same gradient while being much cleaner to implement.

Vision embeddings specifically refer to the vision embeddings after the image encoder, not the VLM output embeddings.

Vision embedding = the image-embedding tokens, i.e. the output of vision tower + projector in the LLM embedding space), before the language model. All camera views should be concatenated, while masked-out images should not occupy token slots).