## Mathematical Definition of Cosine

Under the context of flow matching

- target residual: `u_t = ε - A*`
- cosine:
  - `cosine^(l) = <v_t^(l), u_t> / (||v_t^(l)|| * ||u_t|| + ε)`


## Special notes for gr00t implementation

Cosine is defined under `u_t = ε - A_t` but gr00t uses `v = actions − noise`. Therefore, use `-v_t^(l)` to calculate the cosine value.

The layer index l is retained in notation only for optional per-layer analyses; for gr00t implementation, you should only consider the last layer where `v_t^(l)` is identically the model's velocity prediction used in the flow-matching loss. 