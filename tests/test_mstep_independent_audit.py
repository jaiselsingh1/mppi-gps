"""Independent LQR label -> supervised fit -> serialized deployment trace."""

import hashlib
import io

import numpy as np
import torch

from scripts.training.train_walker_mdgps import TrainConfig, _fit, policy_state
from src.rl.sac import SacActor, load_policy
from test_core_independent_audit import trace_one_label


def _numpy_forward(parameters, observation, scale, bias):
    z1 = parameters["fc1.weight"] @ observation + parameters["fc1.bias"]
    h1 = np.maximum(z1, 0)
    z2 = parameters["fc2.weight"] @ h1 + parameters["fc2.bias"]
    h2 = np.maximum(z2, 0)
    mean = parameters["fc_mean.weight"] @ h2 + parameters["fc_mean.bias"]
    tanh_mean = np.tanh(mean)
    return {"preactivation1": z1, "activation1": h1,
            "preactivation2": z2, "activation2": h2,
            "mean": mean, "tanh_mean": tanh_mean,
            "action": tanh_mean * scale + bias}


def trace_mstep(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    label = np.asarray(trace_one_label()["independent"]["label"])
    observation = np.array([0.05, -0.03, 0.02, 0.04])
    actor = SacActor(4, np.array([-1., -.5]), np.array([1., .9]), hidden_dim=3)
    constants = {
        "fc1.weight": [[.2, -.4, .1, .3], [-.1, .2, -.3, .4], [.3, .1, -.2, .2]],
        "fc1.bias": [.10, -.10, .05],
        "fc2.weight": [[.3, -.2, .1], [.2, .1, -.4], [-.1, .4, .2]],
        "fc2.bias": [.02, .05, -.02],
        "fc_mean.weight": [[.4, -.3, .2], [-.2, .5, -.1]],
        "fc_mean.bias": [-.01, .02],
        "fc_logstd.weight": [[.1, .2, .3], [-.3, -.2, -.1]],
        "fc_logstd.bias": [.04, -.02],
    }
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            parameter.copy_(torch.tensor(constants[name], dtype=torch.float32))
    before = {name: parameter.detach().numpy().astype(float).copy()
              for name, parameter in actor.named_parameters()}
    # The real fit casts observations/labels to float32. Explicitly reconstruct
    # that data boundary, then do all forward/backward arithmetic in NumPy.
    x = observation.astype(np.float32).astype(float)
    y = label.astype(np.float32).astype(float)
    scale = actor.action_scale.numpy().astype(float)
    bias = actor.action_bias.numpy().astype(float)
    forward = _numpy_forward(before, x, scale, bias)
    np.testing.assert_allclose(actor.act(observation), forward["action"], atol=2e-8, rtol=0)
    expected_loss = np.mean((forward["action"] - y) ** 2)
    dmean = 2 / len(y) * (forward["action"] - y) * scale * (1 - forward["tanh_mean"] ** 2)
    d2 = (before["fc_mean.weight"].T @ dmean) * (forward["preactivation2"] > 0)
    d1 = (before["fc2.weight"].T @ d2) * (forward["preactivation1"] > 0)
    gradients = {
        "fc_mean.weight": np.outer(dmean, forward["activation2"]), "fc_mean.bias": dmean,
        "fc2.weight": np.outer(d2, forward["activation1"]), "fc2.bias": d2,
        "fc1.weight": np.outer(d1, x), "fc1.bias": d1,
    }
    cfg = TrainConfig(grad_steps=1, batch_size=1, lr=.003)
    expected_after, adam = dict(before), {}
    for name, gradient in gradients.items():
        moment1 = (1 - .9) * gradient
        moment2 = (1 - .999) * gradient ** 2
        corrected1, corrected2 = moment1 / (1 - .9), moment2 / (1 - .999)
        update = cfg.lr * corrected1 / (np.sqrt(corrected2) + 1e-8)
        expected_after[name] = before[name] - update
        adam[name] = {"moment1": moment1, "moment2": moment2,
                      "bias_corrected1": corrected1, "bias_corrected2": corrected2,
                      "update": update}
    loss = _fit(actor, observation[None], label[None], cfg, np.random.default_rng(173))
    np.testing.assert_allclose(loss, expected_loss, atol=1e-8, rtol=0)
    gradient_errors, parameter_errors = {}, {}
    for name, parameter in actor.named_parameters():
        actual = parameter.detach().numpy()
        if name.startswith("fc_logstd"):
            assert parameter.grad is None
            np.testing.assert_array_equal(actual, before[name])
        else:
            gradient = parameter.grad.numpy()
            gradient_errors[name] = float(np.max(np.abs(gradient - gradients[name])))
            np.testing.assert_allclose(gradient, gradients[name], atol=2e-8, rtol=1e-6)
            np.testing.assert_allclose(actual, expected_after[name], atol=3e-8, rtol=0)
        parameter_errors[name] = float(np.max(np.abs(actual - expected_after[name])))
    checkpoint = io.BytesIO()
    torch.save(policy_state(actor), checkpoint)
    checkpoint_bytes = checkpoint.getvalue()
    checkpoint.seek(0)
    rng_before_load = torch.random.get_rng_state().clone()
    deployed = load_policy(torch.load(checkpoint, map_location="cpu", weights_only=True))
    assert torch.equal(torch.random.get_rng_state(), rng_before_load)
    deployed_action = deployed.act(observation)
    expected_deployed = _numpy_forward(expected_after, x, scale, bias)["action"]
    np.testing.assert_allclose(deployed_action, expected_deployed, atol=5e-8, rtol=0)
    np.testing.assert_array_equal(deployed_action, actor.act(observation))
    report = {
        "scope": "Controlled LQR fixture: independently recomputed E-step label, tiny nonlinear actor, one real M-step Adam update, real checkpoint roundtrip, deployed action. Not a replay of an archived Walker label: those archives omit the originating full simulator state/noise metadata.",
        "observation": observation, "independent_estep_label_float64": label,
        "fit_label_after_float32_boundary": y, "initial_parameters": before,
        "independent_forward": forward, "independent_mse": expected_loss, "runtime_mse": loss,
        "independent_gradients": gradients, "independent_adam": adam,
        "independent_parameters_after": expected_after,
        "max_abs_gradient_errors": gradient_errors, "max_abs_parameter_errors": parameter_errors,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "checkpoint_bytes": len(checkpoint_bytes), "load_preserves_torch_rng": True,
        "deployed_action": deployed_action, "independent_deployed_action": expected_deployed,
        "max_abs_deployed_action_error": float(np.max(np.abs(deployed_action - expected_deployed))),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    return report


def test_independent_label_fit_adam_and_saved_deployment(monkeypatch):
    trace_mstep(monkeypatch)
