"""Classical baselines.

Two kinds of input:
- full state vectors: 2 * 2^n real numbers (real and imaginary parts, global
  phase fixed). Exponentially large and not something an experiment gives you,
  so this is an upper bound on what classical learning could do.
- measurement snapshots: features estimated from the same number of copies of
  the state that the quantum model uses (see measure.py).
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def statevector_features(states):
    """[Re psi, Im psi] with the global phase fixed (largest amplitude real and positive)."""
    S = np.asarray(states, dtype=complex)
    k = np.argmax(np.abs(S), axis=1)
    phase = S[np.arange(len(S)), k]
    S = S * (np.abs(phase) / phase)[:, None]
    return np.concatenate([S.real, S.imag], axis=1)


def classical_models(seed=0):
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale")),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3,
                                                             max_iter=3000, random_state=seed)),
    }


def fit_predict(model, X_train, y_train, X_test):
    if len(np.unique(y_train)) < 2:  # degenerate tiny training sets in the data-efficiency sweep
        return np.full(len(X_test), y_train[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate_all(X_train, y_train, X_test, y_test, seed=0):
    """Accuracy of every classical model on one feature set."""
    return {name: float(np.mean(fit_predict(m, X_train, y_train, X_test) == y_test))
            for name, m in classical_models(seed).items()}
