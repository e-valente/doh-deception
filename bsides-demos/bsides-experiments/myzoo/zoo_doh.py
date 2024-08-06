from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
)
import numpy as np
import logging
from tqdm.auto import trange
from typing import Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

class ZooAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "abort_early",
        "use_importance",
        "nb_parallel",
        "batch_size",
        "variable_h",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 1e-2,
        max_iter: int = 10,
        binary_search_steps: int = 1,
        initial_const: float = 1e-3,
        abort_early: bool = True,
        use_importance: bool = True,
        nb_parallel: int = 128,
        batch_size: int = 1,
        variable_h: float = 1e-4,
        verbose: bool = True,
    ):
        super().__init__(estimator=classifier)

        if len(classifier.input_shape) == 1:
            self.input_is_feature_vector = True
            if batch_size != 1:
                raise ValueError(
                    "The current implementation of Zeroth-Order Optimization attack only supports "
                    "`batch_size=1` with feature vectors as input."
                )
        else:
            self.input_is_feature_vector = False

        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.use_importance = use_importance
        self.nb_parallel = nb_parallel
        self.batch_size = batch_size
        self.variable_h = variable_h
        self.verbose = verbose
        self._check_params()

        # Initialize noise variable to zero
        self._current_noise = np.zeros((batch_size,) + self.estimator.input_shape, dtype=ART_NUMPY_DTYPE)
        self._sample_prob = np.ones(self._current_noise.size, dtype=ART_NUMPY_DTYPE) / self._current_noise.size

        # Initialize Adam variables
        self.adam_mean = np.zeros(self._current_noise.size, dtype=ART_NUMPY_DTYPE)
        self.adam_var = np.zeros(self._current_noise.size, dtype=ART_NUMPY_DTYPE)
        self.adam_epochs = np.ones(self._current_noise.size, dtype=int)

        # internal
        if self.abort_early:
            self._early_stop_iters = self.max_iter // 10 if self.max_iter >= 10 else self.max_iter

    def _loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray, c_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        l2dist = np.sum(np.square(x - x_adv).reshape(x_adv.shape[0], -1), axis=1)
        preds = self.estimator.predict(x_adv, batch_size=self.batch_size)
        z_target = np.sum(preds * target, axis=1)
        z_other = np.max(
            preds * (1 - target) + (np.min(preds, axis=1) - 1)[:, np.newaxis] * target,
            axis=1,
        )

        if self.targeted:
            loss = np.maximum(z_other - z_target + self.confidence, 0)
        else:
            loss = np.maximum(z_target - z_other + self.confidence, 0)

        return preds, l2dist, c_weight * loss + l2dist

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        x_adv_list = []
        for batch_id in trange(nb_batches, desc="ZOO", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            res = self._generate_batch(x_batch, y_batch)
            x_adv_list.append(res)
        x_adv = np.vstack(x_adv_list)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            np.clip(x_adv, clip_min, clip_max, out=x_adv)

        logger.info(
            "Success rate of ZOO attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _generate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        c_current = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 1e10 * np.ones(x_batch.shape[0])

        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        for bss in range(self.binary_search_steps):
            logger.debug(
                "Binary search step %i out of %i (c_mean==%f)",
                bss,
                self.binary_search_steps,
                np.mean(c_current),
            )

            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)

            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            c_current, c_lower_bound, c_upper_bound = self._update_const(
                y_batch, best_label, c_current, c_lower_bound, c_upper_bound
            )

        return o_best_attack

    def _update_const(
        self,
        y_batch: np.ndarray,
        best_label: np.ndarray,
        c_batch: np.ndarray,
        c_lower_bound: np.ndarray,
        c_upper_bound: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        comparison = [
            self._compare(best_label[i], np.argmax(y_batch[i])) and best_label[i] != -np.inf
            for i in range(len(c_batch))
        ]
        for i, comp in enumerate(comparison):
            if comp:
                c_upper_bound[i] = min(c_upper_bound[i], c_batch[i])
                if c_upper_bound[i] < 1e9:
                    c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2
            else:
                c_lower_bound[i] = max(c_lower_bound[i], c_batch[i])
                c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2 if c_upper_bound[i] < 1e9 else c_batch[i] * 10

        return c_batch, c_lower_bound, c_upper_bound

    def _compare(self, object1: Any, object2: Any) -> bool:
        return object1 == object2 if self.targeted else object1 != object2

    def _generate_bss(
        self, x_batch: np.ndarray, y_batch: np.ndarray, c_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_orig = x_batch.astype(ART_NUMPY_DTYPE)
        fine_tuning = np.full(x_batch.shape[0], False, dtype=bool)
        prev_loss = 1e6 * np.ones(x_batch.shape[0])
        prev_l2dist = np.zeros(x_batch.shape[0])

        best_dist = np.inf * np.ones(x_batch.shape[0])
        best_label = -np.inf * np.ones(x_batch.shape[0])
        best_attack = np.array([x_orig[i] for i in range(x_batch.shape[0])])

        for iter_ in range(self.max_iter):
            logger.debug("Iteration step %i out of %i", iter_, self.max_iter)

            x_adv = self._optimizer(x_orig, y_batch, c_batch)
            preds, l2dist, loss = self._loss(x_orig, x_adv, y_batch, c_batch)

            mask_fine_tune = (~fine_tuning) & (loss == l2dist) & (prev_loss != prev_l2dist)
            fine_tuning[mask_fine_tune] = True
            self._reset_adam(self.adam_mean.size, np.repeat(mask_fine_tune, x_adv[0].size))
            prev_l2dist = l2dist

            if self.abort_early and iter_ % self._early_stop_iters == 0:
                if (loss > 0.9999 * prev_loss).all():
                    break
                prev_loss = loss

            labels_batch = np.argmax(y_batch, axis=1)
            for i, (dist, pred) in enumerate(zip(l2dist, np.argmax(preds, axis=1))):
                if dist < best_dist[i] and self._compare(pred, labels_batch[i]):
                    best_dist[i] = dist
                    best_attack[i] = x_adv[i]
                    best_label[i] = pred

        return best_dist, best_label, best_attack

    def _optimizer(self, x: np.ndarray, targets: np.ndarray, c_batch: np.ndarray) -> np.ndarray:
        coord_batch = np.repeat(self._current_noise, 2 * self.nb_parallel, axis=0)
        coord_batch = coord_batch.reshape(2 * self.nb_parallel * self._current_noise.shape[0], -1)

        if self.use_importance and np.unique(self._sample_prob).size != 1:
            indices = (
                np.random.choice(
                    coord_batch.shape[-1] * x.shape[0],
                    self.nb_parallel * self._current_noise.shape[0],
                    replace=False,
                    p=self._sample_prob.flatten(),
                )
                % coord_batch.shape[-1]
            )
        else:
            try:
                indices = (
                    np.random.choice(
                        coord_batch.shape[-1] * x.shape[0],
                        self.nb_parallel * self._current_noise.shape[0],
                        replace=False,
                    )
                    % coord_batch.shape[-1]
                )
            except ValueError as error:
                if "Cannot take a larger sample than population when 'replace=False'" in str(error):
                    raise ValueError(
                        "Too many samples are requested for the random indices. Try to reduce the number of parallel"
                        "coordinate updates `nb_parallel`."
                    ) from error

                raise error

        for i in range(self.nb_parallel * self._current_noise.shape[0]):
            coord_batch[2 * i, indices[i]] += self.variable_h
            coord_batch[2 * i + 1, indices[i]] -= self.variable_h

        expanded_x = np.repeat(x, 2 * self.nb_parallel, axis=0).reshape((-1,) + x.shape[1:])
        expanded_targets = np.repeat(targets, 2 * self.nb_parallel, axis=0).reshape((-1,) + targets.shape[1:])
        expanded_c = np.repeat(c_batch, 2 * self.nb_parallel)
        _, _, loss = self._loss(
            expanded_x,
            expanded_x + coord_batch.reshape(expanded_x.shape),
            expanded_targets,
            expanded_c,
        )

        self._current_noise = self._optimizer_adam_coordinate(
            loss,
            indices,
            self.adam_mean,
            self.adam_var,
            self._current_noise,
            self.learning_rate,
            self.adam_epochs,
            True,
        )

        if self.use_importance:
            self._sample_prob = self._get_prob(self._current_noise).flatten()

        return x + self._current_noise

    def _optimizer_adam_coordinate(
        self,
        losses: np.ndarray,
        index: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        current_noise: np.ndarray,
        learning_rate: float,
        adam_epochs: np.ndarray,
        proj: bool,
    ) -> np.ndarray:
        beta1, beta2 = 0.9, 0.999

        grads = np.array([(losses[i] - losses[i + 1]) / (2 * self.variable_h) for i in range(0, len(losses), 2)])

        mean[index] = beta1 * mean[index] + (1 - beta1) * grads
        var[index] = beta2 * var[index] + (1 - beta2) * grads ** 2

        corr = (np.sqrt(1 - np.power(beta2, adam_epochs[index]))) / (1 - np.power(beta1, adam_epochs[index]))
        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * corr * mean[index] / (np.sqrt(var[index]) + 1e-8)
        adam_epochs[index] += 1

        if proj and hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            current_noise[index] = np.clip(current_noise[index], clip_min, clip_max)

        return current_noise.reshape(orig_shape)

    def _reset_adam(self, nb_vars: int, indices: Optional[np.ndarray] = None) -> None:
        if self.adam_mean is not None and self.adam_mean.size == nb_vars:
            if indices is None:
                self.adam_mean.fill(0)
                self.adam_var.fill(0)
                self.adam_epochs.fill(1)
            else:
                self.adam_mean[indices] = 0
                self.adam_var[indices] = 0
                self.adam_epochs[indices] = 1
        else:
            self.adam_mean = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_var = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_epochs = np.ones(nb_vars, dtype=int)

    def _get_prob(self, prev_noise: np.ndarray) -> np.ndarray:
        prob = np.abs(prev_noise)
        prob /= np.sum(prob)
        return prob

    def _check_params(self) -> None:
        if not isinstance(self.binary_search_steps, int) or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.nb_parallel, int) or self.nb_parallel < 1:
            raise ValueError("The number of parallel coordinates must be an integer greater than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
