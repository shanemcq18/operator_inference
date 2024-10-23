# rom/_bayes.py
"""Classes supporting Bayesian operator inference."""

__all__ = [
    "OperatorPosterior",
    "BayesianROM",
    "BayesianParametricROM",
]

import warnings
import numpy as np
import scipy.linalg
import scipy.stats

from .. import errors, lstsq, post, utils
from ..models import _utils as modutils
from ._base import _identity
from ._nonparametric import ROM
from ._parametric import ParametricROM


VALID_SOLVERS = (
    lstsq.L2Solver,
    lstsq.L2DecoupledSolver,
    lstsq.TikhonovSolver,
    lstsq.TikhonovDecoupledSolver,
)


# Posterior ===================================================================
class OperatorPosterior:
    r"""Posterior distribution for operator matrices.

    Operator inference models are uniquely determined by operator matrices
    :math:`\Ohat\in\RR^{r\times d}` that concatenate the entries of all
    operators in the model. For example, the time-continuous model

    .. math::
       \ddt\qhat(t) = \chat + \Ahat\qhat(t) + \Hhat[\qhat(t)\otimes\qhat(t)]

    is uniquely determined by the operator matrix

    .. math::
       \Ohat = [~\chat~~\Ahat~~\Hhat~] \in \RR^{r \times d}.

    Typical *deterministic* operator inference learns a single operator matrix
    :math:`\Ohat` from state measurements, while *probabilistic* or *Bayesian*
    operator inference constructs a distribution of operator matrices,
    :math:`p(\Ohat)`. This class implements an operator matrix distribution
    where the rows of :math:`\Ohat` are multivariate Normal (Gaussian) random
    variables, i.e.,

    .. math::
       p(\ohat_{i}) = \mathcal{N}(\ohat_i\mid\bfmu_i,\bfSigma_i),
       \\
       \bfmu_i \in \RR^{d},
       \quad
       \bfSigma_i \in \RR^{d\times d},
       \quad
       i = 0, \ldots, r-1,

    where :math:`\ohat_i \in \RR^{d}` is the :math:`i`-th row of :math:`\Ohat`.

    The :class:`BayesianROM` and :class:`BayesianParametricROM` each have a
    ``posterior`` attribute that is an ``OperatorPosterior`` object.

    Parameters
    ----------
    means : list of r (d,) ndarrays
        Mean values for each row of the operator matrix.
    precisions : list of r (d, d) ndarrays
        **INVERSE** covariance matrices for each row of the operator matrix.
    alreadyinverted : bool
        If ``True``, assume ``precisions`` is the collection of covariance
        matrices, not their inverses.
    """

    def __init__(self, means, precisions, *, alreadyinverted=False):
        """Store and pre-process the distribution parameters."""
        if (r := len(means)) != (_r2 := len(precisions)):
            raise ValueError(f"len(means) = {r} != {_r2} = len(precisions)")

        self.__r = r
        self.__randomvariables = []

        for i in range(self.__r):
            # Verify dimensions.
            mean_i, cov_i = means[i], precisions[i]
            if not isinstance(mean_i, np.ndarray) or mean_i.ndim != 1:
                raise ValueError(f"means[{i}] should be a 1D ndarray")
            if not isinstance(cov_i, np.ndarray) or cov_i.ndim != 2:
                raise ValueError(f"precisions[{i}] should be a 2D ndarray")
            d = mean_i.shape[0]
            if cov_i.shape != (d, d):
                raise ValueError(f"means[{i}] and precisions[{i}] not aligned")

            # Make a multivariate Normal distribution for this operator row.
            if not alreadyinverted:
                cov_i = scipy.stats.Covariance.from_precision(cov_i)
            self.__randomvariables.append(
                scipy.stats.multivariate_normal(mean=mean_i, cov=cov_i)
            )

        # If operator rows are all the same size, wrap rvs() output as array.
        self.__rvsasarray = False
        d = means[0].size
        if all(mean.size == d for mean in means):
            self.__rvsasarray = True

    # Properties --------------------------------------------------------------
    @property
    def nrows(self) -> int:
        """Number of rows :math:`r` in the data matrix. This is also the state
        dimension of the corresponding model.
        """
        return self.__r

    @property
    def randomvariables(self) -> list:
        """Multivariate normal random variables for the rows of the operator
        matrix (see :class:`scipy.stats.multivariate_normal`).
        """
        return self.__randomvariables

    @property
    def means(self) -> list:
        r"""Mean vectors :math:`\bfmu_0,\ldots,\bfmu_{r-1}\in\RR^{d}` for the
        rows of the operator matrix.
        """
        return [rv.mean for rv in self.randomvariables]

    @property
    def covs(self) -> list:
        r"""Covariance matrices
        :math:`\bfSigma_0,\ldots,\bfSigma_{r-1}\in\RR^{d\times d}`
        for the rows of the operator matrix.
        """
        return [rv.cov for rv in self.randomvariables]

    def __eq__(self, other):
        if self.nrows != other.nrows:
            return False
        for m1, m2 in zip(self.means, other.means):
            if m1.shape != m2.shape or not np.all(m1 == m2):
                return False
        for C1, C2 in zip(self.covs, other.covs):
            if C1.shape != C2.shape or not np.all(C1 == C2):
                return False
        return True

    # Random draws ------------------------------------------------------------
    def rvs(self):
        r"""Draw a random operator matrix from the posterior operator
        distribution.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix sampled from :math:`p(\Ohat)`.
        """
        ohats = [rv.rvs()[0] for rv in self.randomvariables]
        return np.array(ohats) if self.__rvsasarray else ohats

    # Model persistance -------------------------------------------------------
    def save(self, savefile, overwrite=True):
        """Save the posterior operator distribution.

        Parameters
        ----------
        savefile : str
            File to save data to.
        overwrite : bool
            If False and ``savefile`` exists, raise an exception.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("state_dimension", data=[self.nrows])
            for i, (mean_i, cov_i) in enumerate(zip(self.means, self.covs)):
                hf.create_dataset(f"means_{i}", data=mean_i)
                hf.create_dataset(f"covs_{i}", data=cov_i)

    @classmethod
    def load(cls, loadfile):
        """Load a previously saved posterior operator distribution.

        Parameters
        ----------
        loadfile : str
            File to load data from.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            r = int(hf["state_dimension"][0])
            means = [hf[f"means_{i}"][:] for i in range(r)]
            covs = [hf[f"covs_{i}"][:] for i in range(r)]

        return cls(means, covs, alreadyinverted=True)


# Bayesian ROMs ===============================================================
class _BayesianROMMixin:
    """Mixin for ROM classes with a Bayesian operator posterior."""

    def __init__(self):
        self.__posterior = None
        self._validate_model_solver(self.model)

    @staticmethod
    def _validate_model_solver(model):
        if modutils.is_interpolatory(model):
            raise AttributeError(
                "Fully interpolatory parametric models are not supported "
                "for Bayesian ROMs"
            )
        if not hasattr(model, "solver") or not isinstance(
            model.solver, VALID_SOLVERS
        ):
            types = ", ".join(f"lstsq.{s.__name__}" for s in VALID_SOLVERS)
            raise AttributeError(
                "'model' must have a 'solver' attribute "
                f"of one of the following types: {types}"
            )

    @property
    def posterior(self) -> OperatorPosterior:
        """Posterior distribution for the operator matrices."""
        return self.__posterior

    def _initialize_posterior(self):
        means, precisions = self.model.solver.posterior()
        try:
            self.__posterior = OperatorPosterior(means, precisions)
        except np.linalg.LinAlgError as ex:
            if ex.args[0] == "Matrix is not positive definite":
                self.__posterior = None

    def _draw_operators(self):
        """Set the model operators to a draw from the operator posterior."""
        self.model._extract_operators(self.posterior.rvs())

    def fit_regselect_continuous(
        self,
        candidates: list,
        train_time_domains: list,
        parameters: list,
        states: list,
        ddts: list = None,
        input_functions: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        test_time_length: float = 0,
        stability_margin: float = 5.0,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
        **predict_options: dict,
    ):
        if not self._iscontinuous:
            raise AttributeError(
                "this method is for time-continuous models only, "
                "use fit_regselect_discrete()"
            )

        if parameters is None:
            states, ddts, input_functions = self._fix_single_trajectory(
                states, ddts, input_functions
            )

        # Validate arguments.
        if np.isscalar(train_time_domains[0]):
            train_time_domains = [train_time_domains] * len(states)
        for t, Q in zip(train_time_domains, states):
            if t.shape != (Q.shape[1],):
                raise errors.DimensionalityError(
                    "train_time_domains and states not aligned"
                )
        if input_functions is not None:
            if callable(input_functions):  # one global input function.
                input_functions = [input_functions] * len(states)
            if not callable(input_functions[0]):
                raise TypeError(
                    "argument 'input_functions' must be sequence of callables"
                )
            inputs = [  # evaluate the inputs over the time domain.
                np.column_stack([u(tt) for tt in t])
                for u, t in zip(input_functions, train_time_domains)
            ]
        else:
            inputs = None
        if test_time_length < 0:
            raise ValueError("argument 'test_time_length' must be nonnegative")
        if regularizer_factory is None:
            regularizer_factory = _identity
        processed_test_cases = self._process_test_cases(
            test_cases, utils.ContinuousRegTest
        )

        # Fit the model for the first time.
        states = self._fit_and_return_training_data(
            parameters=parameters,
            states=states,
            lhs=ddts,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )

        # Set up the regularization selection.
        shifts, limits = self._get_stability_limits(states, stability_margin)

        def unstable(_Q, ell, size):
            """Return ``True`` if the solution is unstable."""
            if _Q.shape[-1] != size:
                return True
            return np.abs(_Q - shifts[ell]).max() > limits[ell]

        # Extend the training time domains by the testing time length.
        if test_time_length > 0:
            time_domains = []
            for t_train in train_time_domains:
                dt = np.mean(np.diff(t_train))
                t_test = t_train[-1] + np.linspace(
                    dt,
                    dt + test_time_length,
                    int(test_time_length / dt),
                )
                time_domains.append(np.concatenate(((t_train, t_test))))
        else:
            time_domains = train_time_domains

        if input_functions is None:
            input_functions = [None] * len(states)
        loop_collections = [states, input_functions, time_domains]
        if is_parametric := parameters is not None:
            loop_collections.insert(0, parameters)

        def update_model(reg_params):
            """Reset the regularizer and refit the model operators."""
            self.model.solver.regularizer = regularizer_factory(reg_params)
            self._initialize_posterior()

        def training_error(reg_params):
            """Compute the training error for a single regularization
            candidate by solving the model, checking for stability, and
            comparing to available training data.
            """
            update_model(reg_params)
            if self.posterior is None:
                return np.inf

            # Pass stability checks.
            for tcase in processed_test_cases:
                for _ in range(num_posterior_draws):
                    self._draw_operators()
                    if not tcase.evaluate(self.model, **predict_options):
                        return np.inf

            # Compute training error.
            error = 0
            for ell, entries in enumerate(zip(*loop_collections)):
                if is_parametric:
                    params, Q, input_func, t = entries
                    predict_args = (params, Q[:, 0], t, input_func)
                else:
                    Q, input_func, t = entries
                    predict_args = (Q[:, 0], t, input_func)
                draws = []
                trainsize = Q.shape[-1]
                for _ in range(num_posterior_draws):
                    self._draw_operators()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = self.model.predict(
                            *predict_args, **predict_options
                        )
                    if unstable(solution, ell, t.size):
                        return np.inf
                    draws.append(solution[:, :trainsize])
                solution_train = np.mean(draws, axis=0)
                error += post.Lp_error(Q, solution_train, t[:trainsize])[1]
            return error / len(states)

        best_regularization = utils.gridsearch(
            training_error,
            candidates,
            gridsearch_only=gridsearch_only,
            label="regularization",
            verbose=verbose,
        )

        update_model(best_regularization)
        return self

    def fit_regselect_discrete(
        self,
        candidates: list,
        parameters: list,
        states: list,
        inputs: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        num_test_iters: int = 0,
        stability_margin: float = 5.0,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
    ):
        if self._iscontinuous:
            raise AttributeError(
                "this method is for fully discrete models only, "
                "use fit_regselect_continuous()"
            )

        if parameters is None:
            states, _, inputs = self._fix_single_trajectory(
                states, None, inputs
            )

        # Validate arguments.
        if num_test_iters < 0:
            raise ValueError(
                "argument 'num_test_iters' must be a nonnegative integer"
            )
        if inputs is not None:
            if len(inputs) != len(states):
                raise errors.DimensionalityError(
                    f"{len(states)} state trajectories but "
                    f"{len(inputs)} input trajectories detected"
                )
            for Q, U in zip(states, inputs):
                if U.shape[-1] < Q.shape[1] + num_test_iters:
                    raise ValueError(
                        "argument 'inputs' must contain enough data for "
                        f"{num_test_iters} iterations after the training data"
                    )
        if regularizer_factory is None:
            regularizer_factory = _identity
        processed_test_cases = self._process_test_cases(
            test_cases, utils.DiscreteRegTest
        )

        # Fit the model for the first time.
        states = self._fit_and_return_training_data(
            parameters=parameters,
            states=states,
            lhs=None,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )

        # Set up the regularization selection.
        shifts, limits = self._get_stability_limits(states, stability_margin)

        def unstable(_Q, ell):
            """Return ``True`` if the solution is unstable."""
            if np.isnan(_Q).any() or np.isinf(_Q).any():
                return True
            return np.any(np.abs(_Q - shifts[ell]).max() > limits[ell])

        # Extend the iteration counts by the number of testing iterations.
        num_iters = [Q.shape[-1] for Q in states]
        if num_test_iters > 0:
            num_iters = [n + num_test_iters for n in num_iters]

        if inputs is None:
            inputs = [None] * len(states)
        loop_collections = [states, inputs, num_iters]
        if is_parametric := parameters is not None:
            loop_collections.insert(0, parameters)

        def update_model(reg_params):
            """Reset the regularizer and refit the model operators."""
            self.model.solver.regularizer = regularizer_factory(reg_params)
            self._initialize_posterior()

        def training_error(reg_params):
            """Compute the mean training error for a single regularization
            candidate by solving the model, checking for stability, and
            comparing to available training data.
            """
            update_model(reg_params)
            if self.posterior is None:
                return np.inf

            # Pass stability checks.
            for tcase in processed_test_cases:
                for _ in range(num_posterior_draws):
                    self._draw_operators()
                    if not tcase.evaluate(self.model):
                        return np.inf

            # Solve the model, check for stability, and compute training error.
            error = 0
            for ell, entries in enumerate(zip(*loop_collections)):
                if is_parametric:
                    params, Q, U, niter = entries
                    predict_args = (params, Q[:, 0], niter, U)
                else:
                    Q, U, niter = entries
                    predict_args = (Q[:, 0], niter, U)
                draws = []
                for _ in range(num_posterior_draws):
                    self._draw_operators()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = self.model.predict(*predict_args)
                    if unstable(solution, ell):
                        return np.inf
                    draws.append(solution[:, : Q.shape[-1]])
                error += post.frobenius_error(Q, np.mean(draws, axis=0))[1]
            return error / len(states)

        best_regularization = utils.gridsearch(
            training_error,
            candidates,
            gridsearch_only=gridsearch_only,
            label="regularization",
            verbose=verbose,
        )

        update_model(best_regularization)
        return self


class BayesianROM(ROM, _BayesianROMMixin):
    """Probabilistic nonparametric reduced-order model."""

    def __init__(
        self,
        model,
        *,
        lifter=None,
        transformer=None,
        basis=None,
        ddt_estimator=None,
    ):
        ROM.__init__(
            self,
            model,
            lifter=lifter,
            transformer=transformer,
            basis=basis,
            ddt_estimator=ddt_estimator,
        )
        _BayesianROMMixin.__init__(self)

    def fit(
        self,
        states,
        lhs=None,
        inputs=None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
    ):
        ROM.fit(
            self,
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )
        self._initialize_posterior()

    def fit_regselect_continuous(
        self,
        candidates: list,
        train_time_domains: list,
        states: list,
        ddts: list = None,
        input_functions: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        test_time_length: float = 0,
        stability_margin: float = 5,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
        **predict_options: dict,
    ):
        return _BayesianROMMixin.fit_regselect_continuous(
            self,
            candidates=candidates,
            train_time_domains=train_time_domains,
            parameters=None,
            states=states,
            ddts=ddts,
            input_functions=input_functions,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            test_time_length=test_time_length,
            stability_margin=stability_margin,
            num_posterior_draws=num_posterior_draws,
            test_cases=test_cases,
            verbose=verbose,
            **predict_options,
        )

    def fit_regselect_discrete(
        self,
        candidates: list,
        states: list,
        inputs: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        num_test_iters: int = 0,
        stability_margin: float = 5,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
    ):
        return _BayesianROMMixin.fit_regselect_discrete(
            self,
            candidates=candidates,
            parameters=None,
            states=states,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            num_test_iters=num_test_iters,
            stability_margin=stability_margin,
            num_posterior_draws=num_posterior_draws,
            test_cases=test_cases,
            verbose=verbose,
        )

    def predict(self, state0, *args, **kwargs):
        self._draw_operators()
        return ROM.predict(self, state0, *args, **kwargs)


class BayesianParametricROM(ParametricROM, _BayesianROMMixin):
    """Probabilistic parametric reduced-order model."""

    def __init__(
        self,
        model,
        *,
        lifter=None,
        transformer=None,
        basis=None,
        ddt_estimator=None,
    ):
        ParametricROM.__init__(
            self,
            model,
            lifter=lifter,
            transformer=transformer,
            basis=basis,
            ddt_estimator=ddt_estimator,
        )
        _BayesianROMMixin.__init__(self)

    def fit(
        self,
        parameters,
        states,
        lhs=None,
        inputs=None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
    ):
        ParametricROM.fit(
            self,
            parameters=parameters,
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )
        self._initialize_posterior()

    def fit_regselect_continuous(
        self,
        candidates: list,
        train_time_domains: list,
        parameters: list,
        states: list,
        ddts: list = None,
        input_functions: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        test_time_length: float = 0,
        stability_margin: float = 5,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
        **predict_options: dict,
    ):
        return _BayesianROMMixin.fit_regselect_continuous(
            self,
            candidates=candidates,
            train_time_domains=train_time_domains,
            parameters=parameters,
            states=states,
            ddts=ddts,
            input_functions=input_functions,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            test_time_length=test_time_length,
            stability_margin=stability_margin,
            num_posterior_draws=num_posterior_draws,
            test_cases=test_cases,
            verbose=verbose,
            **predict_options,
        )

    def fit_regselect_discrete(
        self,
        candidates: list,
        parameters: list,
        states: list,
        inputs: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        num_test_iters: int = 0,
        stability_margin: float = 5,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
    ):
        return _BayesianROMMixin.fit_regselect_discrete(
            self,
            candidates=candidates,
            parameters=parameters,
            states=states,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            num_test_iters=num_test_iters,
            stability_margin=stability_margin,
            num_posterior_draws=num_posterior_draws,
            test_cases=test_cases,
            verbose=verbose,
        )

    def predict(self, parameter, state0, *args, **kwargs):
        self._draw_operators()
        return ParametricROM.predict(self, parameter, state0, *args, **kwargs)
