import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm
import typing

class DAGMAOptimizer:
    def __init__(self,  X: np.ndarray,
                 loss_type: str,
            lambda1: float = 0.03, 
            w_threshold: float = 0.3, 
            T: int = 5,
            mu_init: float = 1.0, 
            mu_factor: float = 0.1, 
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6], 
            warm_iter: int = 3e4, 
            max_iter: int = 6e4, 
            lr: float = 0.0003, 
            checkpoint: int = 1000, 
            beta_1: float = 0.99, 
            beta_2: float = 0.999,
            exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None, 
            include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
            verbose: bool = False, dtype: type = np.float64) -> None:
        r"""
        Parameters
        ----------
        loss_type : str
            One of ["l2", "logistic"]. ``l2`` refers to the least squares loss, while ``logistic``
            refers to the logistic loss. For continuous data: use ``l2``. For discrete 0/1 data: use ``logistic``.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.linear.DagmaLinear.fit`. Defaults to ``False``.
        dtype : type, optional
           Defines the float precision, for large number of nodes it is recommened to use ``np.float64``. 
           Defaults to ``np.float64``.
        """
        super().__init__()
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.cov = X.T @ X / float(self.n)
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

    def _score(self, W: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value and gradient of the score function.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            loss value, and gradient of the loss function
        """
        if self.loss_type == 'l2':
            dif = self.Id - W 
            print(W.shape)
            rhs = self.cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            G_loss = -rhs
        elif self.loss_type == 'logistic':
            R = self.X @ W
            loss = 1.0 / self.n * (np.logaddexp(0, R) - self.X * R).sum()
            G_loss = (1.0 / self.n * self.X.T) @ sigmoid(R) - self.cov
        return loss, G_loss

    def _h(self, W: np.ndarray, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value and gradient of the logdet acyclicity constraint.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            h value, and gradient of h
        """
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W: np.ndarray, mu: float, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value of the penalized objective function.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        mu : float
            Weight of the score function.
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            Objective value, and gradient of the objective
        """
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h

    def _adam_update(self, grad: np.ndarray, iter: int, beta_1: float, beta_2: float) -> np.ndarray:
        r"""
        Performs one update of Adam.

        Parameters
        ----------
        grad : np.ndarray
            Current gradient of the objective.
        iter : int
            Current iteration number.
        beta_1 : float
            Adam hyperparameter.
        beta_2 : float
            Adam hyperparameter.

        Returns
        -------
        np.ndarray
            Updates the gradient by the Adam method.
        """
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad

    def minimize(self, 
                 W: np.ndarray, 
                 mu: float, 
                 max_iter: int, 
                 s: float, 
                 lr: float, 
                 tol: float = 1e-6, 
                 beta_1: float = 0.99, 
                 beta_2: float = 0.999, 
                 pbar: typing.Optional[tqdm] = None,
                 ) -> typing.Tuple[np.ndarray, bool]:        
        r"""
        Solves the optimization problem: 
            .. math::
                \arg\min_{W \in \mathbb{W}^s} \mu \cdot Q(W; \mathbf{X}) + h(W),
        where :math:`Q` is the score function. This problem is solved via (sub)gradient descent, where the initial
        point is `W`.

        Parameters
        ----------
        W : np.ndarray
            Initial point of (sub)gradient descent.
        mu : float
            Weights the score function.
        max_iter : int
            Maximum number of (sub)gradient iterations.
        s : float
            Number that controls the domain of M-matrices.
        lr : float
            Learning rate.
        tol : float, optional
            Tolerance to admit convergence. Defaults to 1e-6.
        beta_1 : float, optional
            Hyperparamter for Adam. Defaults to 0.99.
        beta_2 : float, optional
            Hyperparamter for Adam. Defaults to 0.999.
        pbar : tqdm, optional
            Controls bar progress. Defaults to ``tqdm()``.

        Returns
        -------
        typing.Tuple[np.ndarray, bool]
            Returns an adjacency matrix until convergence or `max_iter` is reached.
            A boolean flag is returned to point success of the optimization. This can be False when at any iteration, the current
            W point went outside of the domain of M-matrices.
        """
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')
        mask_inc = np.zeros((self.d, self.d))
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if self.exc_c is not None:
                mask_exc[self.exc_r, self.exc_c] = 0.
                
        for iter in range(1, max_iter+1):
            ## Compute the (sub)gradient of the objective
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0): # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                    return W, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f'Learning rate decreased to lr: {lr}')
            
            if self.loss_type == 'l2':
                G_score = -mu * self.cov @ (self.Id - W) 
            elif self.loss_type == 'logistic':
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T + mask_inc * np.sign(W)
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad
            W *= mask_exc
                
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f'\nInner iteration {iter}')
                self.vprint(f'\th(W_est): {h:.4e}')
                self.vprint(f'\tscore(W_est): {score:.4e}')
                self.vprint(f'\tobj(W_est): {obj_new:.4e}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, True

# 使用示例
optimizer = DAGMAOptimizer(X, W_initial, lr, mu, lambda1, ...)
optimizer.minimize()
