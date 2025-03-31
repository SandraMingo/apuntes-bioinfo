#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carlos M. Alaíz
"""
from IPython.display import display, Markdown
import inspect
from matplotlib import pyplot as plt, colors, lines
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.datasets import (
    load_digits,
    load_sample_images,
    make_blobs,
    make_circles,
    make_moons,
    make_regression,
)
from sklearn.linear_model import LinearRegression, Perceptron, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import time


def plot_dataset(x, y):
    if (len(x.shape) == 1) or (x.shape[1] == 1):
        plt.plot(x.ravel(), y, "*")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("Data")
    else:
        n_plot = x.shape[1]
        fig, axs = plt.subplots(ncols=n_plot, sharey=True)
        for i in range(n_plot):
            ax = axs[i]
            ax.plot(x[:, i], y, "*")
            ax.set_xlabel("$x_{%d}$" % (i + 1))
            if i == 0:
                ax.set_ylabel("$y$")
        plt.suptitle("Data")


def plot_split(x, y, splits):
    splits = list(splits)

    figsize = plt.rcParams["figure.figsize"]
    plt.figure(figsize=[figsize[0], figsize[1] * len(splits)])

    for fold, (ind_tr, ind_te) in enumerate(splits):
        plt.subplot(len(splits), 1, fold + 1)
        plt.plot(x[ind_tr], y[ind_tr], "*")
        plt.plot(x[ind_te], y[ind_te], "*")

        plt.title("Fold {}".format(fold + 1))
        plt.xticks([], [])
        plt.yticks([], [])

    plt.tight_layout()


def plot_split_clas(x, y, splits):
    splits = list(splits)

    values, counts = np.unique(y, return_counts=True)
    min_class = values[np.argmin(counts)]

    distribution = [np.sum([y[ind_te] == min_class]) for _, ind_te in splits]
    labels = ["Fold {}".format(fold) for fold in range(1, len(splits) + 1)]
    plt.pie(distribution, labels=labels, autopct="%1.0f%%")
    plt.title("Distribution of the Minority Class")


def plot_linear_model(x, y_r, w_e, b_e, w_r=None, b_r=None):
    if np.isscalar(w_e) or (len(w_e) == 1):
        y_p = w_e * x + b_e

        plt.plot(x, y_r, "*", label="Obs.")
        plt.plot(x, y_p, "-", label="Pred")

        for i in range(len(x)):
            plt.plot([x[i].item(), x[i].item()], [y_p[i].item(), y_r[i].item()], ":k")

        if (w_r is not None) and (b_r is not None):
            plt.plot(x, w_r * x + b_r, "--k", label="Real")
            plt.legend()

        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title(
            "$y = %.2f x + %.2f$ (MSE: %.2f, MAE: %.2f, R2: %.2f)"
            % (
                w_e,
                b_e,
                mean_squared_error(y_r, y_p),
                mean_absolute_error(y_r, y_p),
                r2_score(y_r, y_p),
            )
        )
    else:
        y_p = x @ w_e + b_e
        pos = np.arange(len(w_e) + 1)

        if (w_r is not None) and (b_r is not None):
            plt.bar(pos - 0.2, np.append(w_e, b_e), width=0.4, alpha=0.5, label="Est.")
            plt.bar(pos + 0.2, np.append(w_r, b_r), width=0.4, alpha=0.5, label="Real")
            plt.legend()
        else:
            plt.bar(pos, np.append(w_e, b_e), alpha=0.5, label="Est.")

        plt.grid()
        labels = []
        for i in range(len(w_e)):
            labels.append("$w_%d$" % (i + 1))
        labels.append("$b$")
        plt.xticks(pos, labels)
        plt.title(
            "MSE: %.2f, MAE: %.2f, R2: %.2f"
            % (
                mean_squared_error(y_r, y_p),
                mean_absolute_error(y_r, y_p),
                r2_score(y_r, y_p),
            )
        )


def evaluate_linear_model(x_tr, y_tr_r, x_te, y_te_r, w, b, plot=False):
    if np.isscalar(w):
        y_tr_p = w * x_tr + b
        y_te_p = w * x_te + b
    else:
        y_tr_p = x_tr @ w.ravel() + b
        y_te_p = x_te @ w.ravel() + b

    er_tr = [
        mean_squared_error(y_tr_r, y_tr_p),
        mean_absolute_error(y_tr_r, y_tr_p),
        r2_score(y_tr_r, y_tr_p),
    ]
    er_te = [
        mean_squared_error(y_te_r, y_te_p),
        mean_absolute_error(y_te_r, y_te_p),
        r2_score(y_te_r, y_te_p),
    ]

    ers = [er_tr, er_te]
    headers = ["MSE", "MAE", "R2"]

    print("%10s" % "", end="")
    for h in headers:
        print("%10s" % h, end="")
    print("")

    headersc = ["Train", "Test"]

    cnt = 0
    for er in ers:
        hc = headersc[cnt]
        cnt = cnt + 1
        print("%10s" % hc, end="")

        for e in er:
            print("%10.2f" % e, end="")
        print("")

    if plot:
        plot_linear_model(x_te, y_te_r, w.ravel(), b)


def plot_dataset_clas(x, y):
    if len(x.shape) == 1:
        plt.plot(x, y, "*")
        plt.xlabel("$x$")
        clas = np.unique(y)
        plt.yticks(clas)
    elif x.shape[1] == 1:
        x = np.column_stack((x.ravel(), np.zeros(x.shape[0])))
    if len(np.unique(y)) == 2:
        ind = y == 1
        plt.scatter(x[ind, 0], x[ind, 1], c="b", zorder=100)
        ind = y != 1
        plt.scatter(x[ind, 0], x[ind, 1], c="r", zorder=100)
    else:
        plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.axis("equal")
    plt.title("Data")


def order_points(points):
    if len(points) == 0:
        return []
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 0] - centre[0], points[:, 1] - centre[1])
    o_points = points[np.argsort(angles), :]

    return np.vstack((o_points, o_points[0, :]))


def plot_linear_model_clas(x, y_r, w, b):
    if (len(x.shape) == 1) or (x.shape[1] != 2):
        raise ValueError("only_r two-dimensional problems can be represented")

    y_p = np.sign(x @ w + b)

    plot_dataset_clas(x, y_r)
    ax = plt.axis("equal")
    lims = np.array([ax[0] - 100, ax[1] + 100, ax[2] - 100, ax[3] + 100])

    if w[1] != 0:
        x1 = lims[0:2]
        x2 = -(w[0] * x1 + b) / w[1]
    else:
        x2 = lims[2:]
        x1 = -(w[1] * x2 + b) / w[0]

    points = np.column_stack(
        (
            np.append(x1, [lims[0], lims[1], lims[0], lims[1]]),
            np.append(x2, [lims[2], lims[3], lims[3], lims[2]]),
        )
    )

    points_p = order_points(points[points @ w + b >= -1e-2])
    if len(points_p) > 0:
        plt.fill(points_p[:, 0], points_p[:, 1], "b", alpha=0.3)

    points_n = order_points(points[points @ w + b <= +1e-2])
    if len(points_n) > 0:
        plt.fill(points_n[:, 0], points_n[:, 1], "r", alpha=0.3)

    plt.plot(x1, x2, "-k")
    plot_dataset_clas(x, y_r)
    plt.axis(ax)

    plt.title(
        "$y = %.2f x_1 + %.2f x_2 + %.2f$ (Acc: %.2f%%)"
        % (w[0], w[1], b, 100 * accuracy_score(y_r, y_p))
    )


def fun_cross_entropy(X, y, w):
    y_b = y.copy()
    y_b[y_b == -1] = 0

    y_p = 1 / (1 + np.exp(-X @ w))

    return (-(1 - y_b) * np.log(1 - y_p) - y_b * np.log(y_p)).sum()


def grad_cross_entropy(X, y, w):
    y_b = y.copy()
    y_b[y_b == -1] = 0

    y_p = 1 / (1 + np.exp(-X @ w))

    return X.T @ (y_p - y_b)


def fit_polylinear_regression(x, y, deg=1):
    X = np.power(np.reshape(x, (len(x), 1)), np.arange(1, deg + 1))
    model = LinearRegression()
    model.fit(X, y)

    return model


def pred_polylinear_regression(model, x):
    X = np.power(np.reshape(x, (len(x), 1)), np.arange(1, len(model.coef_) + 1))
    return model.predict(X)


def plot_polylinear_model(x, y_r, model):
    xv = np.linspace(x.min(), x.max())
    plt.plot(x, y_r, "*", label="Obs.")
    plt.plot(xv, pred_polylinear_regression(model, xv), "-", label="Pred")

    y_p = pred_polylinear_regression(model, x)

    for i in range(len(x)):
        plt.plot([x[i].item(), x[i].item()], [y_p[i].item(), y_r[i].item()], ":k")

        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title(
            "Degree: %d (MSE: %.2f, MAE: %.2f, R2: %.2f)"
            % (
                len(model.coef_),
                mean_squared_error(y_r, y_p),
                mean_absolute_error(y_r, y_p),
                r2_score(y_r, y_p),
            )
        )


def norm_p(w, p):
    if p == 0:
        return np.count_nonzero(w)

    if p == np.inf:
        return np.max(np.abs(w))

    nw = np.sum(np.power(np.abs(w), p))
    if p > 1:
        nw = np.power(nw, 1 / p)
    return nw


def plot_contour_lp(p, mini=-3, maxi=3, npoi=21):
    x = np.linspace(mini, maxi, npoi)
    y = np.linspace(mini, maxi, npoi)
    x, y = np.meshgrid(x, y)

    z = np.apply_along_axis(norm_p, 2, np.stack([x, y], axis=2), p)
    plt.contour(x, y, z)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal", "box")
    plt.title("Norm $\\ell_{%g}$" % p)

    plt.grid()
    plt.show()


def plot_contour_l1_l2(l1_ratio=0.5, mini=-3, maxi=3, npoi=21):
    x = np.linspace(mini, maxi, npoi)
    y = np.linspace(mini, maxi, npoi)
    x, y = np.meshgrid(x, y)

    z = l1_ratio * np.apply_along_axis(norm_p, 2, np.stack([x, y], axis=2), 1) + (
        1 - l1_ratio
    ) * np.apply_along_axis(norm_p, 2, np.stack([x, y], axis=2), 2)
    plt.contour(x, y, z)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal", "box")
    plt.title("%g * Norm $\\ell_1$ + %g * Norm $\\ell_2$" % (l1_ratio, 1 - l1_ratio))

    plt.grid()
    plt.show()


def plot_contour_linear_lp(X, y, p=None, mini=-3, maxi=3, npoi=51):
    def mse_linear(w):
        return mean_squared_error(y, X @ w)

    x1 = np.linspace(mini, maxi, npoi)
    x2 = np.linspace(mini, maxi, npoi)
    x1, x2 = np.meshgrid(x1, x2)

    z = np.apply_along_axis(mse_linear, 2, np.stack([x1, x2], axis=2))
    plt.contour(x1, x2, z, 30)

    if p is not None:
        x = np.linspace(-1, 1, 101)
        if p == 0:
            plt.plot([-1, 1], [0, 0], "-k")
            plt.plot([0, 0], [-1, 1], "-k")
            ball = np.abs(x1) + np.abs(x2) <= 1
        elif p == np.inf:
            plt.plot(
                [-1, 1, 1, -1, -1],
                [
                    1,
                    1,
                    -1,
                    -1,
                    1,
                ],
                "-k",
            )
            plt.fill(
                [-1, 1, 1, -1, -1],
                [
                    1,
                    1,
                    -1,
                    -1,
                    1,
                ],
                "k",
            )
            ball = np.maximum(x1, x2) <= 1
        else:
            y = np.power(1 - np.power(np.abs(x), p), 1 / p)
            plt.plot(
                np.concatenate((x, np.flip(x))), np.concatenate((y, np.flip(-y))), "-k"
            )
            plt.fill(
                np.concatenate((x, np.flip(x))), np.concatenate((y, np.flip(-y))), "k"
            )
            ball = np.power(np.abs(x1), p) + np.power(np.abs(x2), p) <= 1

        obj = z
        obj[ball == False] = np.inf
    else:
        obj = z

    ind = np.unravel_index(np.argmin(obj), obj.shape)
    plt.plot(x1[ind[0], ind[1]], x2[ind[0], ind[1]], "r*")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.gca().set_aspect("equal", "box")
    if p is not None:
        plt.title("Norm $\\ell_{%g}$ + MSE" % p)
    else:
        plt.title("MSE")

    plt.grid()
    plt.show()


def generate_bv_example(n_rep=250, n_mod=15, n_dim=10, noise=3e-1, seed=1234):
    n_pat = n_dim
    alpha_v = np.logspace(-4, 4, n_mod)

    np.random.seed(seed)

    w = np.random.randn(n_dim)
    x_te = np.random.randn(n_pat, n_dim)
    y_te = x_te @ w + noise * np.random.randn(n_pat)

    distances = np.zeros((n_mod, n_rep))
    predictions = np.zeros((n_mod, n_rep, n_pat))
    for i, alpha in enumerate(alpha_v):
        for j in range(n_rep):
            x_tr = np.random.randn(n_pat, n_dim)
            y_tr = x_tr @ w + noise * np.random.randn(n_pat)
            y_te_p = (
                Ridge(alpha=alpha, fit_intercept=False).fit(x_tr, y_tr).predict(x_te)
            )
            predictions[i, j, :] = y_te_p
            distances[i, j] = mean_squared_error(y_te, y_te_p)

    return distances, predictions, y_te


def plot_perceptron_evo_epochs(x, y, max_epochs=5):
    import warnings

    warnings.filterwarnings("ignore", category=Warning)

    fig, ax = plt.subplots(nrows=1, ncols=max_epochs)
    for i in range(max_epochs):
        model = Perceptron(tol=-1, max_iter=i + 1)
        model.fit(x, y)

        plt.sca(ax[i])
        plot_linear_model_clas(x, y, model.coef_[0], model.intercept_)
        if i > 0:
            ax[i].set_yticklabels("")
            ax[i].set_ylabel("")
        plt.title(
            "Epoch %d (Acc: %.2f%%)"
            % (i + 1, 100 * accuracy_score(y, model.predict(x)))
        )
    plt.tight_layout()


def plot_perceptron_evo_iter(x, y, max_iters=5):
    import warnings

    warnings.filterwarnings("ignore", category=Warning)

    n_pat = x.shape[0]
    n_dim = x.shape[1]
    w = np.zeros(n_dim + 1)
    w[0] = 1e-1
    x_b = np.column_stack((np.ones(n_pat), x))

    nrows = int(np.ceil(max_iters / 5))
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    ax = ax.ravel()
    for i in range(max_iters):
        x_i = x_b[i % n_pat, :]
        y_i = y[i % n_pat]

        pred = np.sign(w @ x_i)

        plt.sca(ax[i])
        plot_linear_model_clas(x, y, w[1:], w[0])
        plt.scatter(
            x_i[1],
            x_i[2],
            s=200,
            linewidth=4,
            facecolors="none",
            edgecolors="k",
            zorder=100,
        )
        if i % ncols > 0:
            ax[i].set_yticklabels("")
            ax[i].set_ylabel("")
        if i < (nrows - 1) * ncols:
            ax[i].set_xticklabels("")
            ax[i].set_xlabel("")
        plt.title(
            "Iter. %d (Acc: %.2f%%)"
            % (i + 1, 100 * accuracy_score(y, np.sign(x_b @ w)))
        )

        w += (y_i - pred) / 2 * x_i

    plt.tight_layout()


def plot_nonlinear_model(x, y_r, model, phi=None):
    if phi is None:

        def phi(x):
            return np.reshape(x, (-1, 1))

    y_p = model.predict(phi(x))

    plt.plot(x, y_r, "*", label="Obs.")
    plt.plot(x, y_p, "-", label="Pred")

    for i in range(len(x)):
        plt.plot([x[i].item(), x[i].item()], [y_p[i].item(), y_r[i].item()], ":k")

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(
        "(MSE: %.2f, MAE: %.2f, R2: %.2f)"
        % (
            mean_squared_error(y_r, y_p),
            mean_absolute_error(y_r, y_p),
            r2_score(y_r, y_p),
        )
    )


def plot_nonlinear_model_tf(x, y_r, model, verbose=0):
    y_p = model.predict(x, verbose=verbose)

    plt.plot(x, y_r, "*", label="Obs.")
    plt.plot(x, y_p, "-", label="Pred")

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(
        "(MSE: %.2f, MAE: %.2f, R2: %.2f)"
        % (
            mean_squared_error(y_r, y_p),
            mean_absolute_error(y_r, y_p),
            r2_score(y_r, y_p),
        )
    )


def plot_nonlinear_model_clas(x, y_r, model, phi=None, n_points=31):
    if phi is None:
        if x.ndim == 1:

            def phi(x):
                return np.reshape(x, (-1, 1))

        else:

            def phi(x):
                return x

    alpha = 0.3
    col_1 = np.array([31, 119, 180]) / 255
    col_2 = np.array([214, 39, 40]) / 255

    y_p = model.predict(phi(x))

    ind = y_r < 0
    plt.scatter(x[ind, 0], x[ind, 1], c=[col_1], zorder=100)
    ind = y_r >= 0
    plt.scatter(x[ind, 0], x[ind, 1], c=[col_2], zorder=100)
    ax = plt.axis("equal")

    x_1 = np.linspace(plt.xlim()[0], plt.xlim()[1], n_points)
    x_1 = np.hstack((x_1[0] - 100, x_1, x_1[-1] + 100))
    x_2 = np.linspace(plt.ylim()[0], plt.ylim()[1], n_points)
    x_2 = np.hstack((x_2[0] - 100, x_2, x_2[-1] + 100))

    x_1, x_2 = np.meshgrid(x_1, x_2, indexing="ij")

    plt.pcolormesh(
        x_1,
        x_2,
        np.reshape(
            model.predict(phi(np.column_stack((x_1.ravel(), x_2.ravel())))), x_1.shape
        ),
        shading="auto",
        cmap=colors.ListedColormap(
            [alpha * col_1 + 1 - alpha, alpha * col_2 + 1 - alpha]
        ),
    )

    plt.axis(ax)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.title("(Acc: %.2f%%)" % (100 * accuracy_score(y_r, y_p)))


def basis_polynomial(X, deg):
    def phi(x):
        return np.power(x, np.arange(0, deg + 1))

    return np.array([phi(x) for x in X])


def basis_gaussian(X, mu, sigma):
    sigma_2 = np.power(sigma, 2)

    def phi(x):
        return np.exp(-np.power(x - mu, 2) / sigma_2)

    return np.array([phi(x) for x in X])


def basis_sigmoidal(X, a, b):
    def phi(x):
        return 1 / (1 + np.exp(-(a * x - b)))

    return np.array([phi(x) for x in X])


def plot_krr_coefficients(model, label_gap=5):
    coef = model.dual_coef_
    pos = np.arange(len(coef))
    plt.bar(pos, coef, alpha=0.5)

    plt.grid()
    labels = []
    for i in range(len(coef)):
        labels.append("$\\alpha_{%d}$" % (i + 1))
    plt.xticks(pos[::label_gap], labels[::label_gap])
    plt.ylabel("Value")
    plt.title("Dual Coefficients")


def plot_svc(x, y, model, n_points=151, plot_slack=False):
    alpha = 0.3
    col_1 = np.array([31, 119, 180]) / 255
    col_2 = np.array([214, 39, 40]) / 255

    ind = y != 1
    plt.scatter(x[ind, 0], x[ind, 1], c="r", s=30, zorder=100)
    ind = y == 1
    plt.scatter(x[ind, 0], x[ind, 1], c="b", s=30, zorder=100)

    lims = plt.axis("equal")

    xx = np.linspace(
        lims[0] - 1.1 * (lims[1] - lims[0]),
        lims[1] + 1.1 * (lims[1] - lims[0]),
        n_points,
    )
    yy = np.linspace(lims[2], lims[3], n_points)
    yy, xx = np.meshgrid(yy, xx)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    zz = model.decision_function(xy).reshape(xx.shape)

    plt.pcolormesh(
        xx,
        yy,
        np.sign(zz),
        shading="auto",
        cmap=colors.ListedColormap(
            [alpha * col_2 + 1 - alpha, alpha * col_1 + 1 - alpha]
        ),
    )
    plt.contour(
        xx,
        yy,
        zz,
        colors=["r", "k", "b"],
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        linewidths=[2, 4, 2],
    )
    plt.legend(
        handles=[
            lines.Line2D([], [], color="r", linestyle="--", label="Support Hyp. $-1$"),
            lines.Line2D([], [], color="k", linestyle="-", label="Sepparating Hyp."),
            lines.Line2D([], [], color="b", linestyle="--", label="Support Hyp. $+1$"),
        ]
    )

    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=3,
        facecolors="none",
        edgecolors="k",
    )

    if plot_slack:
        w = model.coef_[0]
        b = model.intercept_
        nws = np.linalg.norm(w) ** 2
        for i in model.support_:
            p = x[i, :] - (w @ x[i, :] + b - y[i]) / nws * w
            style = "b:" if y[i] == 1 else "r:"
            plt.plot([p[0], x[i, 0]], [p[1], x[i, 1]], style)

    plt.axis(lims)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("SVM (%s, C=%.2g)" % (model.kernel, model.C))


def plot_all_linear_separators(x, y, plot_best=False, n_points=51):
    ang_vec = np.linspace(0, 2 * np.pi, n_points)
    b_vec = np.linspace(-5, 5, n_points)

    ang_mat, b_mat = np.meshgrid(ang_vec, b_vec, indexing="ij")
    ws = []
    bs = []
    ms = []
    svs = []

    for i_ang in range(len(ang_vec)):
        ang = ang_vec[i_ang]
        for i_b in range(len(b_vec)):
            b = b_vec[i_b]
            w = np.array([np.sin(ang), np.cos(ang)])
            d = np.abs(x @ w + b) / np.linalg.norm(w)
            m = d.min()
            sv = np.argsort(d)[:3]
            y_p = np.sign(x @ w + b)
            if accuracy_score(y, y_p) == 1:
                ws.append(w)
                bs.append(b)
                ms.append(m)
                svs.append(sv)

    plot_dataset_clas(x, y)
    lims = plt.axis()

    max_m = np.array(ms).max()
    for w, b, m, sv in zip(ws, bs, ms, svs):
        if w[1] != 0:
            x1 = np.asarray(lims[0:2])
            x1[0] -= 1.1 * (lims[1] - lims[0])
            x1[1] += 1.1 * (lims[1] - lims[0])
            x2 = -(w[0] * x1 + b) / w[1]
        else:
            x2 = lims[2:]
            x1 = -(w[1] * x2 + b) / w[0]

        if plot_best:
            if m == max_m:
                plt.plot(x1, x2, "-k", alpha=1.0)
                plt.scatter(
                    x[sv, 0],
                    x[sv, 1],
                    s=100,
                    linewidth=3,
                    facecolors="none",
                    edgecolors="k",
                )
            else:
                plt.plot(x1, x2, "-k", alpha=0.3)
        else:
            plt.plot(x1, x2, "-k", alpha=0.3)

    plt.axis(lims)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Linear Classifiers")


def plot_svr(x, y, model, n_points=151, plot_slack=False):
    x_e = np.linspace(x.min(), x.max(), n_points)
    np.append(x_e, x)
    x_e.sort()

    y_p = model.predict(x_e.reshape(-1, 1))
    y_pi = model.predict(x.reshape(-1, 1))

    plt.plot(x, y, "*", label="Obs.")
    plt.plot(x_e, y_p, "-", label="Model")
    plt.plot(x_e, y_p + model.epsilon, "--k")
    plt.plot(x_e, y_p - model.epsilon, "--k")

    plt.scatter(
        x[model.support_],
        y[model.support_],
        s=100,
        linewidth=3,
        facecolors="none",
        edgecolors="k",
    )

    if plot_slack:
        for i in range(len(x)):
            if y_pi[i] > y[i] + model.epsilon:
                plt.plot(
                    [x[i].item(), x[i].item()],
                    [y_pi[i].item() - model.epsilon, y[i].item()],
                    ":k",
                )
            if y_pi[i] < y[i] - model.epsilon:
                plt.plot(
                    [x[i].item(), x[i].item()],
                    [y_pi[i].item() + model.epsilon, y[i].item()],
                    ":k",
                )

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("SVM (%s, C=%.2g)" % (model.kernel, model.C))


def evaluate_nonlinear_model(x_tr, y_tr_r, x_te, y_te_r, model):
    y_tr_p = model.predict(x_tr)
    y_te_p = model.predict(x_te)

    er_tr = [
        mean_squared_error(y_tr_r, y_tr_p),
        mean_absolute_error(y_tr_r, y_tr_p),
        r2_score(y_tr_r, y_tr_p),
    ]
    er_te = [
        mean_squared_error(y_te_r, y_te_p),
        mean_absolute_error(y_te_r, y_te_p),
        r2_score(y_te_r, y_te_p),
    ]

    ers = [er_tr, er_te]
    headers = ["MSE", "MAE", "R2"]

    print("%10s" % "", end="")
    for h in headers:
        print("%10s" % h, end="")
    print("")

    headersc = ["Train", "Test"]

    cnt = 0
    for er in ers:
        hc = headersc[cnt]
        cnt = cnt + 1
        print("%10s" % hc, end="")

        for e in er:
            print("%10.2f" % e, end="")
        print("")


def generate_rl_sequence(env, early_stop=True, model=None, n_steps=500):
    import tensorflow as tf

    state = env.reset()[0]
    for i in range(n_steps):
        env.render()
        if model is None:
            action = env.action_space.sample()
        else:
            action_probs, _ = model.predict(
                tf.expand_dims(tf.convert_to_tensor(state), 0), verbose=0
            )
            p = np.squeeze(action_probs)
            action = np.random.choice(len(p), p=p)
        ret = env.step(action)
        state = ret[0]
        done = ret[2]
        if early_stop and done:
            print("Finished after {} steps".format(i))
            break


def generate_dataset_lr(seed, noise=0.5):
    n_pat = 50

    np.random.seed(seed)

    w = np.random.randn()
    b = np.random.randn()

    x = np.linspace(0, 10, n_pat)
    y = w * x + b + noise * np.random.randn(n_pat)

    return x, y, w, b


def generate_dataset_lr_mv(seed, n_dim=3, noise=0.5, w=None, b=None):
    n_pat = 50

    np.random.seed(seed)

    if w is None:
        w = np.random.randn(n_dim)
    if b is None:
        b = np.random.randn()

    x = np.random.randn(n_pat, n_dim)
    y = x @ w + b + noise * np.random.randn(n_pat)

    return x, y, w, b


def generate_dataset_lbc(seed, sep=3):
    n_pat = 50

    np.random.seed(seed)

    x1 = np.random.randn(n_pat, 2)
    x2 = np.random.randn(n_pat, 2)
    sep_dir = np.random.randn(2)
    x = np.vstack((x1, x2 + sep * sep_dir / np.linalg.norm(sep_dir)))
    y = np.append(-np.ones(n_pat), np.ones(n_pat))

    return x, y


def generate_dataset_lbc_asym(seed):
    n_pat = 50
    scale = 15

    np.random.seed(seed)

    x1 = np.random.randn(n_pat, 2)
    x2 = np.random.randn(n_pat, 2) * [scale, 1]
    sep_dir = [1.5 * scale, 0]
    x = np.vstack((x1, x2 + sep_dir))
    y = np.append(-np.ones(n_pat), np.ones(n_pat))

    return x, y


def generate_dataset_lbc_lines(seed, n_pat=50):
    np.random.seed(seed)

    t = np.linspace(0, 1, n_pat + 1)[1:]
    x = np.vstack(
        (
            np.column_stack((t, t)),
            np.column_stack((t, 0.9 * t)),
        )
    )
    y = np.repeat([-1, 1], n_pat)

    return x, y


def generate_dataset_square(seed, n_pat=10, noise=0.1):
    np.random.seed(seed)

    x = np.linspace(-2, 2, n_pat)
    y = np.square(x) + noise * np.random.randn(n_pat)

    return x, y


def generate_dataset_uninformative(seed):
    x, y = make_regression(random_state=seed, noise=5e0)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=1.0 / 3.0, random_state=seed
    )

    return x_tr, x_te, y_tr, y_te


def generate_dataset_quadrant(seed, n_pat=200, noise=0.1):
    np.random.seed(seed)

    x = np.random.randn(n_pat, 2)
    y = np.sign(x[:, 0] * x[:, 1])

    return x, y


def generate_dataset_r_mv_1(seed):
    x, y = make_regression(
        n_samples=30, n_features=5, n_informative=5, random_state=seed
    )

    return x, y


def generate_dataset_r_mv_2(seed):
    np.random.seed(seed)

    x = np.random.randn(30, 5)
    y = x[:, 0] ** 2 + x[:, 1] * x[:, 2]

    return x, y


def generate_dataset_r_nl_1(seed, n_pat=100, scale=1.0):
    np.random.seed(seed)

    x = np.linspace(-2, 2, n_pat)
    y = scale * (np.square(x) + np.sin(5 * x) + 0.2 * np.random.randn(len(x)))

    return x.reshape(-1, 1), y


def generate_dataset_r_nl_2(seed, n_pat=100, noise=2e-1):
    np.random.seed(seed)

    x = np.linspace(-2.5, 4, n_pat)
    y = (
        np.square(x)
        + np.power(x / 2, 3)
        + 5 * np.sin(2 * x)
        + noise * np.random.randn(n_pat)
    )

    return x.reshape(-1, 1), y


def generate_dataset_blobs(seed, n_pat=100, n_dim=2, std=2.0, centers=None):
    x, y = make_blobs(
        n_samples=n_pat,
        n_features=n_dim,
        cluster_std=std,
        random_state=seed,
        centers=centers,
    )
    y[y != 1] = -1

    return x, y


def generate_dataset_blobs_long(seed, n_pat=100):
    x, y = make_blobs(n_samples=n_pat, centers=[[-3, 0], [3, 0]], random_state=seed)
    y[y != 1] = -1
    x[:, 1] = 0.25 * x[:, 1]

    return x, y


def generate_dataset_blobs_clus(seed, n_pat=1000):
    x, y = make_blobs(n_samples=n_pat, centers=3, n_features=2, random_state=seed)

    return x, y


def generate_dataset_blobs_clus_asym(seed, n_pat=1000):
    x, y = generate_dataset_blobs_clus(int(1.4 * seed), n_pat=n_pat)
    x = x @ np.array([[0.6, -0.6], [-0.4, 0.85]])

    return x, y


def generate_dataset_outliers(seed, n_pat=100, n_out=5):
    np.random.seed(seed)

    x = 1e-1 * np.random.randn(n_pat - n_out, 2) + 1
    x = np.vstack((x, 1e-1 * np.random.randn(n_out, 2)))
    y = np.ones(n_pat)

    return x, y


def generate_dataset_ring(seed, n_pat=500, noise=1e-1):
    x, y = make_circles(n_samples=n_pat, noise=noise, random_state=seed)
    x = x[y == 1] + 10
    y = y[y == 1]

    return x, y


def generate_dataset_rings(seed, n_pat=500, noise=2e-2):
    x, y = make_circles(n_samples=n_pat, noise=noise, factor=0.2, random_state=seed)

    return x, y


def generate_dataset_rings_ub(seed, noise=2e-2):
    x, y = make_circles(n_samples=(500, 10), noise=noise, factor=0.2, random_state=seed)

    return x, y


def generate_dataset_and():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    return x, y


def generate_dataset_or():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, 1])

    return x, y


def generate_dataset_xor():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])

    return x, y


def generate_dataset_moons(seed, n_pat=200, noise=0.1):
    x, y = make_moons(n_samples=n_pat, noise=noise, random_state=seed)
    y[y == 0] = -1

    return x, y


def generate_dataset_skmnist(seed, enlarge=True):
    x_orig, y = load_digits(n_class=10, return_X_y=True)

    if enlarge:
        x = []
        for img in x_orig:
            img = (
                img.reshape(8, 8)
                .repeat(10, axis=0)
                .repeat(10, axis=1)[:, :, None]
                .repeat(3, axis=2)
                .reshape(80, 80, 3)
            )
            img = img / 8.0 - 1.0
            x.append(img)
        x = np.array(x)
    else:
        x = x_orig

    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.6, random_state=seed)

    return x_tr, x_va, y_tr, y_va


def generate_dataset_mnist(vectorize=True, plot=False):
    from tensorflow import keras

    (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()

    # Samples.
    if plot:
        labels = np.unique(y_tr)
        for i, label in enumerate(labels):
            plt.subplot(1, len(labels), i + 1)
            plt.imshow(x_tr[y_tr == label][0])
            plt.axis("off")

    # The pixels are transformed to the interval [0, 1].
    x_tr = x_tr.astype("float32") / 255.0
    x_te = x_te.astype("float32") / 255.0

    if vectorize:
        # Each image is converted into a 1-dimensional vector.
        x_tr = x_tr.reshape(len(x_tr), -1)
        x_te = x_te.reshape(len(x_te), -1)

    return x_tr, x_te, y_tr, y_te


def generate_dataset_fminst(describe=False, plot=False, scale=False):
    from tensorflow import keras

    (x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()

    if describe:
        print("Training size:", x_tr.shape)
        print("Target size:  ", y_tr.shape)
        print("Maximum:      ", x_tr.max())
        print("Minimum:      ", x_tr.min())

    if plot:
        plt.imshow(x_tr[0])
        plt.axis("off")
        plt.title("Class {}".format(y_tr[0]))
        plt.show()

    if scale:
        x_tr = x_tr / 255.0
        x_te = x_te / 255.0

    return x_tr, x_te, y_tr, y_te


def generate_dataset_categorical():
    df = pd.DataFrame(
        [
            ["Rainy", "Hot", "High", "False", "No"],
            ["Rainy", "Hot", "High", "True", "No"],
            ["Overcast", "Hot", "High", "False", "Yes"],
            ["Sunny", "Mild", "High", "False", "Yes"],
            ["Sunny", "Cool", "Normal", "False", "Yes"],
            ["Sunny", "Cool", "Normal", "True", "No"],
            ["Overcast", "Cool", "Normal", "True", "Yes"],
            ["Rainy", "Mild", "High", "False", "No"],
            ["Rainy", "Cool", "Normal", "False", "Yes"],
            ["Sunny", "Mild", "Normal", "False", "Yes"],
            ["Rainy", "Mild", "Normal", "True", "Yes"],
            ["Overcast", "Mild", "High", "True", "Yes"],
            ["Overcast", "Hot", "Normal", "False", "Yes"],
            ["Sunny", "Mild", "High", "True", "No"],
        ],
        columns=["Outlook", "Temperature", "Humidity", "Windy", "Play Golf"],
    )
    return df


def autoencoder_builder(inp_lay, enc_lays, dec_lays, optimizer="adam"):
    from tensorflow import keras

    # AE.
    autoencoder = keras.Sequential([inp_lay] + enc_lays + dec_lays)
    autoencoder.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

    # Encoder.
    encoder = keras.Sequential([inp_lay] + enc_lays)

    # Decoder.
    decoder = keras.Sequential(
        [keras.Input(shape=enc_lays[-1].output.shape[1:])] + dec_lays
    )

    return [autoencoder, encoder, decoder]


def show_function(function):
    source = inspect.getsource(function)
    display(Markdown("```python\n" + source + "```"))


def plot_encoded_images(original, decoded, y, original_size=(28, 28)):
    labels = np.unique(y)
    for i, label in enumerate(labels):
        plt.subplot(2, len(labels), i + 1)
        plt.imshow(original[y == label][0].reshape(original_size))
        plt.axis("off")

        plt.subplot(2, len(labels), i + 1 + len(labels))
        plt.imshow(decoded[y == label][0].reshape(original_size))
        plt.axis("off")


def plot_mnist_embedding(encoded, y):
    labels = np.unique(y)
    for label in labels:
        plt.scatter(
            encoded[y == label, 0],
            encoded[y == label, 1],
            label="Digit {}".format(label),
        )

    plt.legend()
    plt.show()


def generate_dataset_china(plot=True):
    china = load_sample_images().images[0]
    china = china[: china.shape[0], : china.shape[0], :] / 255.0

    if plot:
        plt.imshow(china)
        plt.title("Original")
        plt.axis("off")

    return china


def plot_filter(image, filter, name):
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    inp = tf.constant([image])

    def kernel_init(shape, dtype=None):
        kernel = np.zeros(shape)
        kernel[:, :, 0, 0] = filter
        kernel[:, :, 1, 1] = filter
        kernel[:, :, 2, 2] = filter
        return kernel

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(image.shape),
            tf.keras.layers.Conv2D(3, filter.shape, kernel_initializer=kernel_init),
        ]
    )
    model.build()
    out = model.predict(inp, verbose=0)[0]
    out = np.clip(out, 0, 1)

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(filter)
    plt.title("Kernel (%s)" % name)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(out)
    plt.title("Convoluted Image")
    plt.axis("off")


def plot_training_evolution(history, metric="accuracy", validation=True):
    n_epochs = len(history.history["loss"])
    epochs = range(1, n_epochs + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history[metric])
    if validation:
        plt.plot(epochs, history.history["val_" + metric])
    plt.title(metric.capitalize())
    plt.ylabel(metric.capitalize())
    plt.xlabel("Epoch")
    if validation:
        plt.legend(["Train", "Validation"])

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"])
    if validation:
        plt.plot(epochs, history.history["val_loss"])
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    if validation:
        plt.legend(["Train", "Validation"])


def plot_confusion_matrix(y_r, y_p):
    cm = confusion_matrix(y_r, y_p)
    cm = cm / cm.sum()

    sn.heatmap(cm, annot=True, fmt=".2%", annot_kws={"fontsize": 6})
    plt.title("Confusion Matrix")
    plt.axis("equal")
    plt.axis("off")


def generate_dataset_temporal_1(n_points=513, plot=False):
    x = np.linspace(-8 * np.pi, 8 * np.pi, n_points)
    x = np.sin(x)

    y = x[1:].reshape(-1, 1)
    x = x[:-1].reshape(-1, 1, 1)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, shuffle=False)

    if plot:
        plt.plot(range(len(y_tr.ravel())), y_tr.ravel())
        plt.plot(
            range(len(y_tr.ravel()), len(y_tr.ravel()) + len(y_te.ravel())),
            y_te.ravel(),
        )
        plt.show()

    return x_tr, x_te, y_tr, y_te


def plot_input_output(x, y_r, y_p):
    plt.subplot(2, 1, 1)
    plt.plot(y_r.ravel(), label="Real")
    plt.plot(y_p.ravel(), label="Pred")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(x, y_r)
    plt.xlabel("Input")
    plt.ylabel("Real")
    plt.axis("equal")

    plt.subplot(2, 2, 4)
    plt.scatter(x, y_p)
    plt.xlabel("Input")
    plt.ylabel("Pred")
    plt.axis("equal")


def build_discriminator():
    from tensorflow import keras

    discriminator = keras.Sequential()

    discriminator.add(keras.Input(shape=(28, 28, 1)))
    discriminator.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"))
    discriminator.add(keras.layers.LeakyReLU(negative_slope=0.2))
    discriminator.add(
        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same")
    )
    discriminator.add(keras.layers.LeakyReLU(negative_slope=0.2))
    discriminator.add(
        keras.layers.Conv2D(128, kernel_size=4, strides=1, padding="same")
    )
    discriminator.add(keras.layers.LeakyReLU(negative_slope=0.2))
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dropout(0.2))
    discriminator.add(keras.layers.Dense(1, activation="sigmoid"))

    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

    return discriminator


def build_generator():
    from tensorflow import keras

    generator = keras.Sequential()

    generator.add(keras.Input(shape=(100,)))
    generator.add(keras.layers.Dense(7 * 7 * 128))
    generator.add(keras.layers.Reshape((7, 7, 128)))
    generator.add(
        keras.layers.Conv2DTranspose(128, kernel_size=4, strides=1, padding="same")
    )
    generator.add(keras.layers.LeakyReLU(negative_slope=0.2))
    generator.add(
        keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")
    )
    generator.add(keras.layers.LeakyReLU(negative_slope=0.2))
    generator.add(
        keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same")
    )
    generator.add(keras.layers.LeakyReLU(negative_slope=0.2))
    generator.add(
        keras.layers.Conv2D(1, kernel_size=5, padding="same", activation="sigmoid")
    )

    return generator


def build_gan(discriminator, generator):
    from tensorflow import keras

    gan_input = keras.Input(shape=(100,))
    gan = keras.Model(inputs=gan_input, outputs=discriminator(generator(gan_input)))
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    return gan


def plot_generated_images(generator, dim=(5, 5)):
    examples = np.prod(dim)
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise, verbose=0)
    generated_images = generated_images.reshape(examples, 28, 28)
    for i in range(generated_images.shape[0]):
        image = generated_images[i]

        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()


def generate_environment_cart():
    import gymnasium

    env = gymnasium.make("CartPole-v1", render_mode="human")

    return env


def train_rl_model(
    model, env, gamma=0.99, max_steps=10000, learning_rate=0.01, goal=80
):
    """Train a RL model.

    Args:
        model (keras.Model): Model implementing the actor and the critic.
        env (gymnasium.Env): Environment of the task.
        gamma (float, optional): Forgetting or discount factor for past rewards. Defaults to 0.99.
        max_steps (int, optional): Maximum number of steps. Defaults to 10000.
        learning_rate (float, optional): Learning rate. Defaults to 0.01.
        goal (float, optional): Objective reward. Defaults to 80.
    """
    import tensorflow as tf
    from tensorflow import keras

    num_actions = env.action_space.n

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0

    while True:
        state = env.reset()[0]
        episode_reward = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps):
                # Show the attemps.
                env.render()

                # Estimate the policy (prediction of the next actions) and the future rewards using the model.
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])

                # Choose random action using the policy.
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # Apply the sampled action.
                res = env.step(action)
                state = res[0]
                reward = res[1]
                done = res[2]
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break

            # Once the episode is finished, update running reward to check condition for solving.
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate the real expected value from rewards.
            returns = []
            discounted_sum = 0
            for r in rewards_history[:-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (
                np.std(returns) + np.finfo(np.float32).eps.item()
            )
            returns = returns.tolist()

            # Compute the loss values (both for Actor and Critic) to update the network.
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Update the weights through backpropagation.
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear variables.
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        episode_count += 1
        if episode_count % 10 == 0:
            print(
                "Running reward: {:6.2f} at episode {:3d}".format(
                    running_reward, episode_count
                )
            )

        if running_reward > goal:
            print("Solved at episode {}!".format(episode_count))
            break


def generate_dataset_lr_tf(n_pat=101):
    import tensorflow as tf

    x = tf.cast(tf.linspace(-1, 1, n_pat), dtype=tf.float32)
    y = x * 2 + 5

    return x, y


def evaluate_model_clas(model, x_tr, y_tr, x_te, y_te, print_table=True):
    y_tr_p = model.predict(x_tr)
    y_te_p = model.predict(x_te)

    er_tr = [accuracy_score(y_tr, y_tr_p), balanced_accuracy_score(y_tr, y_tr_p)]
    er_te = [accuracy_score(y_te, y_te_p), balanced_accuracy_score(y_te, y_te_p)]

    ers = [er_tr, er_te]
    headers = ["Acc", "Bal. Acc"]

    if print_table:
        print("{:>15}".format(""), end="")
        for h in headers:
            print("{:>15}".format(h), end="")
        print("")

        headers_col = ["Train", "Test"]

        cnt = 0
        for er in ers:
            hc = headers_col[cnt]
            cnt = cnt + 1
            print("{:<15}".format(hc), end="")

            for e in er:
                print("{:15.2f}".format(e), end="")
            print("")

    return ers


def save_predictions(y_ch_p, team_number, n_preds=500):
    if len(y_ch_p) != n_preds:
        print(
            "Error saving the predictions, it should be a vector of %d lables" % n_preds
        )
    else:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        np.savetxt("Team_{:02d}_{}.txt".format(team_number, time_str), y_ch_p, fmt="%d")


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(
        self,
        n_clusters=3,
        max_iter=50,
        tol=1e-3,
        random_state=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        verbose=0,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_, update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_, update_within=False)
        return dist.argmin(axis=1)
