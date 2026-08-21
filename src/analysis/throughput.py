"""Significância para VELOCIDADE: amostras independentes, nunca pares.

O [[B-045]] fechou a lacuna para **qualidade** — a paridade lexical do `b047` tem p=0,477 sobre
6.980
consultas. Ele não fecha para velocidade, e a razão é estrutural: o teste pareado precisa de valor
**por consulta**, e QPS não tem — é uma taxa agregada sobre a corrida inteira.

As duas maiores diferenças que o projeto publica são justamente de velocidade e seguiam sem teste:
Elasticsearch a **4,3x** o nosso QPS no lexical, e pgvector a **+16,3%** a recall casado no
vetorial.
A assimetria era indefensável — rigor onde a diferença é minúscula, dispensa onde ela é de 4x.

O ERRO QUE ESTE MÓDULO EXISTE PARA NÃO COMETER, escrito no DoD do [[B-049]]: *"nunca aplicar o
pareado a taxas agregadas"*. Duas corridas de QPS não são pares. Tratá-las como pares inventa uma
correlação que não existe e estreita o intervalo de confiança sem razão — o que produz
"significativo"
onde não há nada.

POR QUE WELCH E NÃO O t DE STUDENT: as duas configurações podem ter variâncias diferentes, e é comum
que tenham — o sistema mais rápido costuma ser também o mais estável. O t de Student assume
variâncias iguais e fica anticonservador quando elas diferem com N desigual (Welch 1947; Delacre,
Lakens & Leys 2017 recomendam Welch como default). O bootstrap sobre a **razão** vem ao lado porque
é a razão que a gente publica ("4,3x"), e um IC sobre a diferença não responde a pergunta que a
frase faz.

NumPy puro, pela mesma razão do módulo pareado: reamostragem é um laço, e um laço não justifica uma
dependência.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from theodb_bench.errors import ConfigError, ErrorContext, Phase

DEFAULT_ALPHA: float = 0.05
DEFAULT_RESAMPLES: int = 10_000
DEFAULT_SEED: int = 20260821

#: Mínimo de corridas para que exista variância. Com uma, o desvio é indefinido e um IC teria
#: largura
#: zero — que se lê como certeza absoluta e é o oposto do que uma amostra de tamanho 1 autoriza.
MIN_RUNS: int = 2


@dataclass(frozen=True)
class RunSummary:
    """O que N corridas de uma configuração dizem, com a dispersão ao lado do centro."""

    n: int
    mean: float
    stdev: float
    ci_low: float
    ci_high: float

    @property
    def coefficient_of_variation(self) -> float:
        """Dispersão relativa. É o número que decide se o resultado merece um `p` ou um aviso.

        Uma diferença de 4x entre médias com CV de 40% em cada lado não é uma diferença medida — é
        duas nuvens que se tocam, e publicar o `p` dela seria esconder isso atrás de um decimal.
        """
        return self.stdev / self.mean if self.mean else float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "stdev": self.stdev,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "coefficient_of_variation": self.coefficient_of_variation,
        }


@dataclass(frozen=True)
class ThroughputComparison:
    """O veredito sobre duas configurações medidas em corridas separadas."""

    a: RunSummary
    b: RunSummary
    ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float
    method: str
    resamples: int
    seed: int

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def n_a(self) -> int:
        return self.a.n

    @property
    def n_b(self) -> int:
        return self.b.n

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": self.a.as_dict(),
            "b": self.b.as_dict(),
            "ratio": self.ratio,
            "ratio_ci_low": self.ci_low,
            "ratio_ci_high": self.ci_high,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
            "method": self.method,
            "resamples": self.resamples,
            "seed": self.seed,
            # EXPLÍCITO no artefato: quem lê precisa saber que isto NÃO é o teste pareado, porque
            # o projeto publica os dois e confundi-los é o erro que este módulo existe para evitar.
            "paired": False,
        }


def summarise_runs(values: Sequence[float], *, alpha: float = DEFAULT_ALPHA) -> RunSummary:
    """Média, desvio e IC de N corridas da MESMA configuração."""
    if len(values) < MIN_RUNS:
        raise ConfigError(
            f"são precisas pelo menos {MIN_RUNS} corridas para haver variância, recebi "
            f"{len(values)}. Uma corrida não tem dispersão, e um intervalo de largura zero se lê "
            "como certeza absoluta — que é o oposto do que uma amostra de tamanho 1 autoriza.",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    media = float(arr.mean())
    desvio = float(arr.std(ddof=1))
    meia_largura = _t_critical(n - 1, alpha) * desvio / math.sqrt(n)
    return RunSummary(
        n=n,
        mean=media,
        stdev=desvio,
        ci_low=media - meia_largura,
        ci_high=media + meia_largura,
    )


def compare_throughput(
    runs_a: Sequence[float],
    runs_b: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> ThroughputComparison:
    """Compara duas configurações medidas em corridas SEPARADAS.

    Os comprimentos podem diferir, e isso é a diferença estrutural com o pareado: lá o elemento *i*
    de cada lado é a mesma consulta, e exigir simetria é o que garante o par. Aqui não há par nenhum
    — exigir mesmo N sugeriria uma correspondência que não existe.
    """
    a = summarise_runs(runs_a, alpha=alpha)
    b = summarise_runs(runs_b, alpha=alpha)

    p = _welch_p_value(np.asarray(runs_a, float), np.asarray(runs_b, float))
    razao, baixo, alto = _bootstrap_ratio_ci(
        np.asarray(runs_a, float), np.asarray(runs_b, float), alpha, resamples, seed
    )
    return ThroughputComparison(
        a=a,
        b=b,
        ratio=razao,
        ci_low=baixo,
        ci_high=alto,
        p_value=p,
        alpha=alpha,
        method="Welch t-test (unequal variances) + bootstrap CI on the ratio",
        resamples=resamples,
        seed=seed,
    )


def precision_for_n(*, cv: float, n: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Meia-largura relativa do IC para um coeficiente de variação e N corridas.

    Responde à pergunta que o bullet 3 do [[B-049]] faz — *"qual N compra qual precisão"* — em
    vez de
    escolher N por hábito. Cada corrida do caso FTS levou ~7 min no droplet depois do dataset em
    cache, então N=5 são ~35 min por motor: saber que isso estreita o intervalo para ±X% é o que
    torna a conta uma decisão.

    Fórmula: `t(n-1, alpha/2) * cv / sqrt(n)`. É a meia-largura do IC dividida pela média, então o
    resultado é uma fração — 0,06 significa ±6%.
    """
    if n < MIN_RUNS:
        raise ConfigError(
            f"são precisas pelo menos {MIN_RUNS} corridas para haver um valor t, recebi {n}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    if cv < 0:
        raise ConfigError(
            f"o coeficiente de variação não pode ser negativo, recebi {cv}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    return _t_critical(n - 1, alpha) * cv / math.sqrt(n)


def _welch_p_value(a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]) -> float:
    """p bicaudal do t de Welch, sem SciPy.

    A CDF do t vem de uma beta incompleta regularizada, que a `math.lgamma` do stdlib sustenta por
    fração continuada. Trazer SciPy para duas funções seria a dependência que o módulo pareado
    também recusou.
    """
    n_a, n_b = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / n_a + vb / n_b
    if se2 == 0:
        # Variância zero dos dois lados: ou são idênticos, ou a medição não variou nada — e nos dois
        # casos um p-valor não é a resposta. `1.0` recusa a alegação de diferença.
        return 1.0 if a.mean() == b.mean() else 0.0
    t = (a.mean() - b.mean()) / math.sqrt(se2)
    gl = se2**2 / ((va / n_a) ** 2 / (n_a - 1) + (vb / n_b) ** 2 / (n_b - 1))
    return 2.0 * (1.0 - _student_t_cdf(abs(t), gl))


def _bootstrap_ratio_ci(
    a: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    alpha: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """IC percentil da RAZÃO b/a, reamostrando cada lado independentemente.

    A razão é o que a gente publica — "4,3x o nosso QPS" —, e um IC sobre a DIFERENÇA não responde a
    pergunta que essa frase faz. Reamostrar os dois lados de forma independente é o que preserva a
    ausência de pareamento: embaralhar juntos assumiria uma correspondência inexistente.
    """
    rng = np.random.default_rng(seed)
    razoes = np.empty(resamples, dtype=float)
    for i in range(resamples):
        ma = rng.choice(a, size=len(a), replace=True).mean()
        mb = rng.choice(b, size=len(b), replace=True).mean()
        razoes[i] = mb / ma if ma else float("inf")
    return (
        float(b.mean() / a.mean()) if a.mean() else float("inf"),
        float(np.percentile(razoes, 100 * alpha / 2)),
        float(np.percentile(razoes, 100 * (1 - alpha / 2))),
    )


def _student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    return 1.0 - 0.5 * _incomplete_beta(df / 2.0, 0.5, x)


def _t_critical(df: int, alpha: float) -> float:
    """t crítico bicaudal, por bisseção sobre a CDF. Preciso o bastante e sem dependência."""
    alvo = 1.0 - alpha / 2.0
    baixo, alto = 0.0, 1000.0
    for _ in range(200):
        meio = (baixo + alto) / 2.0
        if _student_t_cdf(meio, float(df)) < alvo:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2.0


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Beta incompleta regularizada por fração continuada (Lentz), como em Numerical Recipes."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    frente = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return frente * _beta_cf(a, b, x) / a
    return 1.0 - frente * _beta_cf(b, a, 1 - x) / b


def _beta_cf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h
