---
title: "ベイズ推論によるA/Bテストの効果検証入門"
date: 2020-12-25T13:11:25+09:00
draft: false
---

A/Bテストの効果検証では信頼区間を推定したり仮説検定を行ったりすることが多いと思います。しかしp値も信頼区間も直感的な解釈が難しく統計に詳しくない人からするととっつきづらかったります。  
ベイズ統計のアプローチを使った場合, 未知のパラメータを確率変数と考えてその確率分布を観測データをもとに推定します。推定された確率分布を用いることでパラメータのばらつきを可視化したり, BよりAの方がCVRが高い確率といったようなものが求められます。
これらの方法論を比較する為にいくつかA/Bテストをシミュレーションしてみました。
<!--more-->

この記事は [カンム Advent Calendar 2020](https://adventar.org/calendars/5396) $25$日目の記事です。

全てのコードは[github](https://github.com/yosukeyoshida/bayesianAB/blob/main/ab_testing.ipynb)にあります。

# ベイズ推論
ベイズ統計で重要なベイズの定理とそれを用いてどのように未知のパラメータを推論するか簡単に説明します。

## ベイズの定理
観測されたデータを表す確率変数を $D$ とし未知のパラメータを表す確率変数を $\theta$ とするとベイズの定理は以下の式で表されます。

$$
P(\theta \mid D) = \frac{P(D \mid \theta)P(\theta)}{P(D)}
$$

$P(\theta \mid D)$ は事後分布と呼ばれデータ $D$ を観測した後のパラメータ $\theta$ の確率分布となります。
$P(D\mid\theta)$ は尤度関数と呼ばれパラメータ $\theta$ を固定したときに観測されたデータが生じる確率を表します。
$P(\theta)$ は事前分布と呼ばれデータ $D$ を観測する前のパラメータ $\theta$ の確率分布を表します。
最後に $P(D)$ は周辺尤度と呼ばれ事後分布を積分した値が $1$ となるための正規化定数となります。  
ベイズの定理を用いることで観測されたデータによって事前分布を事後分布に更新することができます。これをベイズ更新と呼びます。


## ベイズ更新の例
コインを投げたときの表が出る確率分布について考えます。二値確率変数 $x \in \\{0, 1\\}$ において表を $x=1$, 裏を $x=0$ で表しコインの表が出る確率を $\theta$ とするとその確率分布は以下のようにベルヌーイ分布として表せます。
$$
Bern(x \mid \theta) = \theta^x(1 - \theta)^{1-x}
$$

事前分布を $[0, 1]$ の一様分布とし, 尤度関数をベルヌーイ分布として複数回コインを投げた後の事後分布をベイズ更新によって求めます。
ここでは簡単にする為に $\theta$ を $[0, 1]$ の区間で離散化し積分をせずに事後確率を近似しています。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import stats

# パラメータ
thetas = np.linspace(0, 1, 1000)

# 尤度関数
likelihood = lambda x: thetas if x == 1 else (1-thetas)

# 事後分布
def posterior(r, prior):
    lp = likelihood(r) * prior
    return lp / lp.sum()

def plot(thetas, p, ax, title="", ylim=0.0022):
    ax.plot(thetas, p)
    ax.fill_between(thetas, 0, p, color="#348ABD", alpha=0.4)
    ax.set_ylim([0, ylim])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'$\theta$');

# 事前分布
p = np.array([1 / len(thetas) for _ in thetas])

fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax = ax.ravel()
plot(thetas, p, ax[0], "事前分布")

# コインが裏, 表, 表と出た
trials = [0, 1, 1]
for i, t in enumerate(trials):
    p = posterior(t, p)
    result = '表' if t == 1 else '裏'
    plot(thetas, p, ax[i+1], f"{i+1}回目 {result}")
fig.tight_layout()
fig.savefig("000.png", bbox_inches="tight")

# 真の表が出る確率を0.5として追加で97回投げる
trials = stats.bernoulli.rvs(0.5, size=97, random_state=42)
for i, t in enumerate(trials):
    p = posterior(t, p)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plot(thetas, p, ax, "100回目", 0.01)
```

![000](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/000.png)
左上はコインを投げる前の事前分布, 右上はコインを1回投げて裏だったときの事後分布となります。裏が1回観測されることにより $\theta$ が小さい確率が高くなっていることが分かります。
2回目以降は前回の事後分布が事前分布となり逐次ベイズ更新によって事後分布が更新されていきます。2回目, 3回目は続けて表が出たときの事後分布を示しています。

さらに$97$回コイン投げを行い合計100回ベイズ更新することで以下の事後分布が得られました。だいぶばらつきが小さくなりパラメータ $\theta$ がだいたいどの辺に存在する確率が高いか分かってきました。
![001](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/001.png)
このようにベイズ推論では観測されたデータをもとに未知のパラメータの確率分布を求めます。

# A/Bテストのシミュレーション
ECサイトにおいて商品を購入してもらう為に効果的な広告をA/Bテストで検証したいとします。
広告A, Bの真のCVRをそれぞれ $0.05$, $0.04$ とします。当然これらのCVRは実際には知ることはできずABテストから推測したい真の値となります。  

まずA/Bテストを実施したと想定してユーザ数分のサンプルデータを生成します。
先ほどのコイン投げの例と同様に商品購入を1, 非購入を0と表すとパラメータを真のCVRとしたベルヌーイ分布を用いてテスト用のサンプルデータを生成することができます。
どちらの広告の対象ユーザ数も$1000$人とした場合以下のようになります。
```python
import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import pymc3 as pm
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

# 真のCVR
true_p_A = 0.05
true_p_B = 0.04

# ユーザ数
N_A = 1000
N_B = 1000

# サンプルデータ
sample_A = stats.bernoulli.rvs(true_p_A, size=N_A, random_state=1225)
sample_B = stats.bernoulli.rvs(true_p_B, size=N_B, random_state=1226)
print(f"(A) 購入人数={sample_A.sum()} / 観測されたCVR={sample_A.sum() / len(sample_A)}")
print(f"(B) 購入人数={sample_B.sum()} / 観測されたCVR={sample_B.sum() / len(sample_B)}")
# (A) 購入人数=57 / 観測されたCVR=0.057
# (B) 購入人数=48 / 観測されたCVR=0.048
```
観測されたCVRはそれぞれ $0.057, 0.048$ となりどちらも真のCVRよりも高い結果となりました。これはたまたま観測された結果であり再度A/Bテストを行った場合は異なる結果となるでしょう。

## ベイズ推論
今回はPyMC3を使ってCVRをベイズ推論します。PyMC3はPythonでベイズ統計モデリングを扱うことのできるフレームワークのひとつです。  
事前分布として $[0, 1]$ の一様分布, 尤度関数としてベルヌーイ分布を指定します。
A, BのCVRを$P_A$, $P_B$とし事後分布を求めそれらの差をdeltaとして計算します。

```python
with pm.Model() as model:
    # 事前分布は一様分布
    p_A = pm.Uniform("$p_A$", lower=0, upper=1)
    p_B = pm.Uniform("$p_B$", lower=0, upper=1)
    # AとBのCVRの差
    delta = pm.Deterministic("delta", p_A - p_B)
    # 尤度関数はベルヌーイ分布
    obs_A = pm.Bernoulli("obs_A", p_A, observed=sample_A)
    obs_B = pm.Bernoulli("obs_B", p_B, observed=sample_B)
    # 事後分布をサンプリング
    trace = pm.sample(20000, random_seed=42)
# プロットのコードは省略
```

$P_A$, $P_B$の事後分布は以下のようになりました。グラフの下部に示された区間は$95$%HDI (highest density interval) と呼ばれ確率密度の高いものから順に確率が$0.95$を占めるまでの区間を取ったものです。  
観測されたデータから$P_A$は$[0.044, 0.072]$の区間に, $P_B$は$[0.036, 0.063]$の区間に真の値が存在する可能性が高いと解釈されます。
![002](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/002.png)

$P_A$と$P_B$の事後分布を重ねてプロットしてみると分布のばらつきが大きく重なりが大きいように思います。もう少しA/Bテストを継続した方が良さそうですがこれだけを見ても判断が難しいですね。
追ってサンプルサイズを変えて検証してみたいと思います。
![004](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/004.png)

またdeltaはBよりもAのCVRがどの程度高いか差の分布となっています。$95$%HDIは$[-0.011, 0.029]$となっており推定された区間に0を含んでいるので差があるとはいえなさそうです。
![003](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/003.png)

deltaの分布において0より大きい部分の面積はBよりもAの方がCVRが高い確率と解釈することができます。
充分に確率が高いといえる閾値は$0.95$に取られることが多く$0.814$ではそこまで確信を持ってAの方がCVRが高いとはいえなさそうですがこちらも追ってもう少し詳しくみてみます。
![005](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/005.png)

## 仮説検定
比較のためにRで母比率の差の検定をしてみたところp値は$0.3669$, 信頼区間は$[-0.01054537, 0.02854537]$となり有意ではない結果となりました。
deltaのHDIと信頼区間はほぼ同じ区間を推定しています。

```python
> prop.test(c(57, 48), c(1000, 1000), correct=F)

	2-sample test for equality of proportions without continuity
	correction

data:  c(57, 48) out of c(1000, 1000)
X-squared = 0.81417, df = 1, p-value = 0.3669
alternative hypothesis: two.sided
95 percent confidence interval:
 -0.01054537  0.02854537
sample estimates:
prop 1 prop 2
 0.057  0.048
```

## 異なるサンプルサイズでの検証
1000人ずつではサンプルが少なそうであったのでもう少しユーザ数を増やしてみます。サンプルが増えるにつれて事後分布のばらつきは小さくなり, より確信を持って区間推定ができている様子が伺えます。
![006](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/006.png)

横軸にユーザ数をとり, 縦軸に母比率の差の信頼区間とHDIをプロットしたものです。信頼区間もHDIもどちらもほぼ同じ区間を示しています。
$3000$人を超えたあたりで信頼区間, HDIのどちらも下限が0を上回りはじめます。$6500$人を超えると確信を持って差があると判断ができそうです。区間推定の平均も0.01とほぼ真のCVRの差に漸近しています。
![007](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/007.png)

deltaの0より大きい部分の面積, つまりBよりもAのCVRの方が高い確率です。同様に$3000$人を超えたあたりで$0.95$を上回り, $6500$人を超えると安定して高い確率を示します。
![008](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/b8b24c4330c49bab4ed5a03572ebae214f4939d4/008.png)

ちなみにRで有意水準$5$%, 検出力を$80$%としたときに母比率の差の検定に必要なサンプルサイズを計算すると1郡あたり$6744$人必要となりました。
上の結果からもそれだけのサンプルがあれば充分といえそうです。
```python
> power.prop.test(p1=0.05, p2=0.04, sig.level=0.05, power=0.8)

     Two-sample comparison of proportions power calculation

              n = 6744.933
             p1 = 0.05
             p2 = 0.04
      sig.level = 0.05
          power = 0.8
    alternative = two.sided

NOTE: n is number in *each* group
```

## A/Bテストの早期終了による問題
頻度論における仮説検定では事前にサンプルサイズを見積もり, そのサイズに達するまではA/Bテストを終了すべきではないとされています。
それは早期に終了してしまうと本当は差がないのに差があるという間違った判断(偽陽性)をしてしまうことがあるからです。  

AとBの真のCVRをどちらも$0.05$で差がないとした場合に同様のA/Bテストを行ってシミュレーションしてみました。
ユーザ数が$3000$人のあたりで有意であるか際どいラインになっています。BよりもAのCVRが高い確率も同様に$0.95$まで到達しています。  
このタイミングでA/Bテストを打ち切ってAの方が効果的であると判断した場合, 本当は差がないのに誤って判断することになります。
![009](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/5f8ad40fb1d203fedeb763f6944d1546f0b8ef74/009.png)
![010](https://gist.githubusercontent.com/yosukeyoshida/d9eab0f145b18066a4a41af492573a04/raw/5f8ad40fb1d203fedeb763f6944d1546f0b8ef74/010.png)

慎重に判断すべき問題であればやはり事前にサンプルサイズを見積もって仮説検定にもっていくのがよいのでしょう。
しかし偽陽性を最小限に抑えようとすると時間もかかり必要となるサンプルサイズも大きくなります。扱っている対象によっては精度よりも速度を優先して改善を回していきたいというケースもあります。

ベイズ推論では事後分布を具体的に求めることができるというメリットがあります。
事後分布を用いることで誤った判断を下したときの損失を計算することができます。
A, Bのどちらを採用するかを$x$とし, 損失関数を $L(p_A, p_B, x)$ と定義すると以下のように期待損失が計算できるのでこの期待損失が閾値を下回った場合にA/Bテストを停止するというものです。(参考[1])  

$$
E\[L\](x) = \int_A \int_B L(p_A, p_B, x)f(p_A, p_B)dp_Adp_B
$$

偽陽性はある程度許容するかわりに損失をコントロールすることで精度と速度のバランスを最適化することができる、のだと思います。
本当は損失関数を用いたシミュレーションもやりたかったのですが発展課題としてまた改めてやります。


# 参考
[1] [Bayesian A/B testing — a practical exploration with simulations](https://towardsdatascience.com/exploring-bayesian-a-b-testing-with-simulations-7500b4fc55bc)

