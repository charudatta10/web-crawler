## No Title

No Content
Read more
## Generative adversarial network













Toggle the table of contents
















Generative adversarial network








23 languages










العربية
閩南語 / Bân-lâm-gú
Català
Deutsch
Ελληνικά
Español
فارسی
Français
한국어
Italiano
עברית
Nederlands
日本語
Русский
Simple English
کوردی
Suomi
Svenska
Türkçe
Українська
Tiếng Việt
粵語
中文




Edit links
























Article
Talk












English




































Read
Edit
View history
















Tools












Tools


move to sidebar


hide







		Actions
	






Read
Edit
View history











		General
	






What links here
Related changes
Upload file
Special pages
Permanent link
Page information
Cite this page
Get shortened URL
Download QR code
Wikidata item











		Print/export
	






Download as PDF
Printable version











		In other projects
	






Wikimedia Commons












































Appearance


move to sidebar


hide






















From Wikipedia, the free encyclopedia






Deep learning method


Not to be confused with 
Adversarial machine learning
.








Part of a series on
Machine learning
and 
data mining


Paradigms


Supervised learning


Unsupervised learning


Semi-supervised learning


Self-supervised learning


Reinforcement learning


Meta-learning


Online learning


Batch learning


Curriculum learning


Rule-based learning


Neuro-symbolic AI


Neuromorphic engineering


Quantum machine learning




Problems


Classification


Generative modeling


Regression


Clustering


Dimensionality reduction


Density estimation


Anomaly detection


Data cleaning


AutoML


Association rules


Semantic analysis


Structured prediction


Feature engineering


Feature learning


Learning to rank


Grammar induction


Ontology learning


Multimodal learning




Supervised learning
(
classification
 • 
regression
)
 


Apprenticeship learning


Decision trees


Ensembles


Bagging


Boosting


Random forest


k
-NN


Linear regression


Naive Bayes


Artificial neural networks


Logistic regression


Perceptron


Relevance vector machine (RVM)


Support vector machine (SVM)




Clustering


BIRCH


CURE


Hierarchical


k
-means


Fuzzy


Expectation–maximization (EM)


DBSCAN


OPTICS


Mean shift




Dimensionality reduction


Factor analysis


CCA


ICA


LDA


NMF


PCA


PGD


t-SNE


SDL




Structured prediction


Graphical models


Bayes net


Conditional random field


Hidden Markov




Anomaly detection


RANSAC


k
-NN


Local outlier factor


Isolation forest




Artificial neural network


Autoencoder


Deep learning


Feedforward neural network


Recurrent neural network


LSTM


GRU


ESN


reservoir computing


Boltzmann machine


Restricted


GAN


Diffusion model


SOM


Convolutional neural network


U-Net


LeNet


AlexNet


DeepDream


Neural radiance field


Transformer


Vision


Mamba


Spiking neural network


Memtransistor


Electrochemical RAM
 (ECRAM)




Reinforcement learning


Q-learning


SARSA


Temporal difference (TD)


Multi-agent


Self-play




Learning with humans


Active learning


Crowdsourcing


Human-in-the-loop


RLHF




Model diagnostics


Coefficient of determination


Confusion matrix


Learning curve


ROC curve




Mathematical foundations


Kernel machines


Bias–variance tradeoff


Computational learning theory


Empirical risk minimization


Occam learning


PAC learning


Statistical learning


VC theory




Journals and conferences


ECML PKDD


NeurIPS


ICML


ICLR


IJCAI


ML


JMLR




Related articles


Glossary of artificial intelligence


List of datasets for machine-learning research


List of datasets in computer vision and image processing


Outline of machine learning


v
t
e


An illustration of how a GAN works


A 
generative adversarial network
 (
GAN
) is a class of 
machine learning
 frameworks and a prominent framework for approaching 
generative AI
.
[
1
]
[
2
]
 The concept was initially developed by 
Ian Goodfellow
 and his colleagues in June 2014.
[
3
]
 In a GAN, two 
neural networks
 contest with each other in the form of a 
zero-sum game
, where one agent's gain is another agent's loss.

Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics. Though originally proposed as a form of 
generative model
 for 
unsupervised learning
, GANs have also proved useful for 
semi-supervised learning
,
[
4
]
 fully 
supervised learning
,
[
5
]
 and 
reinforcement learning
.
[
6
]


The core idea of a GAN is based on the "indirect" training through the discriminator, another neural network that can tell how "realistic" the input seems, which itself is also being updated dynamically.
[
7
]
 This means that the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator. This enables the model to learn in an unsupervised manner.

GANs are similar to 
mimicry
 in 
evolutionary biology
, with an 
evolutionary arms race
 between both networks.





Definition
[
edit
]


Mathematical
[
edit
]

The original GAN is defined as the following 
game
:
[
3
]


Each 
probability space
 








(


Ω


,




μ




ref






)






{\displaystyle (\Omega ,\mu _{\text{ref}})}




 defines a GAN game.

There are 2 players: generator and discriminator.

The generator's 
strategy set
 is 












P






(


Ω


)






{\displaystyle {\mathcal {P}}(\Omega )}




, the set of all probability measures 










μ




G










{\displaystyle \mu _{G}}




 on 








Ω






{\displaystyle \Omega }




.

The discriminator's strategy set is the set of 
Markov kernels
 










μ




D






:


Ω


→






P






[


0


,


1


]






{\displaystyle \mu _{D}:\Omega \to {\mathcal {P}}[0,1]}




, where 












P






[


0


,


1


]






{\displaystyle {\mathcal {P}}[0,1]}




 is the set of probability measures on 








[


0


,


1


]






{\displaystyle [0,1]}




.

The GAN game is a 
zero-sum game
, with objective function








L


(




μ




G






,




μ




D






)


:=




E




x


∼




μ




ref






,


y


∼




μ




D






(


x


)






⁡


[


ln


⁡


y


]


+




E




x


∼




μ




G






,


y


∼




μ




D






(


x


)






⁡


[


ln


⁡


(


1


−


y


)


]


.






{\displaystyle L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]+\operatorname {E} _{x\sim \mu _{G},y\sim \mu _{D}(x)}[\ln(1-y)].}





The generator aims to minimize the objective, and the discriminator aims to maximize the objective.



The generator's task is to approach 










μ




G






≈




μ




ref










{\displaystyle \mu _{G}\approx \mu _{\text{ref}}}




, that is, to match its own output distribution as closely as possible to the reference distribution. The discriminator's task is to output a value close to 1 when the input appears to be from the reference distribution, and to output a value close to 0 when the input looks like it came from the generator distribution.

In practice
[
edit
]


The 
generative
 network
 generates candidates while the 
discriminative
 network
 evaluates them.
[
3
]
 The contest operates in terms of data distributions. Typically, the generative network learns to map from a 
latent space
 to a data distribution of interest, while the discriminative network distinguishes candidates produced by the generator from the true data distribution. The generative network's training objective is to increase the error rate of the discriminative network (i.e., "fool" the discriminator network by producing novel candidates that the discriminator thinks are not synthesized (are part of the true data distribution)).
[
3
]
[
8
]


A known dataset serves as the initial training data for the discriminator. Training involves presenting it with samples from the training dataset until it achieves acceptable accuracy. The generator is trained based on whether it succeeds in fooling the discriminator. Typically, the generator is seeded with randomized input that is sampled from a predefined 
latent space
 (e.g. a 
multivariate normal distribution
). Thereafter, candidates synthesized by the generator are evaluated by the discriminator. Independent 
backpropagation
 procedures are applied to both networks so that the generator produces better samples, while the discriminator becomes more skilled at flagging synthetic samples.
[
9
]
 When used for image generation, the generator is typically a 
deconvolutional neural network
, and the discriminator is a 
convolutional neural network
.



Relation to other statistical machine learning methods
[
edit
]


GANs are 
implicit generative models
,
[
10
]
 which means that they do not explicitly model the likelihood function nor provide a means for finding the latent variable corresponding to a given sample, unlike alternatives such as 
flow-based generative model
.



Main types of deep generative models that perform maximum likelihood estimation
[
11
]


Compared to fully visible belief networks such as 
WaveNet
 and PixelRNN and autoregressive models in general, GANs can generate one complete sample in one pass, rather than multiple passes through the network.

Compared to 
Boltzmann machines
 and linear 
ICA
, there is no restriction on the type of function used by the network.

Since neural networks are 
universal approximators
, GANs are 
asymptotically consistent
. 
Variational autoencoders
 might be universal approximators, but it is not proven as of 2017.
[
11
]




Mathematical properties
[
edit
]


Measure-theoretic considerations
[
edit
]


This section provides some of the mathematical theory behind these methods.


In 
modern probability theory
 based on 
measure theory
, a probability space also needs to be equipped with a 
σ-algebra
. As a result, a more rigorous definition of the GAN game would make the following changes:
Each probability space 








(


Ω


,






B






,




μ




ref






)






{\displaystyle (\Omega ,{\mathcal {B}},\mu _{\text{ref}})}




 defines a GAN game.

The generator's strategy set is 












P






(


Ω


,






B






)






{\displaystyle {\mathcal {P}}(\Omega ,{\mathcal {B}})}




, the set of all probability measures 










μ




G










{\displaystyle \mu _{G}}




 on the measure-space 








(


Ω


,






B






)






{\displaystyle (\Omega ,{\mathcal {B}})}




.


The discriminator's strategy set is the set of 
Markov kernels
 










μ




D






:


(


Ω


,






B






)


→






P






(


[


0


,


1


]


,






B






(


[


0


,


1


]


)


)






{\displaystyle \mu _{D}:(\Omega ,{\mathcal {B}})\to {\mathcal {P}}([0,1],{\mathcal {B}}([0,1]))}




, where 












B






(


[


0


,


1


]


)






{\displaystyle {\mathcal {B}}([0,1])}




 is the 
Borel σ-algebra
 on 








[


0


,


1


]






{\displaystyle [0,1]}




.
Since issues of measurability never arise in practice, these will not concern us further.

Choice of the strategy set
[
edit
]


In the most generic version of the GAN game described above, the strategy set for the discriminator contains all Markov kernels 










μ




D






:


Ω


→






P






[


0


,


1


]






{\displaystyle \mu _{D}:\Omega \to {\mathcal {P}}[0,1]}




, and the strategy set for the generator contains arbitrary 
probability distributions
 










μ




G










{\displaystyle \mu _{G}}




 on 








Ω






{\displaystyle \Omega }




.

However, as shown below, the optimal discriminator strategy against any 










μ




G










{\displaystyle \mu _{G}}




 is deterministic, so there is no loss of generality in restricting the discriminator's strategies to deterministic functions 








D


:


Ω


→


[


0


,


1


]






{\displaystyle D:\Omega \to [0,1]}




. In most applications, 








D






{\displaystyle D}




 is a 
deep neural network
 function.

As for the generator, while 










μ




G










{\displaystyle \mu _{G}}




 could theoretically be any computable probability distribution, in practice, it is usually implemented as a 
pushforward
: 










μ




G






=




μ




Z






∘




G




−


1










{\displaystyle \mu _{G}=\mu _{Z}\circ G^{-1}}




. That is, start with a random variable 








z


∼




μ




Z










{\displaystyle z\sim \mu _{Z}}




, where 










μ




Z










{\displaystyle \mu _{Z}}




 is a probability distribution that is easy to compute (such as the 
uniform distribution
, or the 
Gaussian distribution
), then define a function 








G


:




Ω




Z






→


Ω






{\displaystyle G:\Omega _{Z}\to \Omega }




. Then the distribution 










μ




G










{\displaystyle \mu _{G}}




 is the distribution of 








G


(


z


)






{\displaystyle G(z)}




.

Consequently, the generator's strategy is usually defined as just 








G






{\displaystyle G}




, leaving 








z


∼




μ




Z










{\displaystyle z\sim \mu _{Z}}




 implicit. In this formalism, the GAN game objective is








L


(


G


,


D


)


:=




E




x


∼




μ




ref










⁡


[


ln


⁡


D


(


x


)


]


+




E




z


∼




μ




Z










⁡


[


ln


⁡


(


1


−


D


(


G


(


z


)


)


)


]


.






{\displaystyle L(G,D):=\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]+\operatorname {E} _{z\sim \mu _{Z}}[\ln(1-D(G(z)))].}








Generative reparametrization
[
edit
]


The GAN architecture has two main components. One is casting optimization into a game, of form 










min




G








max




D






L


(


G


,


D


)






{\displaystyle \min _{G}\max _{D}L(G,D)}




, which is different from the usual kind of optimization, of form 










min




θ






L


(


θ


)






{\displaystyle \min _{\theta }L(\theta )}




. The other is the decomposition of 










μ




G










{\displaystyle \mu _{G}}




 into 










μ




Z






∘




G




−


1










{\displaystyle \mu _{Z}\circ G^{-1}}




, which can be understood as a reparametrization trick.

To see its significance, one must compare GAN with previous methods for learning generative models, which were plagued with "intractable probabilistic computations that arise in maximum likelihood estimation and related strategies".
[
3
]


At the same time, Kingma and Welling
[
12
]
 and Rezende et al.
[
13
]
 developed the same idea of reparametrization into a general stochastic backpropagation method. Among its first applications was the 
variational autoencoder
.



Move order and strategic equilibria
[
edit
]


In the original paper, as well as most subsequent papers, it is usually assumed that the generator 
moves first
, and the discriminator 
moves second
, thus giving the following minimax game:










min






μ




G












max






μ




D










L


(




μ




G






,




μ




D






)


:=




E




x


∼




μ




ref






,


y


∼




μ




D






(


x


)






⁡


[


ln


⁡


y


]


+




E




x


∼




μ




G






,


y


∼




μ




D






(


x


)






⁡


[


ln


⁡


(


1


−


y


)


]


.






{\displaystyle \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]+\operatorname {E} _{x\sim \mu _{G},y\sim \mu _{D}(x)}[\ln(1-y)].}






If both the generator's and the discriminator's strategy sets are spanned by a finite number of strategies, then by the 
minimax theorem
,










min






μ




G












max






μ




D










L


(




μ




G






,




μ




D






)


=




max






μ




D












min






μ




G










L


(




μ




G






,




μ




D






)






{\displaystyle \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=\max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})}




that is, the move order does not matter.

However, since the strategy sets are both not finitely spanned, the minimax theorem does not apply, and the idea of an "equilibrium" becomes delicate. To wit, there are the following different concepts of equilibrium:



Equilibrium when generator moves first, and discriminator moves second:
















μ


^










G






∈


arg


⁡




min






μ




G












max






μ




D










L


(




μ




G






,




μ




D






)


,












μ


^










D






∈


arg


⁡




max






μ




D










L


(










μ


^










G






,




μ




D






)


,








{\displaystyle {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D}),\quad {\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}L({\hat {\mu }}_{G},\mu _{D}),\quad }






Equilibrium when discriminator moves first, and generator moves second:
















μ


^










D






∈


arg


⁡




max






μ




D












min






μ




G










L


(




μ




G






,




μ




D






)


,












μ


^










G






∈


arg


⁡




min






μ




G










L


(




μ




G






,










μ


^










D






)


,






{\displaystyle {\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D}),\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}L(\mu _{G},{\hat {\mu }}_{D}),}






Nash equilibrium
 








(










μ


^










D






,










μ


^










G






)






{\displaystyle ({\hat {\mu }}_{D},{\hat {\mu }}_{G})}




, which is stable under simultaneous move order:
















μ


^










D






∈


arg


⁡




max






μ




D










L


(










μ


^










G






,




μ




D






)


,












μ


^










G






∈


arg


⁡




min






μ




G










L


(




μ




G






,










μ


^










D






)






{\displaystyle {\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}L({\hat {\mu }}_{G},\mu _{D}),\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}L(\mu _{G},{\hat {\mu }}_{D})}






For general games, these equilibria do not have to agree, or even to exist. For the original GAN game, these equilibria all exist, and are all equal. However, for more general GAN games, these do not necessarily exist, or agree.
[
14
]




Main theorems for GAN game
[
edit
]

The original GAN paper proved the following two theorems:
[
3
]


Theorem
 
(the optimal discriminator computes the Jensen–Shannon divergence)
 — 
For any fixed generator strategy 










μ




G










{\displaystyle \mu _{G}}




, let the optimal reply be 










D




∗






=


arg


⁡




max




D






L


(




μ




G






,


D


)






{\displaystyle D^{*}=\arg \max _{D}L(\mu _{G},D)}




, then



















D




∗






(


x


)








=








d




μ




ref










d


(




μ




ref






+




μ




G






)
















L


(




μ




G






,




D




∗






)








=


2




D




J


S






(




μ




ref






;




μ




G






)


−


2


ln


⁡


2














{\displaystyle {\begin{aligned}D^{*}(x)&={\frac {d\mu _{\text{ref}}}{d(\mu _{\text{ref}}+\mu _{G})}}\\[6pt]L(\mu _{G},D^{*})&=2D_{JS}(\mu _{\text{ref}};\mu _{G})-2\ln 2\end{aligned}}}






where the derivative is the 
Radon–Nikodym derivative
, and 










D




J


S










{\displaystyle D_{JS}}




 is the 
Jensen–Shannon divergence
.



Proof


By Jensen's inequality,











E




x


∼




μ




ref






,


y


∼




μ




D






(


x


)






⁡


[


ln


⁡


y


]


≤




E




x


∼




μ




ref










⁡


[


ln


⁡




E




y


∼




μ




D






(


x


)






⁡


[


y


]


]






{\displaystyle \operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]\leq \operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln \operatorname {E} _{y\sim \mu _{D}(x)}[y]]}





and similarly for the other term. Therefore, the optimal reply can be deterministic, i.e. 










μ




D






(


x


)


=




δ




D


(


x


)










{\displaystyle \mu _{D}(x)=\delta _{D(x)}}




 for some function 








D


:


Ω


→


[


0


,


1


]






{\displaystyle D:\Omega \to [0,1]}




, in which case









L


(




μ




G






,




μ




D






)


:=




E




x


∼




μ




ref










⁡


[


ln


⁡


D


(


x


)


]


+




E




x


∼




μ




G










⁡


[


ln


⁡


(


1


−


D


(


x


)


)


]


.






{\displaystyle L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]+\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))].}






To define suitable density functions, we define a base measure 








μ


:=




μ




ref






+




μ




G










{\displaystyle \mu :=\mu _{\text{ref}}+\mu _{G}}




, which allows us to take the Radon–Nikodym derivatives











ρ




ref






=








d




μ




ref










d


μ












ρ




G






=








d




μ




G










d


μ












{\displaystyle \rho _{\text{ref}}={\frac {d\mu _{\text{ref}}}{d\mu }}\quad \rho _{G}={\frac {d\mu _{G}}{d\mu }}}





with 










ρ




ref






+




ρ




G






=


1






{\displaystyle \rho _{\text{ref}}+\rho _{G}=1}




.

We then have









L


(




μ




G






,




μ




D






)


:=


∫


μ


(


d


x


)




[






ρ




ref






(


x


)


ln


⁡


(


D


(


x


)


)


+




ρ




G






(


x


)


ln


⁡


(


1


−


D


(


x


)


)




]




.






{\displaystyle L(\mu _{G},\mu _{D}):=\int \mu (dx)\left[\rho _{\text{ref}}(x)\ln(D(x))+\rho _{G}(x)\ln(1-D(x))\right].}






The integrand is just the negative 
cross-entropy
 between two Bernoulli random variables with parameters 










ρ




ref






(


x


)






{\displaystyle \rho _{\text{ref}}(x)}




 and 








D


(


x


)






{\displaystyle D(x)}




. We can write this as 








−


H


(




ρ




ref






(


x


)


)


−




D




K


L






(




ρ




ref






(


x


)


∥


D


(


x


)


)






{\displaystyle -H(\rho _{\text{ref}}(x))-D_{KL}(\rho _{\text{ref}}(x)\parallel D(x))}




, where 








H






{\displaystyle H}




 is the 
binary entropy function
, so









L


(




μ




G






,




μ




D






)


=


−


∫


μ


(


d


x


)


(


H


(




ρ




ref






(


x


)


)


+




D




K


L






(




ρ




ref






(


x


)


∥


D


(


x


)


)


)


.






{\displaystyle L(\mu _{G},\mu _{D})=-\int \mu (dx)(H(\rho _{\text{ref}}(x))+D_{KL}(\rho _{\text{ref}}(x)\parallel D(x))).}






This means that the optimal strategy for the discriminator is 








D


(


x


)


=




ρ




ref






(


x


)






{\displaystyle D(x)=\rho _{\text{ref}}(x)}




, with   









L


(




μ




G






,




μ




D






∗






)


=


−


∫


μ


(


d


x


)


H


(




ρ




ref






(


x


)


)


=




D




J


S






(




μ




ref






∥




μ




G






)


−


2


ln


⁡


2






{\displaystyle L(\mu _{G},\mu _{D}^{*})=-\int \mu (dx)H(\rho _{\text{ref}}(x))=D_{JS}(\mu _{\text{ref}}\parallel \mu _{G})-2\ln 2}






after routine calculation.





Interpretation
: For any fixed generator strategy 










μ




G










{\displaystyle \mu _{G}}




, the optimal discriminator keeps track of the likelihood ratio between the reference distribution and the generator distribution:














D


(


x


)






1


−


D


(


x


)








=








d




μ




ref










d




μ




G












(


x


)


=










μ




ref






(


d


x


)








μ




G






(


d


x


)








;




D


(


x


)


=


σ


(


ln


⁡




μ




ref






(


d


x


)


−


ln


⁡




μ




G






(


d


x


)


)






{\displaystyle {\frac {D(x)}{1-D(x)}}={\frac {d\mu _{\text{ref}}}{d\mu _{G}}}(x)={\frac {\mu _{\text{ref}}(dx)}{\mu _{G}(dx)}};\quad D(x)=\sigma (\ln \mu _{\text{ref}}(dx)-\ln \mu _{G}(dx))}




where 








σ






{\displaystyle \sigma }




 is the 
logistic function
.
In particular, if the prior probability for an image 








x






{\displaystyle x}




 to come from the reference distribution is equal to 












1


2










{\displaystyle {\frac {1}{2}}}




, then 








D


(


x


)






{\displaystyle D(x)}




 is just the posterior probability that 








x






{\displaystyle x}




 came from the reference distribution:








D


(


x


)


=


Pr


(


x




 came from reference distribution




∣


x


)


.






{\displaystyle D(x)=\Pr(x{\text{ came from reference distribution}}\mid x).}










Theorem
 
(the unique equilibrium point)
 — 
For any GAN game, there exists a pair 








(










μ


^










D






,










μ


^










G






)






{\displaystyle ({\hat {\mu }}_{D},{\hat {\mu }}_{G})}




 that is both a sequential equilibrium and a Nash equilibrium:



















L


(










μ


^










G






,










μ


^










D






)


=




min






μ




G












max






μ




D










L


(




μ




G






,




μ




D






)


=








max






μ




D












min






μ




G










L


(




μ




G






,




μ




D






)


=


−


2


ln


⁡


2




















μ


^










D






∈


arg


⁡




max






μ




D












min






μ




G










L


(




μ




G






,




μ




D






)


,
















μ


^










G






∈


arg


⁡




min






μ




G












max






μ




D










L


(




μ




G






,




μ




D






)




















μ


^










D






∈


arg


⁡




max






μ




D










L


(










μ


^










G






,




μ




D






)


,
















μ


^










G






∈


arg


⁡




min






μ




G










L


(




μ




G






,










μ


^










D






)












∀


x


∈


Ω


,










μ


^










D






(


x


)


=




δ






1


2








,
















μ


^










G






=




μ




ref


















{\displaystyle {\begin{aligned}&L({\hat {\mu }}_{G},{\hat {\mu }}_{D})=\min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=&\max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=-2\ln 2\\[6pt]&{\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D}),&\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})\\[6pt]&{\hat {\mu }}_{D}\in \arg \max _{\mu _{D}}L({\hat {\mu }}_{G},\mu _{D}),&\quad {\hat {\mu }}_{G}\in \arg \min _{\mu _{G}}L(\mu _{G},{\hat {\mu }}_{D})\\[6pt]&\forall x\in \Omega ,{\hat {\mu }}_{D}(x)=\delta _{\frac {1}{2}},&\quad {\hat {\mu }}_{G}=\mu _{\text{ref}}\end{aligned}}}






That is, the generator perfectly mimics the reference, and the discriminator outputs 












1


2










{\displaystyle {\frac {1}{2}}}




 deterministically on all inputs.





Proof


From the previous proposition,









arg


⁡




min






μ




G












max






μ




D










L


(




μ




G






,




μ




D






)


=




μ




ref






;






min






μ




G












max






μ




D










L


(




μ




G






,




μ




D






)


=


−


2


ln


⁡


2.






{\displaystyle \arg \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=\mu _{\text{ref}};\quad \min _{\mu _{G}}\max _{\mu _{D}}L(\mu _{G},\mu _{D})=-2\ln 2.}






For any fixed discriminator strategy 










μ




D










{\displaystyle \mu _{D}}




, any 










μ




G










{\displaystyle \mu _{G}}




 concentrated on the set









{


x


∣




E




y


∼




μ




D






(


x


)






⁡


[


ln


⁡


(


1


−


y


)


]


=




inf




x








E




y


∼




μ




D






(


x


)






⁡


[


ln


⁡


(


1


−


y


)


]


}






{\displaystyle \{x\mid \operatorname {E} _{y\sim \mu _{D}(x)}[\ln(1-y)]=\inf _{x}\operatorname {E} _{y\sim \mu _{D}(x)}[\ln(1-y)]\}}





is an optimal strategy for the generator. Thus,









arg


⁡




max






μ




D












min






μ




G










L


(




μ




G






,




μ




D






)


=


arg


⁡




max






μ




D












E




x


∼




μ




ref






,


y


∼




μ




D






(


x


)






⁡


[


ln


⁡


y


]


+




inf




x








E




y


∼




μ




D






(


x


)






⁡


[


ln


⁡


(


1


−


y


)


]


.






{\displaystyle \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=\arg \max _{\mu _{D}}\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln y]+\inf _{x}\operatorname {E} _{y\sim \mu _{D}(x)}[\ln(1-y)].}






By Jensen's inequality, the discriminator can only improve by adopting the deterministic strategy of always playing 








D


(


x


)


=




E




y


∼




μ




D






(


x


)






⁡


[


y


]






{\displaystyle D(x)=\operatorname {E} _{y\sim \mu _{D}(x)}[y]}




. Therefore,









arg


⁡




max






μ




D












min






μ




G










L


(




μ




G






,




μ




D






)


=


arg


⁡




max




D








E




x


∼




μ




ref










⁡


[


ln


⁡


D


(


x


)


]


+




inf




x






ln


⁡


(


1


−


D


(


x


)


)






{\displaystyle \arg \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=\arg \max _{D}\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]+\inf _{x}\ln(1-D(x))}






By Jensen's inequality,



















ln


⁡




E




x


∼




μ




ref










⁡


[


D


(


x


)


]


+




inf




x






ln


⁡


(


1


−


D


(


x


)


)










=










ln


⁡




E




x


∼




μ




ref










⁡


[


D


(


x


)


]


+


ln


⁡


(


1


−




sup




x






D


(


x


)


)










=










ln


⁡


[




E




x


∼




μ




ref










⁡


[


D


(


x


)


]


(


1


−




sup




x






D


(


x


)


)


]


≤


ln


⁡


[




sup




x






D


(


x


)


)


(


1


−




sup




x






D


(


x


)


)


]


≤


ln


⁡






1


4






,














{\displaystyle {\begin{aligned}&\ln \operatorname {E} _{x\sim \mu _{\text{ref}}}[D(x)]+\inf _{x}\ln(1-D(x))\\[6pt]={}&\ln \operatorname {E} _{x\sim \mu _{\text{ref}}}[D(x)]+\ln(1-\sup _{x}D(x))\\[6pt]={}&\ln[\operatorname {E} _{x\sim \mu _{\text{ref}}}[D(x)](1-\sup _{x}D(x))]\leq \ln[\sup _{x}D(x))(1-\sup _{x}D(x))]\leq \ln {\frac {1}{4}},\end{aligned}}}






with equality if 








D


(


x


)


=






1


2










{\displaystyle D(x)={\frac {1}{2}}}




, so









∀


x


∈


Ω


,










μ


^










D






(


x


)


=




δ






1


2








;






max






μ




D












min






μ




G










L


(




μ




G






,




μ




D






)


=


−


2


ln


⁡


2.






{\displaystyle \forall x\in \Omega ,{\hat {\mu }}_{D}(x)=\delta _{\frac {1}{2}};\quad \max _{\mu _{D}}\min _{\mu _{G}}L(\mu _{G},\mu _{D})=-2\ln 2.}






Finally, to check that this is a Nash equilibrium, note that when 










μ




G






=




μ




ref










{\displaystyle \mu _{G}=\mu _{\text{ref}}}




, we have









L


(




μ




G






,




μ




D






)


:=




E




x


∼




μ




ref






,


y


∼




μ




D






(


x


)






⁡


[


ln


⁡


(


y


(


1


−


y


)


)


]






{\displaystyle L(\mu _{G},\mu _{D}):=\operatorname {E} _{x\sim \mu _{\text{ref}},y\sim \mu _{D}(x)}[\ln(y(1-y))]}





which is always maximized by 








y


=






1


2










{\displaystyle y={\frac {1}{2}}}




.

When 








∀


x


∈


Ω


,




μ




D






(


x


)


=




δ






1


2












{\displaystyle \forall x\in \Omega ,\mu _{D}(x)=\delta _{\frac {1}{2}}}




, any strategy is optimal for the generator.





Training and evaluating GAN
[
edit
]


Training
[
edit
]


Unstable convergence
[
edit
]


While the GAN game has a unique global equilibrium point when both the generator and discriminator have access to their entire strategy sets, the equilibrium is no longer guaranteed when they have a restricted strategy set.
[
14
]


In practice, the generator has access only to measures of form 










μ




Z






∘




G




θ






−


1










{\displaystyle \mu _{Z}\circ G_{\theta }^{-1}}




, where 










G




θ










{\displaystyle G_{\theta }}




 is a function computed by a neural network with parameters 








θ






{\displaystyle \theta }




, and 










μ




Z










{\displaystyle \mu _{Z}}




 is an easily sampled distribution, such as the uniform or normal distribution. Similarly, the discriminator has access only to functions of form 










D




ζ










{\displaystyle D_{\zeta }}




, a function computed by a neural network with parameters 








ζ






{\displaystyle \zeta }




. These restricted strategy sets take up a 
vanishingly small proportion
 of their entire strategy sets.
[
15
]


Further, even if an equilibrium still exists, it can only be found by searching in the high-dimensional space of all possible neural network functions. The standard strategy of using 
gradient descent
 to find the equilibrium often does not work for GAN, and often the game "collapses" into one of several failure modes. To improve the convergence stability, some training strategies start with an easier task, such as generating low-resolution images
[
16
]
 or simple images (one object with uniform background),
[
17
]
 and gradually increase the difficulty of the task during training. This essentially translates to applying a curriculum learning scheme.
[
18
]




Mode collapse
[
edit
]


GANs often suffer from 
mode collapse
 where they fail to generalize properly, missing entire modes from the input data. For example, a GAN trained on the 
MNIST
 dataset containing many samples of each digit might only generate pictures of digit 0. This was termed "the Helvetica scenario".
[
3
]


One way this can happen is if the generator learns too fast compared to the discriminator. If the discriminator 








D






{\displaystyle D}




 is held constant, then the optimal generator would only output elements of 








arg


⁡




max




x






D


(


x


)






{\displaystyle \arg \max _{x}D(x)}




.
[
19
]
 So for example, if during GAN training for generating MNIST dataset, for a few epochs, the discriminator somehow prefers the digit 0 slightly more than other digits, the generator may seize the opportunity to generate only digit 0, then be unable to escape the local minimum after the discriminator improves.

Some researchers perceive the root problem to be a weak discriminative network that fails to notice the pattern of omission, while others assign blame to a bad choice of 
objective function
. Many solutions have been proposed, but it is still an open problem.
[
20
]
[
21
]


Even the state-of-the-art architecture, BigGAN (2019), could not avoid mode collapse. The authors resorted to "allowing collapse to occur at the later stages of training, by which time a model is sufficiently trained to achieve good results".
[
22
]




Two time-scale update rule
[
edit
]


The 
two time-scale update rule (TTUR)
 is proposed to make GAN convergence more stable by making the learning rate of the generator lower than that of the discriminator. The authors argued that the generator should move slower than the discriminator, so that it does not "drive the discriminator steadily into new regions without capturing its gathered information".

They proved that a general class of games that included the GAN game, when trained under TTUR, "converges under mild assumptions to a stationary local Nash equilibrium".
[
23
]


They also proposed using the 
Adam stochastic optimization
[
24
]
 to avoid mode collapse, as well as the 
Fréchet inception distance
 for evaluating GAN performances.



Vanishing gradient
[
edit
]


Conversely, if the discriminator learns too fast compared to the generator, then the discriminator could almost perfectly distinguish 










μ






G




θ










,




μ




ref










{\displaystyle \mu _{G_{\theta }},\mu _{\text{ref}}}




. In such case, the generator 










G




θ










{\displaystyle G_{\theta }}




 could be stuck with a very high loss no matter which direction it changes its 








θ






{\displaystyle \theta }




, meaning that the gradient 










∇




θ






L


(




G




θ






,




D




ζ






)






{\displaystyle \nabla _{\theta }L(G_{\theta },D_{\zeta })}




 would be close to zero. In such case, the generator cannot learn, a case of the 
vanishing gradient
 problem
.
[
15
]


Intuitively speaking, the discriminator is too good, and since the generator cannot take any small step (only small steps are considered in gradient descent) to improve its payoff, it does not even try.

One important method for solving this problem is the 
Wasserstein GAN
.



Evaluation
[
edit
]


GANs are usually evaluated by 
Inception score
 (IS), which measures how varied the generator's outputs are (as classified by an image classifier, usually 
Inception-v3
), or 
Fréchet inception distance
 (FID), which measures how similar the generator's outputs are to a reference set (as classified by a learned image featurizer, such as Inception-v3 without its final layer). Many papers that propose new GAN architectures for image generation report how their architectures break the 
state of the art
 on FID or IS.

Another evaluation method is the Learned Perceptual Image Patch Similarity (LPIPS), which starts with a learned image featurizer 










f




θ






:




Image




→






R






n










{\displaystyle f_{\theta }:{\text{Image}}\to \mathbb {R} ^{n}}




, and finetunes it by supervised learning on a set of 








(


x


,




x


′




,




p


e


r


c


e


p


t


u


a


l


 


d


i


f


f


e


r


e


n


c


e




⁡


(


x


,




x


′




)


)






{\displaystyle (x,x',\operatorname {perceptual~difference} (x,x'))}




, where 








x






{\displaystyle x}




 is an image, 










x


′








{\displaystyle x'}




 is a perturbed version of it, and 










p


e


r


c


e


p


t


u


a


l


 


d


i


f


f


e


r


e


n


c


e




⁡


(


x


,




x


′




)






{\displaystyle \operatorname {perceptual~difference} (x,x')}




 is how much they differ, as reported by human subjects. The model is finetuned so that it can approximate 








‖




f




θ






(


x


)


−




f




θ






(




x


′




)


‖


≈




p


e


r


c


e


p


t


u


a


l


 


d


i


f


f


e


r


e


n


c


e




⁡


(


x


,




x


′




)






{\displaystyle \|f_{\theta }(x)-f_{\theta }(x')\|\approx \operatorname {perceptual~difference} (x,x')}




. This finetuned model is then used to define 








LPIPS


⁡


(


x


,




x


′




)


:=


‖




f




θ






(


x


)


−




f




θ






(




x


′




)


‖






{\displaystyle \operatorname {LPIPS} (x,x'):=\|f_{\theta }(x)-f_{\theta }(x')\|}




.
[
25
]


Other evaluation methods are reviewed in.
[
26
]




Variants
[
edit
]


There is a veritable zoo of GAN variants.
[
27
]
 Some of the most prominent are as follows:



Conditional GAN
[
edit
]


Conditional GANs are similar to standard GANs except they allow the model to conditionally generate samples based on additional information. For example, if we want to generate a cat face given a dog picture, we could use a conditional GAN.

The generator in a GAN game generates 










μ




G










{\displaystyle \mu _{G}}




, a probability distribution on the probability space 








Ω






{\displaystyle \Omega }




. This leads to the idea of a conditional GAN, where instead of generating one probability distribution on 








Ω






{\displaystyle \Omega }




, the generator generates a different probability distribution 










μ




G






(


c


)






{\displaystyle \mu _{G}(c)}




 on 








Ω






{\displaystyle \Omega }




, for each given class label 








c






{\displaystyle c}




.

For example, for generating images that look like 
ImageNet
, the generator should be able to generate a picture of cat when given the class label "cat".

In the original paper,
[
3
]
 the authors noted that GAN can be trivially extended to conditional GAN by providing the labels to both the generator and the discriminator.

Concretely, the conditional GAN game is just the GAN game with class labels provided:








L


(




μ




G






,


D


)


:=




E




c


∼




μ




C






,


x


∼




μ




ref






(


c


)






⁡


[


ln


⁡


D


(


x


,


c


)


]


+




E




c


∼




μ




C






,


x


∼




μ




G






(


c


)






⁡


[


ln


⁡


(


1


−


D


(


x


,


c


)


)


]






{\displaystyle L(\mu _{G},D):=\operatorname {E} _{c\sim \mu _{C},x\sim \mu _{\text{ref}}(c)}[\ln D(x,c)]+\operatorname {E} _{c\sim \mu _{C},x\sim \mu _{G}(c)}[\ln(1-D(x,c))]}




where 










μ




C










{\displaystyle \mu _{C}}




 is a probability distribution over classes, 










μ




ref






(


c


)






{\displaystyle \mu _{\text{ref}}(c)}




 is the probability distribution of real images of class 








c






{\displaystyle c}




, and 










μ




G






(


c


)






{\displaystyle \mu _{G}(c)}




 the probability distribution of images generated by the generator when given class label 








c






{\displaystyle c}




.

In 2017, a conditional GAN learned to generate 1000 image classes of 
ImageNet
.
[
28
]




GANs with alternative architectures
[
edit
]


The GAN game is a general framework and can be run with any reasonable parametrization of the generator 








G






{\displaystyle G}




 and discriminator 








D






{\displaystyle D}




. In the original paper, the authors demonstrated it using 
multilayer perceptron
 networks and 
convolutional neural networks
. Many alternative architectures have been tried.

Deep convolutional GAN (DCGAN):
[
29
]
 For both generator and discriminator, uses only deep networks consisting entirely of convolution-deconvolution layers, that is, fully convolutional networks.
[
30
]


Self-attention GAN (SAGAN):
[
31
]
 Starts with the DCGAN, then adds residually-connected standard 
self-attention modules
 to the generator and discriminator.

Variational autoencoder GAN (VAEGAN):
[
32
]
 Uses a 
variational autoencoder
 (VAE) for the generator.

Transformer GAN (TransGAN):
[
33
]
 Uses the pure 
transformer
 architecture for both the generator and discriminator, entirely devoid of convolution-deconvolution layers.

Flow-GAN:
[
34
]
 Uses 
flow-based generative model
 for the generator, allowing efficient computation of the likelihood function.



GANs with alternative objectives
[
edit
]


Many GAN variants are merely obtained by changing the loss functions for the generator and discriminator.

Original GAN:


We recast the original GAN objective into a form more convenient for comparison:












{










min




D








L




D






(


D


,




μ




G






)


=


−




E




x


∼




μ




G










⁡


[


ln


⁡


D


(


x


)


]


−




E




x


∼




μ




ref










⁡


[


ln


⁡


(


1


−


D


(


x


)


)


]












min




G








L




G






(


D


,




μ




G






)


=


−




E




x


∼




μ




G










⁡


[


ln


⁡


(


1


−


D


(


x


)


)


]


















{\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln D(x)]-\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln(1-D(x))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\end{cases}}}






Original GAN, non-saturating loss:


This objective for generator was recommended in the original paper for faster convergence.
[
3
]










L




G






=




E




x


∼




μ




G










⁡


[


ln


⁡


D


(


x


)


]






{\displaystyle L_{G}=\operatorname {E} _{x\sim \mu _{G}}[\ln D(x)]}




The effect of using this objective is analyzed in Section 2.2.2 of Arjovsky et al.
[
35
]


Original GAN, maximum likelihood:












L




G






=




E




x


∼




μ




G










⁡


[


(




exp




∘




σ




−


1






∘


D


)


(


x


)


]






{\displaystyle L_{G}=\operatorname {E} _{x\sim \mu _{G}}[({\exp }\circ \sigma ^{-1}\circ D)(x)]}




where 








σ






{\displaystyle \sigma }




 is the logistic function. When the discriminator is optimal, the generator gradient is the same as in 
maximum likelihood estimation
, even though GAN cannot perform maximum likelihood estimation 
itself
.
[
36
]
[
37
]


Hinge loss
 GAN
:
[
38
]










L




D






=


−




E




x


∼




p




ref










⁡




[




min




(




0


,


−


1


+


D


(


x


)




)






]




−




E




x


∼




μ




G










⁡




[




min




(




0


,


−


1


−


D




(


x


)






)






]








{\displaystyle L_{D}=-\operatorname {E} _{x\sim p_{\text{ref}}}\left[\min \left(0,-1+D(x)\right)\right]-\operatorname {E} _{x\sim \mu _{G}}\left[\min \left(0,-1-D\left(x\right)\right)\right]}














L




G






=


−




E




x


∼




μ




G










⁡


[


D


(


x


)


]






{\displaystyle L_{G}=-\operatorname {E} _{x\sim \mu _{G}}[D(x)]}




Least squares GAN:
[
39
]










L




D






=




E




x


∼




μ




ref










⁡


[


(


D


(


x


)


−


b




)




2






]


+




E




x


∼




μ




G










⁡


[


(


D


(


x


)


−


a




)




2






]






{\displaystyle L_{D}=\operatorname {E} _{x\sim \mu _{\text{ref}}}[(D(x)-b)^{2}]+\operatorname {E} _{x\sim \mu _{G}}[(D(x)-a)^{2}]}














L




G






=




E




x


∼




μ




G










⁡


[


(


D


(


x


)


−


c




)




2






]






{\displaystyle L_{G}=\operatorname {E} _{x\sim \mu _{G}}[(D(x)-c)^{2}]}




where 








a


,


b


,


c






{\displaystyle a,b,c}




 are parameters to be chosen. The authors recommended 








a


=


−


1


,


b


=


1


,


c


=


0






{\displaystyle a=-1,b=1,c=0}




.



Wasserstein GAN (WGAN)
[
edit
]


Main article: 
Wasserstein GAN


The Wasserstein GAN modifies the GAN game at two points:



The discriminator's strategy set is the set of measurable functions of type 








D


:


Ω


→




R








{\displaystyle D:\Omega \to \mathbb {R} }




 with bounded 
Lipschitz norm
: 








‖


D




‖




L






≤


K






{\displaystyle \|D\|_{L}\leq K}




, where 








K






{\displaystyle K}




 is a fixed positive constant.


The objective is










L




W


G


A


N






(




μ




G






,


D


)


:=




E




x


∼




μ




G










⁡


[


D


(


x


)


]


−






E






x


∼




μ




ref










[


D


(


x


)


]






{\displaystyle L_{WGAN}(\mu _{G},D):=\operatorname {E} _{x\sim \mu _{G}}[D(x)]-\mathbb {E} _{x\sim \mu _{\text{ref}}}[D(x)]}






One of its purposes is to solve the problem of mode collapse (see above).
[
15
]
 The authors claim "In no experiment did we see evidence of mode collapse for the WGAN algorithm".



GANs with more than two players
[
edit
]


Adversarial autoencoder
[
edit
]


An adversarial autoencoder (AAE)
[
40
]
 is more autoencoder than GAN. The idea is to start with a plain 
autoencoder
, but train a discriminator to discriminate the latent vectors from a reference distribution (often the normal distribution).



InfoGAN
[
edit
]


In conditional GAN, the generator receives both a noise vector 








z






{\displaystyle z}




 and a label 








c






{\displaystyle c}




, and produces an image 








G


(


z


,


c


)






{\displaystyle G(z,c)}




. The discriminator receives image-label pairs 








(


x


,


c


)






{\displaystyle (x,c)}




, and computes 








D


(


x


,


c


)






{\displaystyle D(x,c)}




.

When the training dataset is unlabeled, conditional GAN does not work directly.

The idea of InfoGAN is to decree that every latent vector in the latent space can be decomposed as 








(


z


,


c


)






{\displaystyle (z,c)}




: an incompressible noise part 








z






{\displaystyle z}




, and an informative label part 








c






{\displaystyle c}




, and encourage the generator to comply with the decree, by encouraging it to maximize 








I


(


c


,


G


(


z


,


c


)


)






{\displaystyle I(c,G(z,c))}




, the 
mutual information
 between 








c






{\displaystyle c}




 and 








G


(


z


,


c


)






{\displaystyle G(z,c)}




, while making no demands on the mutual information 








z






{\displaystyle z}




 between 








G


(


z


,


c


)






{\displaystyle G(z,c)}




.

Unfortunately, 








I


(


c


,


G


(


z


,


c


)


)






{\displaystyle I(c,G(z,c))}




 is intractable in general, The key idea of InfoGAN is Variational Mutual Information Maximization:
[
41
]
 indirectly maximize it by maximizing a lower bound














I


^








(


G


,


Q


)


=






E






z


∼




μ




Z






,


c


∼




μ




C










[


ln


⁡


Q


(


c


∣


G


(


z


,


c


)


)


]


;




I


(


c


,


G


(


z


,


c


)


)


≥




sup




Q












I


^








(


G


,


Q


)






{\displaystyle {\hat {I}}(G,Q)=\mathbb {E} _{z\sim \mu _{Z},c\sim \mu _{C}}[\ln Q(c\mid G(z,c))];\quad I(c,G(z,c))\geq \sup _{Q}{\hat {I}}(G,Q)}




where 








Q






{\displaystyle Q}




 ranges over all 
Markov kernels
 of type 








Q


:




Ω




Y






→






P






(




Ω




C






)






{\displaystyle Q:\Omega _{Y}\to {\mathcal {P}}(\Omega _{C})}




.


The InfoGAN game is defined as follows:
[
42
]
Three probability spaces define an InfoGAN game:









(




Ω




X






,




μ




ref






)






{\displaystyle (\Omega _{X},\mu _{\text{ref}})}




, the space of reference images.










(




Ω




Z






,




μ




Z






)






{\displaystyle (\Omega _{Z},\mu _{Z})}




, the fixed random noise generator.










(




Ω




C






,




μ




C






)






{\displaystyle (\Omega _{C},\mu _{C})}




, the fixed random information generator.


There are 3 players in 2 teams: generator, Q, and discriminator. The generator and Q are on one team, and the discriminator on the other team.

The objective function is








L


(


G


,


Q


,


D


)


=




L




G


A


N






(


G


,


D


)


−


λ








I


^








(


G


,


Q


)






{\displaystyle L(G,Q,D)=L_{GAN}(G,D)-\lambda {\hat {I}}(G,Q)}




where 










L




G


A


N






(


G


,


D


)


=




E




x


∼




μ




ref






,






⁡


[


ln


⁡


D


(


x


)


]


+




E




z


∼




μ




Z










⁡


[


ln


⁡


(


1


−


D


(


G


(


z


,


c


)


)


)


]






{\displaystyle L_{GAN}(G,D)=\operatorname {E} _{x\sim \mu _{\text{ref}},}[\ln D(x)]+\operatorname {E} _{z\sim \mu _{Z}}[\ln(1-D(G(z,c)))]}




 is the original GAN game objective, and 














I


^








(


G


,


Q


)


=






E






z


∼




μ




Z






,


c


∼




μ




C










[


ln


⁡


Q


(


c


∣


G


(


z


,


c


)


)


]






{\displaystyle {\hat {I}}(G,Q)=\mathbb {E} _{z\sim \mu _{Z},c\sim \mu _{C}}[\ln Q(c\mid G(z,c))]}







Generator-Q team aims to minimize the objective, and discriminator aims to maximize it:










min




G


,


Q








max




D






L


(


G


,


Q


,


D


)






{\displaystyle \min _{G,Q}\max _{D}L(G,Q,D)}






Bidirectional GAN (BiGAN)
[
edit
]


The standard GAN generator is a function of type 








G


:




Ω




Z






→




Ω




X










{\displaystyle G:\Omega _{Z}\to \Omega _{X}}




, that is, it is a mapping from a latent space 










Ω




Z










{\displaystyle \Omega _{Z}}




 to the image space 










Ω




X










{\displaystyle \Omega _{X}}




. This can be understood as a "decoding" process, whereby every latent vector 








z


∈




Ω




Z










{\displaystyle z\in \Omega _{Z}}




 is a code for an image 








x


∈




Ω




X










{\displaystyle x\in \Omega _{X}}




, and the generator performs the decoding. This naturally leads to the idea of training another network that performs "encoding", creating an 
autoencoder
 out of the encoder-generator pair.

Already in the original paper,
[
3
]
 the authors noted that "Learned approximate inference can be performed by training an auxiliary network to predict 








z






{\displaystyle z}




 given 








x






{\displaystyle x}




". The bidirectional GAN architecture performs exactly this.
[
43
]



The BiGAN is defined as follows: 
Two probability spaces define a BiGAN game:









(




Ω




X






,




μ




X






)






{\displaystyle (\Omega _{X},\mu _{X})}




, the space of reference images.










(




Ω




Z






,




μ




Z






)






{\displaystyle (\Omega _{Z},\mu _{Z})}




, the latent space.


There are 3 players in 2 teams: generator, encoder, and discriminator. The generator and encoder are on one team, and the discriminator on the other team.

The generator's strategies are functions 








G


:




Ω




Z






→




Ω




X










{\displaystyle G:\Omega _{Z}\to \Omega _{X}}




, and the encoder's strategies are functions 








E


:




Ω




X






→




Ω




Z










{\displaystyle E:\Omega _{X}\to \Omega _{Z}}




. The discriminator's strategies are functions 








D


:




Ω




X






→


[


0


,


1


]






{\displaystyle D:\Omega _{X}\to [0,1]}




.

The objective function is








L


(


G


,


E


,


D


)


=






E






x


∼




μ




X










[


ln


⁡


D


(


x


,


E


(


x


)


)


]


+






E






z


∼




μ




Z










[


ln


⁡


(


1


−


D


(


G


(


z


)


,


z


)


)


]






{\displaystyle L(G,E,D)=\mathbb {E} _{x\sim \mu _{X}}[\ln D(x,E(x))]+\mathbb {E} _{z\sim \mu _{Z}}[\ln(1-D(G(z),z))]}







Generator-encoder team aims to minimize the objective, and discriminator aims to maximize it:










min




G


,


E








max




D






L


(


G


,


E


,


D


)






{\displaystyle \min _{G,E}\max _{D}L(G,E,D)}




 
In the paper, they gave a more abstract definition of the objective as:








L


(


G


,


E


,


D


)


=






E






(


x


,


z


)


∼




μ




E


,


X










[


ln


⁡


D


(


x


,


z


)


]


+






E






(


x


,


z


)


∼




μ




G


,


Z










[


ln


⁡


(


1


−


D


(


x


,


z


)


)


]






{\displaystyle L(G,E,D)=\mathbb {E} _{(x,z)\sim \mu _{E,X}}[\ln D(x,z)]+\mathbb {E} _{(x,z)\sim \mu _{G,Z}}[\ln(1-D(x,z))]}




where 










μ




E


,


X






(


d


x


,


d


z


)


=




μ




X






(


d


x


)


⋅




δ




E


(


x


)






(


d


z


)






{\displaystyle \mu _{E,X}(dx,dz)=\mu _{X}(dx)\cdot \delta _{E(x)}(dz)}




 is the probability distribution on 










Ω




X






×




Ω




Z










{\displaystyle \Omega _{X}\times \Omega _{Z}}




 obtained by 
pushing 










μ




X










{\displaystyle \mu _{X}}




 forward
 via 








x


↦


(


x


,


E


(


x


)


)






{\displaystyle x\mapsto (x,E(x))}




, and 










μ




G


,


Z






(


d


x


,


d


z


)


=




δ




G


(


z


)






(


d


x


)


⋅




μ




Z






(


d


z


)






{\displaystyle \mu _{G,Z}(dx,dz)=\delta _{G(z)}(dx)\cdot \mu _{Z}(dz)}




 is the probability distribution on 










Ω




X






×




Ω




Z










{\displaystyle \Omega _{X}\times \Omega _{Z}}




 obtained by pushing 










μ




Z










{\displaystyle \mu _{Z}}




 forward via 








z


↦


(


G


(


x


)


,


z


)






{\displaystyle z\mapsto (G(x),z)}




.

Applications of bidirectional models include 
semi-supervised learning
,
[
44
]
 
interpretable machine learning
,
[
45
]
 and 
neural machine translation
.
[
46
]




CycleGAN
[
edit
]


CycleGAN is an architecture for performing translations between two domains, such as between photos of horses and photos of zebras, or photos of night cities and photos of day cities.


The CycleGAN game is defined as follows:
[
47
]
There are two probability spaces 








(




Ω




X






,




μ




X






)


,


(




Ω




Y






,




μ




Y






)






{\displaystyle (\Omega _{X},\mu _{X}),(\Omega _{Y},\mu _{Y})}




, corresponding to the two domains needed for translations fore-and-back.

There are 4 players in 2 teams: generators 










G




X






:




Ω




X






→




Ω




Y






,




G




Y






:




Ω




Y






→




Ω




X










{\displaystyle G_{X}:\Omega _{X}\to \Omega _{Y},G_{Y}:\Omega _{Y}\to \Omega _{X}}




, and discriminators 










D




X






:




Ω




X






→


[


0


,


1


]


,




D




Y






:




Ω




Y






→


[


0


,


1


]






{\displaystyle D_{X}:\Omega _{X}\to [0,1],D_{Y}:\Omega _{Y}\to [0,1]}




.

The objective function is








L


(




G




X






,




G




Y






,




D




X






,




D




Y






)


=




L




G


A


N






(




G




X






,




D




X






)


+




L




G


A


N






(




G




Y






,




D




Y






)


+


λ




L




c


y


c


l


e






(




G




X






,




G




Y






)






{\displaystyle L(G_{X},G_{Y},D_{X},D_{Y})=L_{GAN}(G_{X},D_{X})+L_{GAN}(G_{Y},D_{Y})+\lambda L_{cycle}(G_{X},G_{Y})}







where 








λ






{\displaystyle \lambda }




 is a positive adjustable parameter, 










L




G


A


N










{\displaystyle L_{GAN}}




 is the GAN game objective, and 










L




c


y


c


l


e










{\displaystyle L_{cycle}}




 is the 
cycle consistency loss
:










L




c


y


c


l


e






(




G




X






,




G




Y






)


=




E




x


∼




μ




X










‖




G




X






(




G




Y






(


x


)


)


−


x


‖


+




E




y


∼




μ




Y










‖




G




Y






(




G




X






(


y


)


)


−


y


‖






{\displaystyle L_{cycle}(G_{X},G_{Y})=E_{x\sim \mu _{X}}\|G_{X}(G_{Y}(x))-x\|+E_{y\sim \mu _{Y}}\|G_{Y}(G_{X}(y))-y\|}




The generators aim to minimize the objective, and the discriminators aim to maximize it:










min






G




X






,




G




Y












max






D




X






,




D




Y










L


(




G




X






,




G




Y






,




D




X






,




D




Y






)






{\displaystyle \min _{G_{X},G_{Y}}\max _{D_{X},D_{Y}}L(G_{X},G_{Y},D_{X},D_{Y})}




 
Unlike previous work like pix2pix,
[
48
]
 which requires paired training data, cycleGAN requires no paired data. For example, to train a pix2pix model to turn a summer scenery photo to winter scenery photo and back, the dataset must contain pairs of the same place in summer and winter, shot at the same angle; cycleGAN would only need a set of summer scenery photos, and an unrelated set of winter scenery photos.

GANs with particularly large or small scales
[
edit
]


BigGAN
[
edit
]


The BigGAN is essentially a self-attention GAN trained on a large scale (up to 80 million parameters) to generate large images of ImageNet (up to 512 x 512 resolution), with numerous engineering tricks to make it converge.
[
22
]
[
49
]




Invertible data augmentation
[
edit
]


When there is insufficient training data, the reference distribution 










μ




ref










{\displaystyle \mu _{\text{ref}}}




 cannot be well-approximated by the 
empirical distribution
 given by the training dataset. In such cases, 
data augmentation
 can be applied, to allow training GAN on smaller datasets. Naïve data augmentation, however, brings its problems.

Consider the original GAN game, slightly reformulated as follows:












{










min




D








L




D






(


D


,




μ




G






)


=


−




E




x


∼




μ




ref










⁡


[


ln


⁡


D


(


x


)


]


−




E




x


∼




μ




G










⁡


[


ln


⁡


(


1


−


D


(


x


)


)


]












min




G








L




G






(


D


,




μ




G






)


=


−




E




x


∼




μ




G










⁡


[


ln


⁡


(


1


−


D


(


x


)


)


]


















{\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{\text{ref}}}[\ln D(x)]-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\end{cases}}}




Now we use data augmentation by randomly sampling semantic-preserving transforms 








T


:


Ω


→


Ω






{\displaystyle T:\Omega \to \Omega }




 and applying them to the dataset, to obtain the reformulated GAN game:












{










min




D








L




D






(


D


,




μ




G






)


=


−




E




x


∼




μ




ref






,


T


∼




μ




trans










⁡


[


ln


⁡


D


(


T


(


x


)


)


]


−




E




x


∼




μ




G










⁡


[


ln


⁡


(


1


−


D


(


x


)


)


]












min




G








L




G






(


D


,




μ




G






)


=


−




E




x


∼




μ




G










⁡


[


ln


⁡


(


1


−


D


(


x


)


)


]


















{\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{\text{ref}},T\sim \mu _{\text{trans}}}[\ln D(T(x))]-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G}}[\ln(1-D(x))]\end{cases}}}




This is equivalent to a GAN game with a different distribution 










μ




ref




′








{\displaystyle \mu _{\text{ref}}'}




, sampled by 








T


(


x


)






{\displaystyle T(x)}




, with 








x


∼




μ




ref






,


T


∼




μ




trans










{\displaystyle x\sim \mu _{\text{ref}},T\sim \mu _{\text{trans}}}




. For example, if 










μ




ref










{\displaystyle \mu _{\text{ref}}}




 is the distribution of images in ImageNet, and 










μ




trans










{\displaystyle \mu _{\text{trans}}}




 samples identity-transform with probability 0.5, and horizontal-reflection with probability 0.5, then 










μ




ref




′








{\displaystyle \mu _{\text{ref}}'}




 is the distribution of images in ImageNet and horizontally-reflected ImageNet, combined.

The result of such training would be a generator that mimics 










μ




ref




′








{\displaystyle \mu _{\text{ref}}'}




. For example, it would generate images that look like they are randomly cropped, if the data augmentation uses random cropping.

The solution is to apply data augmentation to both generated and real images:












{










min




D








L




D






(


D


,




μ




G






)


=


−




E




x


∼




μ




ref






,


T


∼




μ




trans










⁡


[


ln


⁡


D


(


T


(


x


)


)


]


−




E




x


∼




μ




G






,


T


∼




μ




trans










⁡


[


ln


⁡


(


1


−


D


(


T


(


x


)


)


)


]












min




G








L




G






(


D


,




μ




G






)


=


−




E




x


∼




μ




G






,


T


∼




μ




trans










⁡


[


ln


⁡


(


1


−


D


(


T


(


x


)


)


)


]


















{\displaystyle {\begin{cases}\min _{D}L_{D}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{\text{ref}},T\sim \mu _{\text{trans}}}[\ln D(T(x))]-\operatorname {E} _{x\sim \mu _{G},T\sim \mu _{\text{trans}}}[\ln(1-D(T(x)))]\\\min _{G}L_{G}(D,\mu _{G})=-\operatorname {E} _{x\sim \mu _{G},T\sim \mu _{\text{trans}}}[\ln(1-D(T(x)))]\end{cases}}}




The authors demonstrated high-quality generation using just 100-picture-large datasets.
[
50
]


The StyleGAN-2-ADA paper points out a further point on data augmentation: it must be 
invertible
.
[
51
]
 Continue with the example of generating ImageNet pictures. If the data augmentation is "randomly rotate the picture by 0, 90, 180, 270 degrees with 
equal
 probability", then there is no way for the generator to know which is the true orientation: Consider two generators 








G


,




G


′








{\displaystyle G,G'}




, such that for any latent 








z






{\displaystyle z}




, the generated image 








G


(


z


)






{\displaystyle G(z)}




 is a 90-degree rotation of 










G


′




(


z


)






{\displaystyle G'(z)}




. They would have exactly the same expected loss, and so neither is preferred over the other.

The solution is to only use invertible data augmentation: instead of "randomly rotate the picture by 0, 90, 180, 270 degrees with 
equal
 probability", use "randomly rotate the picture by 90, 180, 270 degrees with 0.1 probability, and keep the picture as it is with 0.7 probability". This way, the generator is still rewarded  to keep images oriented the same way as un-augmented ImageNet pictures.

Abstractly, the effect of randomly sampling transformations 








T


:


Ω


→


Ω






{\displaystyle T:\Omega \to \Omega }




 from the distribution 










μ




trans










{\displaystyle \mu _{\text{trans}}}




 is to define a Markov kernel 










K




trans






:


Ω


→






P






(


Ω


)






{\displaystyle K_{\text{trans}}:\Omega \to {\mathcal {P}}(\Omega )}




. Then, the data-augmented GAN game pushes the generator to find some 
















μ


^










G






∈






P






(


Ω


)






{\displaystyle {\hat {\mu }}_{G}\in {\mathcal {P}}(\Omega )}




, such that 










K




trans






∗




μ




ref






=




K




trans






∗










μ


^










G










{\displaystyle K_{\text{trans}}*\mu _{\text{ref}}=K_{\text{trans}}*{\hat {\mu }}_{G}}




where 








∗






{\displaystyle *}




 is the 
Markov kernel convolution
.
A data-augmentation method is defined to be 
invertible
 if its Markov kernel 










K




trans










{\displaystyle K_{\text{trans}}}




 satisfies










K




trans






∗


μ


=




K




trans






∗




μ


′






⟹




μ


=




μ


′






∀


μ


,




μ


′




∈






P






(


Ω


)






{\displaystyle K_{\text{trans}}*\mu =K_{\text{trans}}*\mu '\implies \mu =\mu '\quad \forall \mu ,\mu '\in {\mathcal {P}}(\Omega )}




Immediately by definition, we see that composing multiple invertible data-augmentation methods results in yet another invertible method. Also by definition, if the data-augmentation method is invertible, then using it in a GAN game does not change the optimal strategy 
















μ


^










G










{\displaystyle {\hat {\mu }}_{G}}




 for the generator, which is still 










μ




ref










{\displaystyle \mu _{\text{ref}}}




.

There are two prototypical examples of invertible Markov kernels:

Discrete case
: Invertible 
stochastic matrices
, when 








Ω






{\displaystyle \Omega }




 is finite.

For example, if 








Ω


=


{


↑


,


↓


,


←


,


→


}






{\displaystyle \Omega =\{\uparrow ,\downarrow ,\leftarrow ,\rightarrow \}}




 is the set of four images of an arrow, pointing in 4 directions, and the data augmentation is "randomly rotate the picture by 90, 180, 270 degrees with probability 








p






{\displaystyle p}




, and keep the picture as it is with probability 








(


1


−


3


p


)






{\displaystyle (1-3p)}




", then the Markov kernel 










K




trans










{\displaystyle K_{\text{trans}}}




 can be represented as a stochastic matrix:








[




K




trans






]


=






[








(


1


−


3


p


)






p






p






p










p






(


1


−


3


p


)






p






p










p






p






(


1


−


3


p


)






p










p






p






p






(


1


−


3


p


)








]










{\displaystyle [K_{\text{trans}}]={\begin{bmatrix}(1-3p)&p&p&p\\p&(1-3p)&p&p\\p&p&(1-3p)&p\\p&p&p&(1-3p)\end{bmatrix}}}




 and 










K




trans










{\displaystyle K_{\text{trans}}}




 is an invertible kernel iff 








[




K




trans






]






{\displaystyle [K_{\text{trans}}]}




 is an invertible matrix, that is, 








p


≠


1




/




4






{\displaystyle p\neq 1/4}




.

Continuous case
: The gaussian kernel, when 








Ω


=






R






n










{\displaystyle \Omega =\mathbb {R} ^{n}}




 for some 








n


≥


1






{\displaystyle n\geq 1}




.

For example, if 








Ω


=






R








256




2














{\displaystyle \Omega =\mathbb {R} ^{256^{2}}}




 is the space of 256x256 images, and the data-augmentation method is "generate a gaussian noise 








z


∼






N






(


0


,




I






256




2










)






{\displaystyle z\sim {\mathcal {N}}(0,I_{256^{2}})}




, then add 








ϵ


z






{\displaystyle \epsilon z}




 to the image", then 










K




trans










{\displaystyle K_{\text{trans}}}




 is just convolution by the density function of 












N






(


0


,




ϵ




2








I






256




2










)






{\displaystyle {\mathcal {N}}(0,\epsilon ^{2}I_{256^{2}})}




. This is invertible, because convolution by a gaussian is just convolution by the 
heat kernel
, so given any 








μ


∈






P






(






R






n






)






{\displaystyle \mu \in {\mathcal {P}}(\mathbb {R} ^{n})}




, the convolved distribution 










K




trans






∗


μ






{\displaystyle K_{\text{trans}}*\mu }




 can be obtained by heating up 












R






n










{\displaystyle \mathbb {R} ^{n}}




 precisely according to 








μ






{\displaystyle \mu }




, then wait for time 










ϵ




2








/




4






{\displaystyle \epsilon ^{2}/4}




. With that, we can recover 








μ






{\displaystyle \mu }




 by running the 
heat equation
 
backwards in time
 for 










ϵ




2








/




4






{\displaystyle \epsilon ^{2}/4}




.

More examples of invertible data augmentations are found in the paper.
[
51
]




SinGAN
[
edit
]


SinGAN pushes data augmentation to the limit, by using only a single image as training data and performing data augmentation on it. The GAN architecture is adapted to this training method by using a multi-scale pipeline.

The generator 








G






{\displaystyle G}




 is decomposed into a pyramid of generators 








G


=




G




1






∘




G




2






∘


⋯


∘




G




N










{\displaystyle G=G_{1}\circ G_{2}\circ \cdots \circ G_{N}}




, with the lowest one generating the image 










G




N






(




z




N






)






{\displaystyle G_{N}(z_{N})}




 at the lowest resolution, then the generated image is scaled up to 








r


(




G




N






(




z




N






)


)






{\displaystyle r(G_{N}(z_{N}))}




, and fed to the next level to generate an image 










G




N


−


1






(




z




N


−


1






+


r


(




G




N






(




z




N






)


)


)






{\displaystyle G_{N-1}(z_{N-1}+r(G_{N}(z_{N})))}




 at a higher resolution, and so on. The discriminator is decomposed into a pyramid as well.
[
52
]




StyleGAN series
[
edit
]


Main article: 
StyleGAN


The StyleGAN family is a series of architectures published by 
Nvidia
's research division.



Progressive GAN
[
edit
]


Progressive GAN
[
16
]
 is a method for training GAN for large-scale image generation stably, by growing a GAN generator from small to large scale in a pyramidal fashion. Like SinGAN, it decomposes the generator as








G


=




G




1






∘




G




2






∘


⋯


∘




G




N










{\displaystyle G=G_{1}\circ G_{2}\circ \cdots \circ G_{N}}




, and the discriminator as 








D


=




D




1






∘




D




2






∘


⋯


∘




D




N










{\displaystyle D=D_{1}\circ D_{2}\circ \cdots \circ D_{N}}




.

During training, at first only 










G




N






,




D




N










{\displaystyle G_{N},D_{N}}




 are used in a GAN game to generate 4x4 images. Then 










G




N


−


1






,




D




N


−


1










{\displaystyle G_{N-1},D_{N-1}}




 are added to reach the second stage of GAN game, to generate 8x8 images, and so on, until we reach a GAN game to generate 1024x1024 images.

To avoid shock between stages of the GAN game, each new layer is "blended in" (Figure 2 of the paper
[
16
]
). For example, this is how the second stage GAN game starts:



Just before, the GAN game consists of the pair 










G




N






,




D




N










{\displaystyle G_{N},D_{N}}




 generating and discriminating 4x4 images.


Just after, the GAN game consists of the pair 








(


(


1


−


α


)


+


α


⋅




G




N


−


1






)


∘


u


∘




G




N






,




D




N






∘


d


∘


(


(


1


−


α


)


+


α


⋅




D




N


−


1






)






{\displaystyle ((1-\alpha )+\alpha \cdot G_{N-1})\circ u\circ G_{N},D_{N}\circ d\circ ((1-\alpha )+\alpha \cdot D_{N-1})}




 generating and discriminating 8x8 images. Here, the functions 








u


,


d






{\displaystyle u,d}




 are image up- and down-sampling functions, and 








α






{\displaystyle \alpha }




 is a blend-in factor (much like an 
alpha
 in image composing) that smoothly glides from 0 to 1.


StyleGAN-1
[
edit
]


The main architecture of StyleGAN-1 and StyleGAN-2


StyleGAN-1 is designed as a combination of Progressive GAN with 
neural style transfer
.
[
53
]


The key architectural choice of StyleGAN-1 is a progressive growth mechanism, similar to Progressive GAN. Each generated image starts as a constant 








4


×


4


×


512






{\displaystyle 4\times 4\times 512}




 array, and repeatedly passed through style blocks. Each style block applies a "style latent vector" via affine transform ("adaptive instance normalization"), similar to how neural style transfer uses 
Gramian matrix
. It then adds noise, and normalize (subtract the mean, then divide by the variance).

At training time, usually only one style latent vector is used per image generated, but sometimes two ("mixing regularization") in order to encourage each style block to independently perform its stylization without expecting help from other style blocks (since they might receive an entirely different style latent vector).

After training, multiple style latent vectors can be fed into each style block. Those fed to the lower layers control the large-scale styles, and those fed to the higher layers control the fine-detail styles.

Style-mixing between two images 








x


,




x


′








{\displaystyle x,x'}




 can be performed as well. First, run a gradient descent to find 








z


,




z


′








{\displaystyle z,z'}




 such that 








G


(


z


)


≈


x


,


G


(




z


′




)


≈




x


′








{\displaystyle G(z)\approx x,G(z')\approx x'}




. This is called "projecting an image back to style latent space". Then, 








z






{\displaystyle z}




 can be fed to the lower style blocks, and 










z


′








{\displaystyle z'}




 to the higher style blocks, to generate a composite image that has the large-scale style of 








x






{\displaystyle x}




, and the fine-detail style of 










x


′








{\displaystyle x'}




. Multiple images can also be composed this way.



StyleGAN-2
[
edit
]


StyleGAN-2 improves upon StyleGAN-1, by using the style latent vector to transform the convolution layer's weights instead, thus solving the "blob" problem.
[
54
]


This was updated by the StyleGAN-2-ADA ("ADA" stands for "adaptive"),
[
51
]
 which uses invertible data augmentation as described above. It also tunes the amount of data augmentation applied by starting at zero, and gradually increasing it until an "overfitting heuristic" reaches a target level, thus the name "adaptive".



StyleGAN-3
[
edit
]


StyleGAN-3
[
55
]
 improves upon StyleGAN-2 by solving the "texture sticking" problem, which can be seen in the official videos.
[
56
]
 They analyzed the problem by the 
Nyquist–Shannon sampling theorem
, and argued that the layers in the generator learned to exploit the high-frequency signal in the pixels they operate upon.

To solve this, they proposed imposing strict 
lowpass filters
 between each generator's layers, so that the generator is forced to operate on the pixels in a way 
faithful
 to the continuous signals they represent, rather than operate on them as merely discrete signals. They further imposed rotational and translational invariance by using more 
signal filters
. The resulting StyleGAN-3 is able to solve the texture sticking problem, as well as generating images that rotate and translate smoothly.



Applications
[
edit
]


GAN applications have increased rapidly.
[
57
]




Transfer learning
[
edit
]


State-of-art 
transfer learning
 research use GANs to enforce the alignment of the latent feature space, such as in deep reinforcement learning.
[
58
]
 This works by feeding the embeddings of the source and target task to the discriminator which tries to guess the context. The resulting loss is then (inversely) backpropagated through the encoder.



Commercial
[
edit
]


GANs that produce 
photorealistic
 images can be used to visualize 
interior design
, 
industrial design
, shoes,
[
59
]
 bags, and 
clothing
 items or items for 
computer games
' scenes.
[
citation needed
]
 Such networks were reported to be used by 
Facebook
.
[
60
]




Fashion, art and advertising
[
edit
]


GANs can be used to generate art; 
The Verge
 wrote in March 2019 that "The images created by GANs have become the defining look of contemporary AI art."
[
61
]
 GANs can also be used to 
inpaint
 photographs
[
62
]
 or create photos of imaginary fashion models, with no need to hire a model, photographer or makeup artist, or pay for a studio and transportation.
[
63
]
 GANs have also been used for virtual shadow generation.
[
64
]


In 2018, GANs reached the 
video game modding
 community, as a method of 
up-scaling
 low-resolution 2D textures in old video games by recreating them in 
4k
 or higher resolutions via image training, and then down-sampling them to fit the game's native resolution (with results resembling the 
supersampling
 method of 
anti-aliasing
).
[
65
]
 With proper training, GANs provide a clearer and sharper 2D texture image magnitudes higher in quality than the original, while fully retaining the original's level of details, colors, etc. Known examples of extensive GAN usage include 
Final Fantasy VIII
, 
Final Fantasy IX
, 
Resident Evil REmake
 HD Remaster, and 
Max Payne
. 
[
citation needed
]


In 2020, 
Artbreeder
 was used to create the main antagonist in the sequel to the psychological web horror series 
Ben Drowned
. The author would later go on to praise GAN applications for their ability to help generate assets for independent artists who are short on budget and manpower.
[
66
]
[
67
]


In May 2020, 
Nvidia
 researchers taught an AI system (termed "GameGAN") to recreate the game of 
Pac-Man
 simply by watching it being played.
[
68
]
[
69
]




Science
[
edit
]


GANs can 
improve
 
astronomical images
[
70
]
 and simulate 
gravitational lensing
 for dark matter research.
[
71
]
[
72
]
[
73
]
 They were used in 2019 to successfully model the distribution of 
dark matter
 in a particular direction in space and to predict the gravitational lensing that will occur.
[
74
]
[
75
]


GANs have been proposed as a fast and accurate way of modeling high energy jet formation
[
76
]
 and modeling 
showers
 through 
calorimeters
 of 
high-energy physics
 experiments.
[
77
]
[
78
]
[
79
]
[
80
]
 GANs have also been trained to accurately approximate bottlenecks in computationally expensive simulations of particle physics experiments. Applications in the context of present and proposed 
CERN
 experiments have demonstrated the potential of these methods for accelerating simulation and/or improving simulation fidelity.
[
81
]
[
82
]


In 2016 GANs were used to generate new molecules for a variety of protein targets implicated in cancer, inflammation, and fibrosis. In 2019 GAN-generated molecules were validated experimentally all the way into mice.
[
83
]
[
84
]
 Moreover, GANs have gained significant attention for their potential in reconstructing velocity and scalar fields in turbulent flows.
[
85
]
[
86
]
[
87
]




Medical
[
edit
]


One of the major concerns in medical imaging is preserving patient privacy. Due to these reasons, researchers often face difficulties in obtaining medical images for their research purposes. Recently, GANs have been widely used for generating synthetic medical images, such as MRI and PET images, to address this challenge. 
[
88
]


GAN can be used to detect glaucomatous images helping the early diagnosis which is essential to avoid partial or total loss
of vision.
[
89
]


GANs have been used to create 
forensic facial reconstructions
 of deceased historical figures.
[
90
]




Audio
[
edit
]


In August 2019, a large dataset consisting of 12,197 MIDI songs each with paired lyrics and melody alignment was created for neural melody generation from lyrics using conditional GAN-LSTM (refer to sources at GitHub 
AI Melody Generation from Lyrics
).
[
91
]




Concerns about malicious applications
[
edit
]


Main article: 
Deepfake


An image generated by a 
StyleGAN
 that looks deceptively like a photograph of a real person. This image was generated by a StyleGAN based on an analysis of portraits.


Another example of a GAN generated portrait


Concerns have been raised about the potential use of GAN-based 
human image synthesis
 for sinister purposes, e.g., to produce fake, possibly incriminating, photographs and videos.
[
92
]

GANs can be used to generate unique, realistic profile photos of people who do not exist, in order to automate creation of fake social media profiles.
[
93
]


In 2019 the state of California considered
[
94
]
 and passed on October 3, 2019, the 
bill AB-602
, which bans the use of human image synthesis technologies to make fake pornography without the consent of the people depicted, and 
bill AB-730
, which prohibits distribution of manipulated videos of a political candidate within 60 days of an election. Both bills were authored by Assembly member 
Marc Berman
 and signed by Governor 
Gavin Newsom
. The laws went into effect in 2020.
[
95
]


DARPA's Media Forensics program studies ways to counteract fake media, including fake media produced using GANs.
[
96
]




Miscellaneous applications
[
edit
]


GANs have been used to 



show how an individual's appearance might change with age.
[
97
]


reconstruct 3D models of objects from images
,
[
98
]


generate novel objects as 3D point clouds,
[
99
]


model patterns of motion in video.
[
100
]


inpaint missing features in maps, transfer map styles in cartography
[
101
]
 or augment street view imagery.
[
102
]


use feedback to generate images and replace image search systems.
[
103
]


visualize the effect that climate change will have on specific houses.
[
104
]


reconstruct an image of a person's face after listening to their voice.
[
105
]


produces videos of a person speaking, given only a single photo of that person.
[
106
]


recurrent sequence generation.
[
107
]


History
[
edit
]


In 1991, 
Juergen Schmidhuber
 published "artificial curiosity", 
neural networks
 in a 
zero-sum game
.
[
108
]
 The first network is a 
generative model
 that models a 
probability distribution
 over output patterns. The second network learns by 
gradient descent
 to predict the reactions of the environment to these patterns. GANs can be regarded as a case where the environmental reaction is 1 or 0 depending on whether the first network's output is in a given set.
[
109
]


Other people had similar ideas but did not develop them similarly. An idea involving adversarial networks was published in a 2010 blog post by Olli Niemitalo.
[
110
]
 This idea was never implemented and did not involve 
stochasticity
 in the generator and thus was not a generative model. It is now known as a conditional GAN or cGAN.
[
111
]
 An idea similar to GANs was used to model animal behavior by Li, Gauci and Gross in 2013.
[
112
]


Another inspiration for GANs was noise-contrastive estimation,
[
113
]
 which uses the same loss function as GANs and which Goodfellow studied during his PhD in 2010–2014.

Adversarial machine learning
 has other uses besides generative modeling and can be applied to models other than neural networks. In control theory, adversarial learning based on neural networks was used in 2006 to train robust controllers in a game theoretic sense, by alternating the iterations between a minimizer policy, the controller, and a maximizer policy, the disturbance.
[
114
]
[
115
]


In 2017, a GAN was used for image enhancement focusing on realistic textures rather than pixel-accuracy, producing a higher image quality at high magnification.
[
116
]
 In 2017, the first faces were generated.
[
117
]
 These were exhibited in February 2018 at the Grand Palais.
[
118
]
[
119
]
 Faces generated by 
StyleGAN
[
120
]
 in 2019 drew comparisons with 
Deepfakes
.
[
121
]
[
122
]
[
123
]


Beginning in 2017, GAN technology began to make its presence felt in the fine arts arena with the appearance of a newly developed implementation which was said to have crossed the threshold of being able to generate unique and appealing abstract paintings, and thus dubbed a "CAN", for "creative adversarial network".
[
124
]
 A GAN system was used to create the 2018 painting 
Edmond de Belamy
,
 which sold for US$432,500.
[
125
]
 An early 2019 article by members of the original CAN team discussed further progress with that system, and gave consideration as well to the overall prospects for an AI-enabled art.
[
126
]




References
[
edit
]






^
 
"Generative AI and Future"
. November 15, 2022.




^
 
"CSDL | IEEE Computer Society"
.




^ 
a
 
b
 
c
 
d
 
e
 
f
 
g
 
h
 
i
 
j
 
Goodfellow, Ian; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua (2014). 
Generative Adversarial Nets
 
(PDF)
. Proceedings of the International Conference on Neural Information Processing Systems (NIPS 2014). pp. 2672–2680.




^
 
Salimans, Tim; Goodfellow, Ian; Zaremba, Wojciech; Cheung, Vicki; Radford, Alec; Chen, Xi (2016). "Improved Techniques for Training GANs". 
arXiv
:
1606.03498
 [
cs.LG
].




^
 
Isola, Phillip; Zhu, Jun-Yan; Zhou, Tinghui; Efros, Alexei (2017). 
"Image-to-Image Translation with Conditional Adversarial Nets"
. 
Computer Vision and Pattern Recognition
.




^
 
Ho, Jonathon; Ermon, Stefano (2016). 
"Generative Adversarial Imitation Learning"
. 
Advances in Neural Information Processing Systems
. 
29
: 4565–4573. 
arXiv
:
1606.03476
.




^
 
"Vanilla GAN (GANs in computer vision: Introduction to generative learning)"
. 
theaisummer.com
. AI Summer. April 10, 2020. 
Archived
 from the original on June 3, 2020
. Retrieved 
September 20,
 2020
.




^
 
Luc, Pauline; Couprie, Camille; Chintala, Soumith; Verbeek, Jakob (November 25, 2016). "Semantic Segmentation using Adversarial Networks". 
NIPS Workshop on Adversarial Training, Dec, Barcelona, Spain
. 
2016
. 
arXiv
:
1611.08408
.




^
 
Andrej Karpathy
; 
Pieter Abbeel
; Greg Brockman; Peter Chen; Vicki Cheung; Rocky Duan; Ian Goodfellow; Durk Kingma; Jonathan Ho; Rein Houthooft; Tim Salimans; John Schulman; Ilya Sutskever; Wojciech Zaremba, 
Generative Models
, 
OpenAI
, retrieved 
April 7,
 2016




^
 
Mohamed, Shakir; Lakshminarayanan, Balaji (2016). "Learning in Implicit Generative Models". 
arXiv
:
1610.03483
 [
stat.ML
].




^ 
a
 
b
 
Goodfellow, Ian (April 3, 2017). "NIPS 2016 Tutorial: Generative Adversarial Networks". 
arXiv
:
1701.00160
 [
cs.LG
].




^
 
Kingma, Diederik P.; Welling, Max (May 1, 2014). "Auto-Encoding Variational Bayes". 
arXiv
:
1312.6114
 [
stat.ML
].




^
 
Rezende, Danilo Jimenez; Mohamed, Shakir; Wierstra, Daan (June 18, 2014). 
"Stochastic Backpropagation and Approximate Inference in Deep Generative Models"
. 
International Conference on Machine Learning
. PMLR: 1278–1286. 
arXiv
:
1401.4082
.




^ 
a
 
b
 
Farnia, Farzan; Ozdaglar, Asuman (November 21, 2020). 
"Do GANs always have Nash equilibria?"
. 
International Conference on Machine Learning
. PMLR: 3029–3039.




^ 
a
 
b
 
c
 
Weng, Lilian (April 18, 2019). "From GAN to WGAN". 
arXiv
:
1904.08994
 [
cs.LG
].




^ 
a
 
b
 
c
 
Karras, Tero; Aila, Timo; Laine, Samuli; Lehtinen, Jaakko (October 1, 2017). "Progressive Growing of GANs for Improved Quality, Stability, and Variation". 
arXiv
:
1710.10196
 [
cs.NE
].




^
 
Soviany, Petru; Ardei, Claudiu; Ionescu, Radu Tudor; Leordeanu, Marius (October 22, 2019). "Image Difficulty Curriculum for Generative Adversarial Networks (CuGAN)". 
arXiv
:
1910.08967
 [
cs.LG
].




^
 
Hacohen, Guy; Weinshall, Daphna (May 24, 2019). 
"On The Power of Curriculum Learning in Training Deep Networks"
. 
International Conference on Machine Learning
. PMLR: 2535–2544. 
arXiv
:
1904.03626
.




^
 
"r/MachineLearning - Comment by u/ian_goodfellow on "[R] [1701.07875] Wasserstein GAN"
. 
reddit
. January 30, 2017
. Retrieved 
July 15,
 2022
.




^
 
Lin, Zinan; et al. (December 2018). 
PacGAN: the power of two samples in generative adversarial networks
. 32nd International Conference on Neural Information Processing Systems. pp. 1505–1514. 
arXiv
:
1712.04086
.




^
 
Mescheder, Lars; Geiger, Andreas; Nowozin, Sebastian (July 31, 2018). "Which Training Methods for GANs do actually Converge?". 
arXiv
:
1801.04406
 [
cs.LG
].




^ 
a
 
b
 
Brock, Andrew; Donahue, Jeff; Simonyan, Karen (September 1, 2018). 
Large Scale GAN Training for High Fidelity Natural Image Synthesis
. International Conference on Learning Representations 2019. 
arXiv
:
1809.11096
.




^
 
Heusel, Martin; Ramsauer, Hubert; Unterthiner, Thomas; Nessler, Bernhard; Hochreiter, Sepp (2017). 
"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
. 
Advances in Neural Information Processing Systems
. 
30
. Curran Associates, Inc. 
arXiv
:
1706.08500
.




^
 
Kingma, Diederik P.; Ba, Jimmy (January 29, 2017). "Adam: A Method for Stochastic Optimization". 
arXiv
:
1412.6980
 [
cs.LG
].




^
 
Zhang, Richard; Isola, Phillip; Efros, Alexei A.; Shechtman, Eli; Wang, Oliver (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric". pp. 586–595. 
arXiv
:
1801.03924
 [
cs.CV
].




^
 
Borji, Ali (February 1, 2019). 
"Pros and cons of GAN evaluation measures"
. 
Computer Vision and Image Understanding
. 
179
: 41–65. 
arXiv
:
1802.03446
. 
doi
:
10.1016/j.cviu.2018.10.009
. 
ISSN
 
1077-3142
. 
S2CID
 
3627712
.




^
 
Hindupur, Avinash (July 15, 2022), 
The GAN Zoo
, retrieved 
July 15,
 2022




^
 
Odena, Augustus; Olah, Christopher; Shlens, Jonathon (July 17, 2017). 
"Conditional Image Synthesis with Auxiliary Classifier GANs"
. 
International Conference on Machine Learning
. PMLR: 2642–2651. 
arXiv
:
1610.09585
.




^
 
Radford, Alec; Metz, Luke; Chintala, Soumith (2016). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks". 
ICLR
. 
S2CID
 
11758569
.




^
 
Long, Jonathan; Shelhamer, Evan; Darrell, Trevor (2015). 
"Fully Convolutional Networks for Semantic Segmentation"
. 
CVF
: 3431–3440.




^
 
Zhang, Han; Goodfellow, Ian; Metaxas, Dimitris; Odena, Augustus (May 24, 2019). 
"Self-Attention Generative Adversarial Networks"
. 
International Conference on Machine Learning
. PMLR: 7354–7363.




^
 
Larsen, Anders Boesen Lindbo; Sønderby, Søren Kaae; Larochelle, Hugo; Winther, Ole (June 11, 2016). 
"Autoencoding beyond pixels using a learned similarity metric"
. 
International Conference on Machine Learning
. PMLR: 1558–1566. 
arXiv
:
1512.09300
.




^
 
Jiang, Yifan; Chang, Shiyu; Wang, Zhangyang (December 8, 2021). "TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up". 
arXiv
:
2102.07074
 [
cs.CV
].




^
 
Grover, Aditya; Dhar, Manik; Ermon, Stefano (May 1, 2017). "Flow-GAN: Combining Maximum Likelihood and Adversarial Learning in Generative Models". 
arXiv
:
1705.08868
 [
cs.LG
].




^
 
Arjovsky, Martin; Bottou, Léon (January 1, 2017). "Towards Principled Methods for Training Generative Adversarial Networks". 
arXiv
:
1701.04862
 [
stat.ML
].




^
 
Goodfellow, Ian J. (December 1, 2014). "On distinguishability criteria for estimating generative models". 
arXiv
:
1412.6515
 [
stat.ML
].




^
 
Goodfellow, Ian (August 31, 2016). 
"Generative Adversarial Networks (GANs), Presentation at Berkeley Artificial Intelligence Lab"
 
(PDF)
. 
Archived
 
(PDF)
 from the original on May 8, 2022.




^
 
Lim, Jae Hyun; Ye, Jong Chul (May 8, 2017). "Geometric GAN". 
arXiv
:
1705.02894
 [
stat.ML
].




^
 
Mao, Xudong; Li, Qing; Xie, Haoran; Lau, Raymond Y. K.; Wang, Zhen; Paul Smolley, Stephen (2017). 
"Least Squares Generative Adversarial Networks"
: 2794–2802.
 
{{
cite journal
}}
: 
Cite journal requires 
|journal=
 (
help
)




^
 
Makhzani, Alireza; Shlens, Jonathon; Jaitly, Navdeep; 
Goodfellow, Ian
; 
Frey, Brendan
 (2016). "Adversarial Autoencoders". 
arXiv
:
1511.05644
 [
cs.LG
].




^
 
Barber, David; Agakov, Felix (December 9, 2003). 
"The IM algorithm: a variational approach to Information Maximization"
. 
Proceedings of the 16th International Conference on Neural Information Processing Systems
. NIPS'03. Cambridge, MA, USA: MIT Press: 201–208.




^
 
Chen, Xi; Duan, Yan; Houthooft, Rein; Schulman, John; Sutskever, Ilya; Abbeel, Pieter (2016). 
"InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"
. 
Advances in Neural Information Processing Systems
. 
29
. Curran Associates, Inc. 
arXiv
:
1606.03657
.




^
 
Donahue, Jeff; Krähenbühl, Philipp; 
Darrell, Trevor
 (2016). "Adversarial Feature Learning". 
arXiv
:
1605.09782
 [
cs.LG
].




^
 
Dumoulin, Vincent; Belghazi, Ishmael; Poole, Ben; Mastropietro, Olivier; Arjovsky, Alex; Courville, Aaron (2016). "Adversarially Learned Inference". 
arXiv
:
1606.00704
 [
stat.ML
].




^
 
Xi Chen; Yan Duan; Rein Houthooft; John Schulman; 
Ilya Sutskever
; 
Pieter Abeel
 (2016). "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets". 
arXiv
:
1606.03657
 [
cs.LG
].




^
 
Zhirui Zhang; Shujie Liu; Mu Li; Ming Zhou; Enhong Chen (October 2018). 
"Bidirectional Generative Adversarial Networks for Neural Machine Translation"
 
(PDF)
. pp. 190–199.




^
 
Zhu, Jun-Yan; Park, Taesung; Isola, Phillip; Efros, Alexei A. (2017). "Unpaired Image-To-Image Translation Using Cycle-Consistent Adversarial Networks". pp. 2223–2232. 
arXiv
:
1703.10593
 [
cs.CV
].




^
 
Isola, Phillip; Zhu, Jun-Yan; Zhou, Tinghui; Efros, Alexei A. (2017). "Image-To-Image Translation With Conditional Adversarial Networks". pp. 1125–1134. 
arXiv
:
1611.07004
 [
cs.CV
].




^
 
Brownlee, Jason (August 22, 2019). 
"A Gentle Introduction to BigGAN the Big Generative Adversarial Network"
. 
Machine Learning Mastery
. Retrieved 
July 15,
 2022
.




^
 
Shengyu, Zhao; Zhijian, Liu; Ji, Lin; Jun-Yan, Zhu; Song, Han (2020). 
"Differentiable Augmentation for Data-Efficient GAN Training"
. 
Advances in Neural Information Processing Systems
. 
33
. 
arXiv
:
2006.10738
.




^ 
a
 
b
 
c
 
Tero, Karras; Miika, Aittala; Janne, Hellsten; Samuli, Laine; Jaakko, Lehtinen; Timo, Aila (2020). 
"Training Generative Adversarial Networks with Limited Data"
. 
Advances in Neural Information Processing Systems
. 
33
.




^
 
Shaham, Tamar Rott; Dekel, Tali; Michaeli, Tomer (October 2019). 
"SinGAN: Learning a Generative Model from a Single Natural Image"
. 
2019 IEEE/CVF International Conference on Computer Vision (ICCV)
. IEEE. pp. 4569–4579. 
arXiv
:
1905.01164
. 
doi
:
10.1109/iccv.2019.00467
. 
ISBN
 
978-1-7281-4803-8
. 
S2CID
 
145052179
.




^
 
Karras, Tero; Laine, Samuli; Aila, Timo (June 2019). 
"A Style-Based Generator Architecture for Generative Adversarial Networks"
. 
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
. IEEE. pp. 4396–4405. 
arXiv
:
1812.04948
. 
doi
:
10.1109/cvpr.2019.00453
. 
ISBN
 
978-1-7281-3293-8
. 
S2CID
 
54482423
.




^
 
Karras, Tero; Laine, Samuli; Aittala, Miika; Hellsten, Janne; Lehtinen, Jaakko; Aila, Timo (June 2020). 
"Analyzing and Improving the Image Quality of StyleGAN"
. 
2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
. IEEE. pp. 8107–8116. 
arXiv
:
1912.04958
. 
doi
:
10.1109/cvpr42600.2020.00813
. 
ISBN
 
978-1-7281-7168-5
. 
S2CID
 
209202273
.




^
 
Timo, Karras, Tero Aittala, Miika Laine, Samuli Härkönen, Erik Hellsten, Janne Lehtinen, Jaakko Aila (June 23, 2021). 
Alias-Free Generative Adversarial Networks
. 
OCLC
 
1269560084
.
{{
cite book
}}
:  CS1 maint: multiple names: authors list (
link
)




^
 
Karras, Tero; Aittala, Miika; Laine, Samuli; Härkönen, Erik; Hellsten, Janne; Lehtinen, Jaakko; Aila, Timo. 
"Alias-Free Generative Adversarial Networks (StyleGAN3)"
. 
nvlabs.github.io
. Retrieved 
July 16,
 2022
.




^
 
Caesar, Holger (March 1, 2019), 
A list of papers on Generative Adversarial (Neural) Networks: nightrome/really-awesome-gan
, retrieved 
March 2,
 2019




^
 
Li, Bonnie; François-Lavet, Vincent; Doan, Thang; Pineau, Joelle (February 14, 2021). "Domain Adversarial Reinforcement Learning". 
arXiv
:
2102.07097
 [
cs.LG
].




^
 
Wei, Jerry (July 3, 2019). 
"Generating Shoe Designs with Machine Learning"
. 
Medium
. Retrieved 
November 6,
 2019
.




^
 
Greenemeier, Larry (June 20, 2016). 
"When Will Computers Have Common Sense? Ask Facebook"
. 
Scientific American
. Retrieved 
July 31,
 2016
.




^
 
Vincent, James (March 5, 2019). 
"A never-ending stream of AI art goes up for auction"
. 
The Verge
. Retrieved 
June 13,
 2020
.




^
 
Yu, Jiahui, et al. "
Generative image inpainting with contextual attention
." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.




^
 
Wong, Ceecee (May 27, 2019). 
"The Rise of AI Supermodels"
. 
CDO Trends
.




^
 
Taif, K.; Ugail, H.; Mehmood, I. (2020). "Cast Shadow Generation Using Generative Adversarial Networks". 
Computational Science – ICCS 2020
. Lecture Notes in Computer Science. Vol. 12141. pp. 481–495. 
doi
:
10.1007/978-3-030-50426-7_36
. 
ISBN
 
978-3-030-50425-0
. 
PMC
 
7302543
.




^
 
Tang, Xiaoou; Qiao, Yu; Loy, Chen Change; Dong, Chao; Liu, Yihao; Gu, Jinjin; Wu, Shixiang; Yu, Ke; Wang, Xintao (September 1, 2018). "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks". 
arXiv
:
1809.00219
. 
Bibcode
:
2018arXiv180900219W
.




^
 
Allen, Eric Van (July 8, 2020). 
"An Infamous Zelda Creepypasta Saga Is Using Artificial Intelligence to Craft Its Finale"
. 
USgamer
. Archived from 
the original
 on November 7, 2022
. Retrieved 
November 7,
 2022
.




^
 
arcadeattack (September 28, 2020). 
"Arcade Attack Podcast – September (4 of 4) 2020 - Alex Hall (Ben Drowned) - Interview"
. 
Arcade Attack
. Retrieved 
November 7,
 2022
.




^
 
"Nvidia's AI recreates Pac-Man from scratch just by watching it being played"
. 
The Verge
. May 22, 2020.




^
 
Seung Wook Kim; Zhou, Yuhao; Philion, Jonah; Torralba, Antonio; Fidler, Sanja (2020). "Learning to Simulate Dynamic Environments with GameGAN". 
arXiv
:
2005.12126
 [
cs.CV
].




^
 
Schawinski, Kevin; Zhang, Ce; Zhang, Hantian; Fowler, Lucas; Santhanam, Gokula Krishnan (February 1, 2017). 
"Generative Adversarial Networks recover features in astrophysical images of galaxies beyond the deconvolution limit"
. 
Monthly Notices of the Royal Astronomical Society: Letters
. 
467
 (1): L110–L114. 
arXiv
:
1702.00403
. 
Bibcode
:
2017MNRAS.467L.110S
. 
doi
:
10.1093/mnrasl/slx008
. 
S2CID
 
7213940
.




^
 
Kincade, Kathy. 
"Researchers Train a Neural Network to Study Dark Matter"
. R&D Magazine.




^
 
Kincade, Kathy (May 16, 2019). 
"CosmoGAN: Training a neural network to study dark matter"
. 
Phys.org
.




^
 
"Training a neural network to study dark matter"
. 
Science Daily
. May 16, 2019.




^
 
at 06:13, Katyanna Quach 20 May 2019. 
"Cosmoboffins use neural networks to build dark matter maps the easy way"
. 
www.theregister.co.uk
. Retrieved 
May 20,
 2019
.
{{
cite web
}}
:  CS1 maint: numeric names: authors list (
link
)




^
 
Mustafa, Mustafa; Bard, Deborah; Bhimji, Wahid; Lukić, Zarija; Al-Rfou, Rami; Kratochvil, Jan M. (May 6, 2019). 
"CosmoGAN: creating high-fidelity weak lensing convergence maps using Generative Adversarial Networks"
. 
Computational Astrophysics and Cosmology
. 
6
 (1): 1. 
arXiv
:
1706.02390
. 
Bibcode
:
2019ComAC...6....1M
. 
doi
:
10.1186/s40668-019-0029-9
. 
ISSN
 
2197-7909
. 
S2CID
 
126034204
.




^
 
Paganini, Michela; de Oliveira, Luke; Nachman, Benjamin (2017). "Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis". 
Computing and Software for Big Science
. 
1
: 4. 
arXiv
:
1701.05927
. 
Bibcode
:
2017arXiv170105927D
. 
doi
:
10.1007/s41781-017-0004-6
. 
S2CID
 
88514467
.




^
 
Paganini, Michela; de Oliveira, Luke; Nachman, Benjamin (2018). "Accelerating Science with Generative Adversarial Networks: An Application to 3D Particle Showers in Multi-Layer Calorimeters". 
Physical Review Letters
. 
120
 (4): 042003. 
arXiv
:
1705.02355
. 
Bibcode
:
2018PhRvL.120d2003P
. 
doi
:
10.1103/PhysRevLett.120.042003
. 
PMID
 
29437460
. 
S2CID
 
3330974
.




^
 
Paganini, Michela; de Oliveira, Luke; Nachman, Benjamin (2018). "CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks". 
Phys. Rev. D
. 
97
 (1): 014021. 
arXiv
:
1712.10321
. 
Bibcode
:
2018PhRvD..97a4021P
. 
doi
:
10.1103/PhysRevD.97.014021
. 
S2CID
 
41265836
.




^
 
Erdmann, Martin; Glombitza, Jonas; Quast, Thorben (2019). "Precise Simulation of Electromagnetic Calorimeter Showers Using a Wasserstein Generative Adversarial Network". 
Computing and Software for Big Science
. 
3
: 4. 
arXiv
:
1807.01954
. 
doi
:
10.1007/s41781-018-0019-7
. 
S2CID
 
54216502
.




^
 
Musella, Pasquale; Pandolfi, Francesco (2018). "Fast and Accurate Simulation of Particle Detectors Using Generative Adversarial Networks". 
Computing and Software for Big Science
. 
2
: 8. 
arXiv
:
1805.00850
. 
Bibcode
:
2018arXiv180500850M
. 
doi
:
10.1007/s41781-018-0015-y
. 
S2CID
 
119474793
.




^
 
"Deep generative models for fast shower simulation in ATLAS"
. 2018.




^
 
SHiP, Collaboration (2019). "Fast simulation of muons produced at the SHiP experiment using Generative Adversarial Networks". 
Journal of Instrumentation
. 
14
 (11): 11028. 
arXiv
:
1909.04451
. 
Bibcode
:
2019JInst..14P1028A
. 
doi
:
10.1088/1748-0221/14/11/P11028
. 
S2CID
 
202542604
.




^
 
Zhavoronkov, Alex (2019). "Deep learning enables rapid identification of potent DDR1 kinase inhibitors". 
Nature Biotechnology
. 
37
 (9): 1038–1040. 
doi
:
10.1038/s41587-019-0224-x
. 
PMID
 
31477924
. 
S2CID
 
201716327
.




^
 
Barber, Gregory. 
"A Molecule Designed By AI Exhibits "Druglike" Qualities"
. 
Wired
.




^
 
Nista, Ludovico; Pitsch, Heinz; Schumann, Christoph D. K.; Bode, Mathis; Grenga, Temistocle; MacArt, Jonathan F.; Attili, Antonio (June 4, 2024). 
"Influence of adversarial training on super-resolution turbulence reconstruction"
. 
Physical Review Fluids
. 
9
 (6): 064601. 
arXiv
:
2308.16015
. 
doi
:
10.1103/PhysRevFluids.9.064601
.




^
 
Nista, L.; Schumann, C. D. K.; Grenga, T.; Attili, A.; Pitsch, H. (January 1, 2023). 
"Investigation of the generalization capability of a generative adversarial network for large eddy simulation of turbulent premixed reacting flows"
. 
Proceedings of the Combustion Institute
. 
39
 (4): 5279–5288. 
doi
:
10.1016/j.proci.2022.07.244
. 
ISSN
 
1540-7489
.




^
 
Fukami, Kai; Fukagata, Koji; Taira, Kunihiko (August 1, 2020). 
"Assessment of supervised machine learning methods for fluid flows"
. 
Theoretical and Computational Fluid Dynamics
. 
34
 (4): 497–519. 
doi
:
10.1007/s00162-020-00518-y
. 
ISSN
 
1432-2250
.




^
 
Moradi, M; Demirel, H (2024). "Alzheimer's disease classification using 3D conditional progressive GAN-and LDA-based data selection". 
Signal, Image and Video Processing
. 
18
 (2): 1847–1861. 
doi
:
10.1007/s11760-023-02878-4
.




^
 
Bisneto, Tomaz Ribeiro Viana; de Carvalho Filho, Antonio Oseas; Magalhães, Deborah Maria Vieira (February 2020). "Generative adversarial network and texture features applied to automatic glaucoma detection". 
Applied Soft Computing
. 
90
: 106165. 
doi
:
10.1016/j.asoc.2020.106165
. 
S2CID
 
214571484
.




^
 
Reconstruction of the Roman Emperors: Interview with Daniel Voshart
, November 16, 2020
, retrieved 
June 3,
 2022




^
 
Yu, Yi; Canales, Simon (2021). "Conditional LSTM-GAN for Melody Generation from Lyrics". 
ACM Transactions on Multimedia Computing, Communications, and Applications
. 
17
: 1–20. 
arXiv
:
1908.05551
. 
doi
:
10.1145/3424116
. 
ISSN
 
1551-6857
. 
S2CID
 
199668828
.




^
 
msmash (February 14, 2019). 
"
'This Person Does Not Exist' Website Uses AI To Create Realistic Yet Horrifying Faces"
. 
Slashdot
. Retrieved 
February 16,
 2019
.




^
 
Doyle, Michael (May 16, 2019). 
"John Beasley lives on Saddlehorse Drive in Evansville. Or does he?"
. Courier and Press.




^
 
Targett, Ed (May 16, 2019). "California moves closer to making deepfake pornography illegal". Computer Business Review.




^
 


Mihalcik, Carrie (October 4, 2019). 
"California laws seek to crack down on deepfakes in politics and porn"
. 
cnet.com
. 
CNET
. Retrieved 
October 13,
 2019
.




^
 
Knight, Will (August 7, 2018). 
"The Defense Department has produced the first tools for catching deepfakes"
. 
MIT Technology Review
.




^
 
Antipov, Grigory; Baccouche, Moez; Dugelay, Jean-Luc (2017). "Face Aging With Conditional Generative Adversarial Networks". 
arXiv
:
1702.01983
 [
cs.CV
].




^
 
"3D Generative Adversarial Network"
. 
3dgan.csail.mit.edu
.




^
 
Achlioptas, Panos; Diamanti, Olga; Mitliagkas, Ioannis; Guibas, Leonidas (2018). "Learning Representations and Generative Models for 3D Point Clouds". 
arXiv
:
1707.02392
 [
cs.CV
].




^
 
Vondrick, Carl; Pirsiavash, Hamed; Torralba, Antonio (2016). 
"Generating Videos with Scene Dynamics"
. 
carlvondrick.com
. 
arXiv
:
1609.02612
. 
Bibcode
:
2016arXiv160902612V
.




^
 
Kang, Yuhao; Gao, Song; Roth, Rob (2019). 
"Transferring Multiscale Map Styles Using Generative Adversarial Networks"
. 
International Journal of Cartography
. 
5
 (2–3): 115–141. 
arXiv
:
1905.02200
. 
Bibcode
:
2019IJCar...5..115K
. 
doi
:
10.1080/23729333.2019.1615729
. 
S2CID
 
146808465
.




^
 
Wijnands, Jasper; Nice, Kerry; Thompson, Jason; Zhao, Haifeng; Stevenson, Mark (2019). "Streetscape augmentation using generative adversarial networks: Insights related to health and wellbeing". 
Sustainable Cities and Society
. 
49
: 101602. 
arXiv
:
1905.06464
. 
Bibcode
:
2019SusCS..4901602W
. 
doi
:
10.1016/j.scs.2019.101602
. 
S2CID
 
155100183
.




^
 
Ukkonen, Antti; Joona, Pyry; Ruotsalo, Tuukka (2020). 
"Generating Images Instead of Retrieving Them"
. 
Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval
. pp. 1329–1338. 
doi
:
10.1145/3397271.3401129
. 
hdl
:
10138/328471
. 
ISBN
 
9781450380164
. 
S2CID
 
220730163
.




^
 
"AI can show us the ravages of climate change"
. 
MIT Technology Review
. May 16, 2019.




^
 
Christian, Jon (May 28, 2019). 
"ASTOUNDING AI GUESSES WHAT YOU LOOK LIKE BASED ON YOUR VOICE"
. Futurism.




^
 
Kulp, Patrick (May 23, 2019). 
"Samsung's AI Lab Can Create Fake Video Footage From a Single Headshot"
. 
AdWeek
.




^
 
Mohammad Navid Fekri; Ananda Mohon Ghosh; Katarina Grolinger (2020). 
"Generating Energy Data for Machine Learning with Recurrent Generative Adversarial Networks"
. 
Energies
. 
13
 (1): 130. 
doi
:
10.3390/en13010130
.




^
 
Schmidhuber, Jürgen
 (1991). "A possibility for implementing curiosity and boredom in model-building neural controllers". 
Proc. SAB'1991
. MIT Press/Bradford Books. pp. 222–227.




^
 
Schmidhuber, Jürgen
 (2020). "Generative Adversarial Networks are Special Cases of Artificial Curiosity (1990) and also Closely Related to Predictability Minimization (1991)". 
Neural Networks
. 
127
: 58–66. 
arXiv
:
1906.04493
. 
doi
:
10.1016/j.neunet.2020.04.008
. 
PMID
 
32334341
. 
S2CID
 
216056336
.




^
 
Niemitalo, Olli (February 24, 2010). 
"A method for training artificial neural networks to generate missing data within a variable context"
. 
Internet Archive (Wayback Machine)
. 
Archived
 from the original on March 12, 2012
. Retrieved 
February 22,
 2019
.




^
 
"GANs were invented in 2010?"
. 
reddit r/MachineLearning
. 2019
. Retrieved 
May 28,
 2019
.




^
 
Li, Wei; Gauci, Melvin; Gross, Roderich (July 6, 2013). "Proceeding of the fifteenth annual conference on Genetic and evolutionary computation conference - GECCO '13". 
Proceedings of the 15th Annual Conference on Genetic and Evolutionary Computation (GECCO 2013)
. Amsterdam, the Netherlands: ACM. pp. 223–230. 
doi
:
10.1145/2463372.2465801
. 
ISBN
 
9781450319638
.




^
 
Gutmann, Michael; Hyvärinen, Aapo. 
"Noise-Contrastive Estimation"
 
(PDF)
. 
International Conference on AI and Statistics
.




^
 
Abu-Khalaf, Murad; Lewis, Frank L.; Huang, Jie (July 1, 2008). "Neurodynamic Programming and Zero-Sum Games for Constrained Control Systems". 
IEEE Transactions on Neural Networks
. 
19
 (7): 1243–1252. 
doi
:
10.1109/TNN.2008.2000204
. 
S2CID
 
15680448
.




^
 
Abu-Khalaf, Murad; Lewis, Frank L.; Huang, Jie (December 1, 2006). "Policy Iterations on the Hamilton–Jacobi–Isaacs Equation for 
H
∞
 State Feedback Control With Input Saturation". 
IEEE Transactions on Automatic Control
. 
doi
:
10.1109/TAC.2006.884959
. 
S2CID
 
1338976
.




^
 
Sajjadi, Mehdi S. M.; Schölkopf, Bernhard; Hirsch, Michael (December 23, 2016). "EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis". 
arXiv
:
1612.07919
 [
cs.CV
].




^
 
"This Person Does Not Exist: Neither Will Anything Eventually with AI"
. March 20, 2019.




^
 
"ARTificial Intelligence enters the History of Art"
. December 28, 2018.




^
 
Tom Février (February 17, 2019). 
"Le scandale de l'intelligence ARTificielle"
.




^
 
"StyleGAN: Official TensorFlow Implementation"
. March 2, 2019 – via GitHub.




^
 
Paez, Danny (February 13, 2019). 
"This Person Does Not Exist Is the Best One-Off Website of 2019"
. Retrieved 
February 16,
 2019
.




^
 
Beschizza, Rob (February 15, 2019). 
"This Person Does Not Exist"
. 
Boing-Boing
. Retrieved 
February 16,
 2019
.




^
 
Horev, Rani (December 26, 2018). 
"Style-based GANs – Generating and Tuning Realistic Artificial Faces"
. 
Lyrn.AI
. Archived from 
the original
 on November 5, 2020
. Retrieved 
February 16,
 2019
.




^
 
Elgammal, Ahmed; Liu, Bingchen; Elhoseiny, Mohamed; Mazzone, Marian (2017). "CAN: Creative Adversarial Networks, Generating "Art" by Learning About Styles and Deviating from Style Norms". 
arXiv
:
1706.07068
 [
cs.AI
].




^
 
Cohn, Gabe (October 25, 2018). 
"AI Art at Christie's Sells for $432,500"
. 
The New York Times
.




^
 
Mazzone, Marian; Ahmed Elgammal (February 21, 2019). 
"Art, Creativity, and the Potential of Artificial Intelligence"
. 
Arts
. 
8
: 26. 
doi
:
10.3390/arts8010026
.






External links
[
edit
]




Art portal


Knight, Will. 
"5 Big Predictions for Artificial Intelligence in 2017"
. 
MIT Technology Review
. Retrieved 
January 5,
 2017
.


Karras, Tero; Laine, Samuli; Aila, Timo (2018). "A Style-Based Generator Architecture for Generative Adversarial Networks". 
arXiv
:
1812.04948
 [
cs.NE
].


This Person Does Not Exist
 –  photorealistic images of people who do not exist, generated by 
StyleGAN


This Cat Does Not Exist
 
Archived
 March 5, 2019, at the 
Wayback Machine
 –  photorealistic images of cats who do not exist, generated by 
StyleGAN


Wang, Zhengwei; She, Qi; Ward, Tomas E. (2019). "Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy". 
arXiv
:
1906.01529
 [
cs.LG
].


v
t
e
Generative AI
Concepts


Autoencoder


Deep learning


Generative adversarial network


Generative pre-trained transformer


Large language model


Neural network


Prompt engineering


RAG


RLHF


Self-supervised learning


Transformer


Variational autoencoder


Vision transformer


Word embedding


Models
Text


Claude


Gemini


GPT-2


GPT-3


GPT-4


LLaMA


Images


DALL-E


Midjourney


Stable Diffusion


Videos


Sora


Musics


Suno AI


Udio


Companies


Anthropic


Google DeepMind


Hugging Face


OpenAI


Meta AI


Mistral AI


 
Category
v
t
e
Differentiable computing
General


Differentiable programming


Information geometry


Statistical manifold


Automatic differentiation


Neuromorphic engineering


Pattern recognition


Tensor calculus


Computational learning theory


Inductive bias


Concepts


Gradient descent


SGD


Clustering


Regression


Overfitting


Hallucination


Adversary


Attention


Convolution


Loss functions


Backpropagation


Batchnorm


Activation


Softmax


Sigmoid


Rectifier


Regularization


Datasets


Augmentation


Diffusion


Autoregression


Applications


Machine learning


In-context learning


Artificial neural network


Deep learning


Scientific computing


Artificial Intelligence


Language model


Large language model


Hardware


IPU


TPU


VPU


Memristor


SpiNNaker


Software libraries


TensorFlow


PyTorch


Keras


Theano


JAX


Flux.jl


MindSpore


Implementations
Audio–visual


AlexNet


WaveNet


Human image synthesis


HWR


OCR


Speech synthesis


Speech recognition


Facial recognition


AlphaFold


Text-to-image models


DALL-E


Midjourney


Stable Diffusion


Text-to-video models


Sora


VideoPoet


Whisper


Verbal


Word2vec


Seq2seq


BERT


Gemini


LaMDA


Bard


NMT


Project Debater


IBM Watson


IBM Watsonx


Granite


GPT-1


GPT-2


GPT-3


GPT-4


ChatGPT


GPT-J


Chinchilla AI


PaLM


BLOOM


LLaMA


PanGu-Σ


Decisional


AlphaGo


AlphaZero


Q-learning


SARSA


OpenAI Five


Self-driving car


MuZero


Action selection


Auto-GPT


Robot control


People


Yoshua Bengio


Alex Graves


Ian Goodfellow


Stephen Grossberg


Demis Hassabis


Geoffrey Hinton


Yann LeCun


Fei-Fei Li


Andrew Ng


Jürgen Schmidhuber


David Silver


Ilya Sutskever


Organizations


Anthropic


EleutherAI


Google DeepMind


Hugging Face


OpenAI


Meta AI


Mila


MIT CSAIL


Huawei


Architectures


Neural Turing machine


Differentiable neural computer


Transformer


Recurrent neural network (RNN)


Long short-term memory (LSTM)


Gated recurrent unit (GRU)


Echo state network


Multilayer perceptron (MLP)


Convolutional neural network


Residual neural network


Mamba


Autoencoder


Variational autoencoder (VAE)


Generative adversarial network (GAN)


Graph neural network




 Portals

Computer programming


Technology


 Categories

Artificial neural networks


Machine learning












Retrieved from "
https://en.wikipedia.org/w/index.php?title=Generative_adversarial_network&oldid=1241071646
"


Categories
: 
Neural network architectures
Cognitive science
Unsupervised learning
Generative artificial intelligence
Hidden categories: 
CS1 errors: missing periodical
CS1 maint: multiple names: authors list
CS1 maint: numeric names: authors list
Articles with short description
Short description is different from Wikidata
Use mdy dates from April 2021
All articles with unsourced statements
Articles with unsourced statements from February 2018
Articles with unsourced statements from January 2020
Webarchive template wayback links




Read more
## Adversarial machine learning













Toggle the table of contents
















Adversarial machine learning








9 languages










العربية
Čeština
Ελληνικά
Español
فارسی
한국어
Italiano
Português
粵語




Edit links
























Article
Talk












English




































Read
Edit
View history
















Tools












Tools


move to sidebar


hide







		Actions
	






Read
Edit
View history











		General
	






What links here
Related changes
Upload file
Special pages
Permanent link
Page information
Cite this page
Get shortened URL
Download QR code
Wikidata item











		Print/export
	






Download as PDF
Printable version











		In other projects
	
















































Appearance


move to sidebar


hide






















From Wikipedia, the free encyclopedia






Research field that lies at the intersection of machine learning and computer security


Not to be confused with 
Generative adversarial network
.




Part of a series on
Machine learning
and 
data mining


Paradigms


Supervised learning


Unsupervised learning


Semi-supervised learning


Self-supervised learning


Reinforcement learning


Meta-learning


Online learning


Batch learning


Curriculum learning


Rule-based learning


Neuro-symbolic AI


Neuromorphic engineering


Quantum machine learning




Problems


Classification


Generative modeling


Regression


Clustering


Dimensionality reduction


Density estimation


Anomaly detection


Data cleaning


AutoML


Association rules


Semantic analysis


Structured prediction


Feature engineering


Feature learning


Learning to rank


Grammar induction


Ontology learning


Multimodal learning




Supervised learning
(
classification
 • 
regression
)
 


Apprenticeship learning


Decision trees


Ensembles


Bagging


Boosting


Random forest


k
-NN


Linear regression


Naive Bayes


Artificial neural networks


Logistic regression


Perceptron


Relevance vector machine (RVM)


Support vector machine (SVM)




Clustering


BIRCH


CURE


Hierarchical


k
-means


Fuzzy


Expectation–maximization (EM)


DBSCAN


OPTICS


Mean shift




Dimensionality reduction


Factor analysis


CCA


ICA


LDA


NMF


PCA


PGD


t-SNE


SDL




Structured prediction


Graphical models


Bayes net


Conditional random field


Hidden Markov




Anomaly detection


RANSAC


k
-NN


Local outlier factor


Isolation forest




Artificial neural network


Autoencoder


Deep learning


Feedforward neural network


Recurrent neural network


LSTM


GRU


ESN


reservoir computing


Boltzmann machine


Restricted


GAN


Diffusion model


SOM


Convolutional neural network


U-Net


LeNet


AlexNet


DeepDream


Neural radiance field


Transformer


Vision


Mamba


Spiking neural network


Memtransistor


Electrochemical RAM
 (ECRAM)




Reinforcement learning


Q-learning


SARSA


Temporal difference (TD)


Multi-agent


Self-play




Learning with humans


Active learning


Crowdsourcing


Human-in-the-loop


RLHF




Model diagnostics


Coefficient of determination


Confusion matrix


Learning curve


ROC curve




Mathematical foundations


Kernel machines


Bias–variance tradeoff


Computational learning theory


Empirical risk minimization


Occam learning


PAC learning


Statistical learning


VC theory




Journals and conferences


ECML PKDD


NeurIPS


ICML


ICLR


IJCAI


ML


JMLR




Related articles


Glossary of artificial intelligence


List of datasets for machine-learning research


List of datasets in computer vision and image processing


Outline of machine learning


v
t
e


Adversarial machine learning
 is the study of the attacks on 
machine learning
 algorithms, and of the defenses against such attacks.
[
1
]
 A survey from May 2020 exposes the fact that practitioners report a dire need for better protecting machine learning systems in industrial applications.
[
2
]


Most machine learning techniques are mostly designed to work on specific problem sets, under the assumption that the training and test data are generated from the same statistical distribution (
IID
). However, this assumption is often dangerously violated in practical high-stake applications, where users may intentionally supply fabricated data that violates the statistical assumption.

Most common attacks in adversarial machine learning include 
evasion attacks
,
[
3
]
 
data poisoning attacks
,
[
4
]
 
Byzantine attacks
[
5
]
 and model extraction.
[
6
]






History
[
edit
]


At the MIT Spam Conference in January 2004, 
John Graham-Cumming
 showed that a machine-learning spam filter could be used to defeat another machine-learning spam filter by automatically learning which words to add to a spam email to get the email classified as not spam.
[
7
]


In 2004, Nilesh Dalvi and others noted that 
linear classifiers
 used in 
spam filters
 could be defeated by simple "
evasion
 attacks" as spammers inserted "good words" into their spam emails. (Around 2007, some spammers added random noise to fuzz words within "image spam" in order to defeat 
OCR
-based filters.) In 2006, Marco Barreno and others published "Can Machine Learning Be Secure?", outlining a broad taxonomy of attacks. As late as 2013 many researchers continued to hope that non-linear classifiers (such as 
support vector machines
 and 
neural networks
) might be robust to adversaries, until Battista Biggio and others demonstrated the first gradient-based attacks on such machine-learning models (2012
[
8
]
–2013
[
9
]
). In 2012, 
deep neural networks
 began to dominate computer vision problems; starting in 2014, Christian Szegedy and others demonstrated that deep neural networks could be fooled by adversaries, again using a gradient-based attack to craft adversarial perturbations.
[
10
]
[
11
]


Recently, it was observed that adversarial attacks are harder to produce in the practical world due to the different environmental constraints that cancel out the effect of noise.
[
12
]
[
13
]
 For example, any small rotation or slight illumination on an adversarial image can destroy the adversariality. In addition, researchers such as Google Brain's Nicholas Frosst point out that it is much easier to make self-driving cars
[
14
]
 miss stop signs by physically removing the sign itself, rather than creating adversarial examples.
[
15
]
 Frosst also believes that the adversarial machine learning community incorrectly assumes models trained on a certain data distribution will also perform well on a completely different data distribution. He suggests that a new approach to machine learning should be explored, and is currently working on a unique neural network that has characteristics more similar to human perception than state-of-the-art approaches.
[
15
]


While adversarial machine learning continues to be heavily rooted in academia, large tech companies such as Google, Microsoft, and IBM have begun curating documentation and open source code bases to allow others to concretely assess the 
robustness
 of machine learning models and minimize the risk of adversarial attacks.
[
16
]
[
17
]
[
18
]




Examples
[
edit
]


Examples include attacks in 
spam filtering
, where spam messages are obfuscated through the misspelling of "bad" words or the insertion of "good" words;
[
19
]
[
20
]
 attacks in 
computer security
, such as obfuscating malware code within 
network packets
 or modifying the characteristics of a 
network flow
 to mislead intrusion detection;
[
21
]
[
22
]
 attacks in biometric recognition where fake biometric traits may be exploited to impersonate a legitimate user;
[
23
]
 or to compromise users' template galleries that adapt to updated traits over time.

Researchers showed that by changing only one-pixel it was possible to fool deep learning algorithms.
[
24
]
 Others 
3-D printed
 a toy turtle with a texture engineered to make Google's object detection 
AI
 classify it as a rifle regardless of the angle from which the turtle was viewed.
[
25
]
 Creating the turtle required only low-cost commercially available 3-D printing technology.
[
26
]


A machine-tweaked image of a dog was shown to look like a cat to both computers and humans.
[
27
]
 A 2019 study reported that humans can guess how machines will classify adversarial images.
[
28
]
 Researchers discovered methods for perturbing the appearance of a stop sign such that an autonomous vehicle classified it as a merge or speed limit sign.
[
14
]
[
29
]


McAfee
 attacked 
Tesla
's former 
Mobileye
 system, fooling it into driving 50 mph over the speed limit, simply by adding a two-inch strip of black tape to a speed limit sign.
[
30
]
[
31
]


Adversarial patterns on glasses or clothing designed to deceive facial-recognition systems or license-plate readers, have led to a niche industry of "stealth streetwear".
[
32
]


An adversarial attack on a neural network can allow an attacker to inject algorithms into the target system.
[
33
]
 Researchers can also create adversarial audio inputs to disguise commands to intelligent assistants in benign-seeming audio;
[
34
]
 a parallel literature explores human perception of such stimuli.
[
35
]
[
36
]


Clustering algorithms are used in security applications. Malware and 
computer virus
 analysis aims to identify malware families, and to generate specific detection signatures.
[
37
]
[
38
]




Attack modalities
[
edit
]


Taxonomy
[
edit
]


Attacks against (supervised) machine learning algorithms have been categorized along three primary axes:
[
39
]
 influence on the classifier, the security violation and their specificity.



Classifier influence: An attack can influence the classifier by disrupting the classification phase. This may be preceded by an exploration phase to identify vulnerabilities. The attacker's capabilities might be restricted by the presence of data manipulation constraints.
[
40
]


Security violation: An attack can supply malicious data that gets classified as legitimate. Malicious data supplied during training can cause legitimate data to be rejected after training.


Specificity: A targeted attack attempts to allow a specific intrusion/disruption. Alternatively, an indiscriminate attack creates general mayhem.


This taxonomy has been extended into a more comprehensive threat model that allows explicit assumptions about the adversary's goal, knowledge of the attacked system, capability of manipulating the input data/system components, and on attack strategy.
[
41
]
[
42
]
 This taxonomy has further been extended to include dimensions for defense strategies against adversarial attacks.
[
43
]




Strategies
[
edit
]


Below are some of the most commonly encountered attack scenarios.



Data poisoning
[
edit
]


Poisoning consists of contaminating the training dataset with data designed to increase errors in the output. Given that learning algorithms are shaped by their training datasets, poisoning can effectively reprogram algorithms with potentially malicious intent. Concerns have been raised especially for user-generated training data, e.g. for content recommendation or natural language models. The ubiquity of fake accounts offers many opportunities for poisoning. Facebook reportedly removes around 7 billion fake accounts per year.
[
44
]
[
45
]
 Poisoning has been reported as the leading concern for industrial applications.
[
2
]


On social medias, 
disinformation campaigns
 attempt to bias recommendation and moderation algorithms, to push certain content over others.

A particular case of data poisoning is the 
backdoor
 attack,
[
46
]
 which aims to teach a specific behavior for inputs with a given trigger, e.g. a small defect on images, sounds, videos or texts.

For instance, 
intrusion detection systems
 are often trained using collected data. An attacker may poison this data by injecting malicious samples during operation that subsequently disrupt retraining.
[
41
]
[
42
]
[
39
]
[
47
]
[
48
]


Data poisoning techniques can also be applied to 
text-to-image models
 to alter their output.
[
49
]


Data poisoning can also happen unintentionally through 
model collapse
, where models are trained on synthetic data.
[
50
]




Byzantine attacks
[
edit
]


As machine learning is scaled, it often relies on multiple computing machines. In 
federated learning
, for instance, edge devices collaborate with a central server, typically by sending gradients or model parameters. However, some of these devices may deviate from their expected behavior, e.g. to harm the central server's model
[
51
]
 or to bias algorithms towards certain behaviors (e.g., amplifying the recommendation of disinformation content). On the other hand, if the training is performed on a single machine, then the model is very vulnerable to a failure of the machine, or an attack on the machine; the machine is a 
single point of failure
.
[
52
]
 In fact, the machine owner may themselves insert provably undetectable 
backdoors
.
[
53
]


The current leading solutions to make (distributed) learning algorithms provably resilient to a minority of malicious (a.k.a. 
Byzantine
) participants are based on robust gradient aggregation rules.
[
54
]
[
55
]
[
56
]
[
57
]
[
58
]
[
59
]
 The robust aggregation rules do not always work especially when the data across participants has a non-iid distribution. Nevertheless, in the context of heterogeneous honest participants, such as users with different consumption habits for recommendation algorithms or writing styles for language models, there are provable impossibility theorems on what any robust learning algorithm can guarantee.
[
5
]
[
60
]




Evasion
[
edit
]


Evasion attacks
[
9
]
[
41
]
[
42
]
[
61
]
 consist of exploiting the imperfection of a trained model. For instance, spammers and hackers often attempt to evade detection by obfuscating the content of spam emails and 
malware
. Samples are modified to evade detection; that is, to be classified as legitimate. This does not involve influence over the training data. A clear example of evasion is 
image-based spam
 in which the spam content is embedded within an attached image to evade textual analysis by anti-spam filters. Another example of evasion is given by spoofing attacks against biometric verification systems.
[
23
]


Evasion attacks can be generally split into two different categories: 
black box attacks
 and 
white box attacks
.
[
17
]




Model extraction
[
edit
]


Model extraction involves an adversary probing a black box machine learning system in order to extract the data it was trained on.
[
62
]
[
63
]
  This can cause issues when either the training data or the model itself is sensitive and confidential. For example, model extraction could be used to extract a proprietary stock trading model which the adversary could then use for their own financial benefit.

In the extreme case, model extraction can lead to 
model stealing
, which corresponds to extracting a sufficient amount of data from the model to enable the complete reconstruction of the model.

On the other hand, membership inference is a targeted model extraction attack, which infers the owner of a data point, often by leveraging the 
overfitting
 resulting from poor machine learning practices.
[
64
]
 Concerningly, this is sometimes achievable even without knowledge or access to a target model's parameters, raising security concerns for models trained on sensitive data, including but not limited to medical records and/or personally identifiable information. With the emergence of 
transfer learning
 and public accessibility of many state of the art machine learning models, tech companies are increasingly drawn to create models based on public ones, giving attackers freely accessible information to the structure and type of model being used.
[
64
]




Categories
[
edit
]


Adversarial deep reinforcement learning
[
edit
]


Adversarial deep reinforcement learning is an active area of research in reinforcement learning focusing on vulnerabilities of learned policies. In this research area, some studies initially showed that reinforcement learning policies are susceptible to imperceptible adversarial manipulations.
[
65
]
[
66
]
 While some methods have been proposed to overcome these susceptibilities, in the most recent studies it has been shown that these proposed solutions are far from providing an accurate representation of current vulnerabilities of deep reinforcement learning policies.
[
67
]




Adversarial natural language processing
[
edit
]


Adversarial attacks on 
speech recognition
 have been introduced for speech-to-text applications, in particular for Mozilla's implementation of DeepSpeech.
[
68
]




Adversarial attacks and training in linear models
[
edit
]


There is a growing literature about adversarial attacks in 
linear models. Indeed, since the seminal work from Goodfellow at al. 
[
69
]
 studying these models in linear models has been an important tool to understand how adversarial attacks affect machine learning models. 
The analysis of these models is simplified because the computation of adversarial attacks can be simplified in linear regression and classification problems. Moreover, adversarial training is convex in this case. 
[
70
]


Linear models allow for analytical analysis while still reproducing phenomena observed in state-of-the-art models.
One prime example of that is how this model can be used to explain the trade-off between robustness and accuracy.
[
71
]
 
Diverse work indeed provides analysis of adversarial attacks in linear models, including asymptotic analysis for  classification 
[
72
]
 and for linear regression.
[
73
]
[
74
]
 And, finite-sample analysis based on Rademacher complexity.
[
75
]






Specific attack types
[
edit
]


There are a large variety of different adversarial attacks that can be used against machine learning systems. Many of these work on both 
deep learning
 systems as well as traditional machine learning models such as 
SVMs
[
8
]
 and  
linear regression
.
[
76
]
 A high level sample of these attack types include:



Adversarial Examples
[
77
]


Trojan Attacks / Backdoor Attacks
[
78
]


Model Inversion
[
79
]


Membership Inference
[
80
]


Adversarial examples
[
edit
]


An adversarial example refers to specially crafted input that is designed to look "normal" to humans but causes misclassification to a machine learning model.  Often, a form of specially designed "noise"  is used to elicit the misclassifications. Below are some current techniques for generating adversarial examples in the literature (by no means an exhaustive list).



Gradient-based evasion attack
[
9
]


Fast Gradient Sign Method (FGSM)
[
81
]


Projected Gradient Descent (PGD)
[
82
]


Carlini and Wagner (C&W) attack
[
83
]


Adversarial patch attack
[
84
]


Black box attacks
[
edit
]


Black box attacks in adversarial machine learning assume that the adversary can only get outputs for provided inputs and has no knowledge of the model structure or parameters.
[
17
]
[
85
]
 In this case, the adversarial example is generated either using a model created from scratch, or without any model at all (excluding the ability to query the original model). In either case, the objective of these attacks is to create adversarial examples that are able to transfer to the black box model in question.
[
86
]




Square Attack
[
edit
]


The Square Attack was introduced in 2020 as a black box evasion adversarial attack based on querying classification scores without the need of gradient information.
[
87
]
 As a score based black box attack, this adversarial approach is able to query probability distributions across model output classes, but has no other access to the model itself. According to the paper's authors, the proposed Square Attack required fewer queries than when compared to state-of-the-art score-based black box attacks at the time.
[
87
]


To describe the function objective, the attack defines the classifier as 








f


:


[


0


,


1




]




d






→






R






K










{\textstyle f:[0,1]^{d}\rightarrow \mathbb {R} ^{K}}




, with 








d






{\textstyle d}




 representing the dimensions of the input and 








K






{\textstyle K}




 as the total number of output classes. 










f




k






(


x


)






{\textstyle f_{k}(x)}




 returns the score (or a probability between 0 and 1) that the input 








x






{\textstyle x}




 belongs to class 








k






{\textstyle k}




, which allows the classifier's class output for any input 








x






{\textstyle x}




 to be defined as 












argmax






k


=


1


,


.


.


.


,


K








f




k






(


x


)






{\textstyle {\text{argmax}}_{k=1,...,K}f_{k}(x)}




. The goal of this attack is as follows:
[
87
]














argmax






k


=


1


,


.


.


.


,


K








f




k






(








x


^








)


≠


y


,




|






|










x


^








−


x




|








|






p






≤


ϵ




 and 










x


^








∈


[


0


,


1




]




d










{\displaystyle {\text{argmax}}_{k=1,...,K}f_{k}({\hat {x}})\neq y,||{\hat {x}}-x||_{p}\leq \epsilon {\text{ and }}{\hat {x}}\in [0,1]^{d}}






In other words, finding some perturbed adversarial example 














x


^












{\textstyle {\hat {x}}}




 such that the classifier incorrectly classifies it to some other class under the constraint that 














x


^












{\textstyle {\hat {x}}}




 and 








x






{\textstyle x}




 are similar. The paper then defines 
loss
 








L






{\textstyle L}




 as 








L


(


f


(








x


^








)


,


y


)


=




f




y






(








x


^








)


−




max




k


≠


y








f




k






(








x


^








)






{\textstyle L(f({\hat {x}}),y)=f_{y}({\hat {x}})-\max _{k\neq y}f_{k}({\hat {x}})}




 and proposes the solution to finding adversarial example 














x


^












{\textstyle {\hat {x}}}




 as solving the below 
constrained optimization problem
:
[
87
]












min










x


^








∈


[


0


,


1




]




d










L


(


f


(








x


^








)


,


y


)


,




 s.t. 






|






|










x


^








−


x




|








|






p






≤


ϵ






{\displaystyle \min _{{\hat {x}}\in [0,1]^{d}}L(f({\hat {x}}),y),{\text{ s.t. }}||{\hat {x}}-x||_{p}\leq \epsilon }






The result in theory is an adversarial example that is highly confident in the incorrect class but is also very similar to the original image. To find such example, Square Attack utilizes the iterative 
random search
 technique to randomly perturb the image in hopes of improving the objective function. In each step, the algorithm perturbs only a small square section of pixels, hence the name Square Attack, which terminates as soon as an adversarial example is found in order to improve query efficiency. Finally, since the attack algorithm uses scores and not gradient information, the authors of the paper indicate that this approach is not affected by gradient masking, a common technique formerly used to prevent evasion attacks.
[
87
]




HopSkipJump Attack
[
edit
]


This black box attack was also proposed as a query efficient attack, but one that relies solely on access to any input's predicted output class. In other words, the HopSkipJump attack does not require the ability to calculate gradients or access to score values like the Square Attack, and will require just the model's class prediction output (for any given input). The proposed attack is split into two different settings, targeted and untargeted, but both are built from the general idea of adding minimal perturbations that leads to a different model output. In the targeted setting, the goal is to cause the model to misclassify the perturbed image to a specific target label (that is not the original label). In the untargeted setting, the goal is to cause the model to misclassify the perturbed image to any label that is not the original label. The attack objectives for both are as follows where 








x






{\textstyle x}




 is the original image, 










x




′










{\textstyle x^{\prime }}




 is the adversarial image, 








d






{\textstyle d}




 is a distance function between images, 










c




∗










{\textstyle c^{*}}




 is the target label, and 








C






{\textstyle C}




 is the model's classification class label function:
[
88
]














Targeted:








min






x




′










d


(




x




′






,


x


)




 subject to 




C


(




x




′






)


=




c




∗










{\displaystyle {\textbf {Targeted:}}\min _{x^{\prime }}d(x^{\prime },x){\text{ subject to }}C(x^{\prime })=c^{*}}


















Untargeted:








min






x




′










d


(




x




′






,


x


)




 subject to 




C


(




x




′






)


≠


C


(


x


)






{\displaystyle {\textbf {Untargeted:}}\min _{x^{\prime }}d(x^{\prime },x){\text{ subject to }}C(x^{\prime })\neq C(x)}






To solve this problem, the attack proposes the following boundary function 








S






{\textstyle S}




 for both the untargeted and targeted setting:
[
88
]










S


(




x




′






)


:=






{










max




c


≠


C


(


x


)








F


(




x




′








)




c








−


F


(




x




′








)




C


(


x


)






,








(Untargeted)












F


(




x




′








)






c




∗










−




max




c


≠




c




∗












F


(




x




′








)




c








,








(Targeted)




















{\displaystyle S(x^{\prime }):={\begin{cases}\max _{c\neq C(x)}{F(x^{\prime })_{c}}-F(x^{\prime })_{C(x)},&{\text{(Untargeted)}}\\F(x^{\prime })_{c^{*}}-\max _{c\neq c^{*}}{F(x^{\prime })_{c}},&{\text{(Targeted)}}\end{cases}}}






This can be further simplified to better visualize the boundary between different potential adversarial examples:
[
88
]










S


(




x




′






)


>


0




⟺








{








a


r


g


m


a




x




c






F


(




x




′






)


≠


C


(


x


)


,








(Untargeted)












a


r


g


m


a




x




c






F


(




x




′






)


=




c




∗






,








(Targeted)




















{\displaystyle S(x^{\prime })>0\iff {\begin{cases}argmax_{c}F(x^{\prime })\neq C(x),&{\text{(Untargeted)}}\\argmax_{c}F(x^{\prime })=c^{*},&{\text{(Targeted)}}\end{cases}}}






With this boundary function, the attack then follows an iterative algorithm to find adversarial examples 










x




′










{\textstyle x^{\prime }}




 for a given image 








x






{\textstyle x}




 that satisfies the attack objectives.



Initialize 








x






{\textstyle x}




 to some point where 








S


(


x


)


>


0






{\textstyle S(x)>0}






Iterate below

Boundary search


Gradient update

Compute the gradient


Find the step size


Boundary search uses a modified 
binary search
 to find the point in which the boundary (as defined by 








S






{\textstyle S}




) intersects with the line between 








x






{\textstyle x}




 and 










x




′










{\textstyle x^{\prime }}




. The next step involves calculating the gradient for 








x






{\textstyle x}




, and update the original 








x






{\textstyle x}




 using this gradient and a pre-chosen step size. HopSkipJump authors prove that this iterative algorithm will converge, leading 








x






{\textstyle x}




 to a point right along the boundary that is very close in distance to the original image.
[
88
]


However, since HopSkipJump is a proposed black box attack and the iterative algorithm above requires the calculation of a gradient in the second iterative step (which black box attacks do not have access to), the authors propose a solution to gradient calculation that requires only the model's output predictions alone.
[
88
]
 By generating many random vectors in all directions, denoted as 










u




b










{\textstyle u_{b}}




, an approximation of the gradient can be calculated using the average of these random vectors weighted by the sign of the boundary function on the image 










x




′






+




δ






u




b














{\textstyle x^{\prime }+\delta _{u_{b}}}




, where 










δ






u




b














{\textstyle \delta _{u_{b}}}




 is the size of the random vector perturbation:
[
88
]










∇


S


(




x




′






,


δ


)


≈






1


B








∑




b


=


1






B






ϕ


(




x




′






+




δ






u




b










)




u




b










{\displaystyle \nabla S(x^{\prime },\delta )\approx {\frac {1}{B}}\sum _{b=1}^{B}\phi (x^{\prime }+\delta _{u_{b}})u_{b}}






The result of the equation above gives a close approximation of the gradient required in step 2 of the iterative algorithm, completing HopSkipJump as a black box attack.
[
89
]
[
90
]
[
88
]




White box attacks
[
edit
]


White box attacks assumes that the adversary has access to model parameters on top of being able to get labels for provided inputs.
[
86
]




Fast gradient sign method
[
edit
]


One of the first proposed attacks for generating adversarial examples was proposed by Google researchers 
Ian J. Goodfellow
, Jonathon Shlens, and Christian Szegedy.
[
91
]
 The attack was called fast gradient sign method (FGSM), and it consists of adding a linear amount of in-perceivable noise to the image and causing a model to incorrectly classify it. This noise is calculated by multiplying the sign of the gradient with respect to the image we want to perturb by a small constant epsilon. As epsilon increases, the model is more likely to be fooled, but the perturbations become easier to identify as well. Shown below is the equation to generate an adversarial example where 








x






{\textstyle x}




 is the original image, 








ϵ






{\textstyle \epsilon }




 is a very small number, 










Δ




x










{\textstyle \Delta _{x}}




 is the gradient function, 








J






{\textstyle J}




 is the loss function, 








θ






{\textstyle \theta }




 is the model weights, and 








y






{\textstyle y}




 is the true label.
[
92
]










a


d




v




x






=


x


+


ϵ


⋅


s


i


g


n


(




Δ




x






J


(


θ


,


x


,


y


)


)






{\displaystyle adv_{x}=x+\epsilon \cdot sign(\Delta _{x}J(\theta ,x,y))}






One important property of this equation is that the gradient is calculated with respect to the input image since the goal is to generate an image that maximizes the loss for the original image of true label 








y






{\textstyle y}




. In traditional 
gradient descent
 (for model training), the gradient is used to update the weights of the model since the goal is to minimize the loss for the model on a ground truth dataset. The Fast Gradient Sign Method was proposed as a fast way to generate adversarial examples to evade the model, based on the hypothesis that neural networks cannot resist even linear amounts of perturbation to the input.
[
93
]
[
92
]
[
91
]
 FGSM has shown to be effective in adversarial attacks for image classification and skeletal action recognition.
[
94
]




Carlini & Wagner (C&W)
[
edit
]


In an effort to analyze existing adversarial attacks and defenses, researchers at the University of California, Berkeley, 
Nicholas Carlini
 and 
David Wagner
 in 2016 propose a faster and more robust method to generate adversarial examples.
[
95
]


The attack proposed by Carlini and Wagner begins with trying to solve a difficult non-linear optimization equation:
[
63
]










min


(




|






|




δ




|








|






p






)




 subject to 




C


(


x


+


δ


)


=


t


,


x


+


δ


∈


[


0


,


1




]




n










{\displaystyle \min(||\delta ||_{p}){\text{ subject to }}C(x+\delta )=t,x+\delta \in [0,1]^{n}}






Here the objective is to minimize the noise (








δ






{\textstyle \delta }




), added to the original input 








x






{\textstyle x}




, such that the machine learning algorithm (








C






{\textstyle C}




) predicts the original input with delta (or 








x


+


δ






{\textstyle x+\delta }




) as some other class 








t






{\textstyle t}




. However instead of directly the above equation, Carlini and Wagner propose using a new function 








f






{\textstyle f}




 such that:
[
63
]










C


(


x


+


δ


)


=


t




⟺




f


(


x


+


δ


)


≤


0






{\displaystyle C(x+\delta )=t\iff f(x+\delta )\leq 0}






This condenses the first equation to the problem below:
[
63
]










min


(




|






|




δ




|








|






p






)




 subject to 




f


(


x


+


δ


)


≤


0


,


x


+


δ


∈


[


0


,


1




]




n










{\displaystyle \min(||\delta ||_{p}){\text{ subject to }}f(x+\delta )\leq 0,x+\delta \in [0,1]^{n}}






and even more to the equation below:
[
63
]










min


(




|






|




δ




|








|






p






+


c


⋅


f


(


x


+


δ


)


)


,


x


+


δ


∈


[


0


,


1




]




n










{\displaystyle \min(||\delta ||_{p}+c\cdot f(x+\delta )),x+\delta \in [0,1]^{n}}






Carlini and Wagner then propose the use of the below function in place of 








f






{\textstyle f}




 using 








Z






{\textstyle Z}




, a function that determines class probabilities for given input 








x






{\textstyle x}




. When substituted in, this equation can be thought of as finding a target class that is more confident than the next likeliest class by some constant amount:
[
63
]










f


(


x


)


=


(


[




max




i


≠


t






Z


(


x




)




i






]


−


Z


(


x




)




t








)




+










{\displaystyle f(x)=([\max _{i\neq t}Z(x)_{i}]-Z(x)_{t})^{+}}






When solved using gradient descent, this equation is able to produce stronger adversarial examples when compared to fast gradient sign method that is also able to bypass defensive distillation, a defense that was once proposed to be effective against adversarial examples.
[
96
]
[
97
]
[
95
]
[
63
]




Defenses
[
edit
]


Conceptual representation of the proactive arms race
[
42
]
[
38
]


Researchers have proposed a multi-step approach to protecting machine learning.
[
11
]




Threat modeling – Formalize the attackers goals and capabilities with respect to the target system.


Attack simulation – Formalize the optimization problem the attacker tries to solve according to possible attack strategies.


Attack impact evaluation


Countermeasure design


Noise detection (For evasion based attack)
[
98
]


Information laundering – Alter the information received by adversaries (for model stealing attacks)
[
63
]


Mechanisms
[
edit
]


A number of defense mechanisms against evasion, poisoning, and privacy attacks have been proposed, including:



Secure learning algorithms
[
20
]
[
99
]
[
100
]


Byzantine-resilient algorithms
[
54
]
[
5
]


Multiple classifier systems
[
19
]
[
101
]


AI-written algorithms.
[
33
]


AIs that explore the training environment; for example, in image recognition, actively navigating a 3D environment rather than passively scanning a fixed set of 2D images.
[
33
]


Privacy-preserving learning
[
42
]
[
102
]


Ladder algorithm for 
Kaggle
-style competitions


Game theoretic models
[
103
]
[
104
]
[
105
]


Sanitizing training data


Adversarial training
[
81
]
[
22
]


Backdoor detection algorithms
[
106
]


Gradient masking/obfuscation techniques: to prevent the adversary exploiting the gradient in white-box attacks. This family of defenses is deemed unreliable as these models are still vulnerable to black-box attacks or can be circumvented in other ways.
[
107
]


Ensembles
 of models have been proposed in the literature but caution should be applied when relying on them: usually ensembling weak classifiers results in a more accurate model but it does not seem to apply in the adversarial context.
[
108
]


See also
[
edit
]


Pattern recognition


Fawkes (image cloaking software)


Generative adversarial network


References
[
edit
]






^
 
Kianpour, Mazaher; Wen, Shao-Fang (2020). "Timing Attacks on Machine Learning: State of the Art". 
Intelligent Systems and Applications
. Advances in Intelligent Systems and Computing. Vol. 1037. pp. 111–125. 
doi
:
10.1007/978-3-030-29516-5_10
. 
ISBN
 
978-3-030-29515-8
. 
S2CID
 
201705926
.




^ 
a
 
b
 
Siva Kumar, Ram Shankar; Nyström, Magnus; Lambert, John; Marshall, Andrew; Goertzel, Mario; Comissoneru, Andi; Swann, Matt; Xia, Sharon (May 2020). 
"Adversarial Machine Learning-Industry Perspectives"
. 
2020 IEEE Security and Privacy Workshops (SPW)
. pp. 69–75. 
doi
:
10.1109/SPW50608.2020.00028
. 
ISBN
 
978-1-7281-9346-5
. 
S2CID
 
229357721
.




^
 
Goodfellow, Ian; McDaniel, Patrick; Papernot, Nicolas (25 June 2018). 
"Making machine learning robust against adversarial inputs"
. 
Communications of the ACM
. 
61
 (7): 56–66. 
doi
:
10.1145/3134599
. 
ISSN
 
0001-0782
.
[
permanent dead link
]




^
 
Geiping, Jonas; Fowl, Liam H.; Huang, W. Ronny; Czaja, Wojciech; Taylor, Gavin; Moeller, Michael; Goldstein, Tom (2020-09-28). 
Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching
. International Conference on Learning Representations 2021 (Poster).




^ 
a
 
b
 
c
 
El-Mhamdi, El Mahdi; Farhadkhani, Sadegh; Guerraoui, Rachid; Guirguis, Arsany; Hoang, Lê-Nguyên; Rouault, Sébastien (2021-12-06). 
"Collaborative Learning in the Jungle (Decentralized, Byzantine, Heterogeneous, Asynchronous and Nonconvex Learning)"
. 
Advances in Neural Information Processing Systems
. 
34
. 
arXiv
:
2008.00742
.




^
 
Tramèr, Florian; Zhang, Fan; Juels, Ari; Reiter, Michael K.; Ristenpart, Thomas (2016). 
Stealing Machine Learning Models via Prediction {APIs}
. 25th USENIX Security Symposium. pp. 601–618. 
ISBN
 
978-1-931971-32-4
.




^
 
"How to beat an adaptive/Bayesian spam filter (2004)"
. Retrieved 
2023-07-05
.




^ 
a
 
b
 
Biggio, Battista; Nelson, Blaine; Laskov, Pavel (2013-03-25). "Poisoning Attacks against Support Vector Machines". 
arXiv
:
1206.6389
 [
cs.LG
].




^ 
a
 
b
 
c
 
Biggio, Battista; Corona, Igino; Maiorca, Davide; Nelson, Blaine; Srndic, Nedim; Laskov, Pavel; Giacinto, Giorgio; Roli, Fabio (2013). "Evasion Attacks against Machine Learning at Test Time". 
Advanced Information Systems Engineering
. Lecture Notes in Computer Science. Vol. 7908. Springer. pp. 387–402. 
arXiv
:
1708.06131
. 
doi
:
10.1007/978-3-642-40994-3_25
. 
ISBN
 
978-3-642-38708-1
. 
S2CID
 
18716873
.




^
 
Szegedy, Christian; Zaremba, Wojciech; Sutskever, Ilya; Bruna, Joan; Erhan, Dumitru; Goodfellow, Ian; Fergus, Rob (2014-02-19). "Intriguing properties of neural networks". 
arXiv
:
1312.6199
 [
cs.CV
].




^ 
a
 
b
 
Biggio, Battista; Roli, Fabio (December 2018). "Wild patterns: Ten years after the rise of adversarial machine learning". 
Pattern Recognition
. 
84
: 317–331. 
arXiv
:
1712.03141
. 
Bibcode
:
2018PatRe..84..317B
. 
doi
:
10.1016/j.patcog.2018.07.023
. 
S2CID
 
207324435
.




^
 
Kurakin, Alexey; Goodfellow, Ian; Bengio, Samy (2016). "Adversarial examples in the physical world". 
arXiv
:
1607.02533
 [
cs.CV
].




^
 
Gupta, Kishor Datta, Dipankar Dasgupta, and Zahid Akhtar. "Applicability issues of Evasion-Based Adversarial Attacks and Mitigation Techniques." 2020 IEEE Symposium Series on Computational Intelligence (SSCI). 2020.




^ 
a
 
b
 
Lim, Hazel Si Min; Taeihagh, Araz (2019). 
"Algorithmic Decision-Making in AVs: Understanding Ethical and Technical Concerns for Smart Cities"
. 
Sustainability
. 
11
 (20): 5791. 
arXiv
:
1910.13122
. 
Bibcode
:
2019arXiv191013122L
. 
doi
:
10.3390/su11205791
. 
S2CID
 
204951009
.




^ 
a
 
b
 
"Google Brain's Nicholas Frosst on Adversarial Examples and Emotional Responses"
. 
Synced
. 2019-11-21
. Retrieved 
2021-10-23
.




^
 
"Responsible AI practices"
. 
Google AI
. Retrieved 
2021-10-23
.




^ 
a
 
b
 
c
 
Adversarial Robustness Toolbox (ART) v1.8
, Trusted-AI, 2021-10-23
, retrieved 
2021-10-23




^
 
amarshal. 
"Failure Modes in Machine Learning - Security documentation"
. 
docs.microsoft.com
. Retrieved 
2021-10-23
.




^ 
a
 
b
 
Biggio, Battista; Fumera, Giorgio; Roli, Fabio (2010). 
"Multiple classifier systems for robust classifier design in adversarial environments"
. 
International Journal of Machine Learning and Cybernetics
. 
1
 (1–4): 27–41. 
doi
:
10.1007/s13042-010-0007-7
. 
hdl
:
11567/1087824
. 
ISSN
 
1868-8071
. 
S2CID
 
8729381
. Archived from 
the original
 on 2023-01-19
. Retrieved 
2015-01-14
.




^ 
a
 
b
 
Brückner, Michael; Kanzow, Christian; Scheffer, Tobias (2012). 
"Static Prediction Games for Adversarial Learning Problems"
 
(PDF)
. 
Journal of Machine Learning Research
. 
13
 (Sep): 2617–2654. 
ISSN
 
1533-7928
.




^
 
Apruzzese, Giovanni; Andreolini, Mauro; Ferretti, Luca; Marchetti, Mirco; Colajanni, Michele (2021-06-03). "Modeling Realistic Adversarial Attacks against Network Intrusion Detection Systems". 
Digital Threats: Research and Practice
. 
3
 (3): 1–19. 
arXiv
:
2106.09380
. 
doi
:
10.1145/3469659
. 
ISSN
 
2692-1626
. 
S2CID
 
235458519
.




^ 
a
 
b
 
Vitorino, João; Oliveira, Nuno; Praça, Isabel (March 2022). 
"Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust Intrusion Detection"
. 
Future Internet
. 
14
 (4): 108. 
doi
:
10.3390/fi14040108
. 
hdl
:
10400.22/21851
. 
ISSN
 
1999-5903
.




^ 
a
 
b
 
Rodrigues, Ricardo N.; Ling, Lee Luan; Govindaraju, Venu (1 June 2009). 
"Robustness of multimodal biometric fusion methods against spoof attacks"
 
(PDF)
. 
Journal of Visual Languages & Computing
. 
20
 (3): 169–179. 
doi
:
10.1016/j.jvlc.2009.01.010
. 
ISSN
 
1045-926X
.




^
 
Su, Jiawei; Vargas, Danilo Vasconcellos; Sakurai, Kouichi (October 2019). "One Pixel Attack for Fooling Deep Neural Networks". 
IEEE Transactions on Evolutionary Computation
. 
23
 (5): 828–841. 
arXiv
:
1710.08864
. 
doi
:
10.1109/TEVC.2019.2890858
. 
ISSN
 
1941-0026
. 
S2CID
 
2698863
.




^
 
"Single pixel change fools AI programs"
. 
BBC News
. 3 November 2017
. Retrieved 
12 February
 2018
.




^
 
Athalye, Anish; Engstrom, Logan; Ilyas, Andrew; Kwok, Kevin (2017). "Synthesizing Robust Adversarial Examples". 
arXiv
:
1707.07397
 [
cs.CV
].




^
 
"AI Has a Hallucination Problem That's Proving Tough to Fix"
. 
WIRED
. 2018
. Retrieved 
10 March
 2018
.




^
 
Zhou, Zhenglong; Firestone, Chaz (2019). 
"Humans can decipher adversarial images"
. 
Nature Communications
. 
10
 (1): 1334. 
arXiv
:
1809.04120
. 
Bibcode
:
2019NatCo..10.1334Z
. 
doi
:
10.1038/s41467-019-08931-6
. 
PMC
 
6430776
. 
PMID
 
30902973
.




^
 
Ackerman, Evan (2017-08-04). 
"Slight Street Sign Modifications Can Completely Fool Machine Learning Algorithms"
. 
IEEE Spectrum: Technology, Engineering, and Science News
. Retrieved 
2019-07-15
.




^
 
"A Tiny Piece of Tape Tricked Teslas Into Speeding Up 50 MPH"
. 
Wired
. 2020
. Retrieved 
11 March
 2020
.




^
 
"Model Hacking ADAS to Pave Safer Roads for Autonomous Vehicles"
. 
McAfee Blogs
. 2020-02-19
. Retrieved 
2020-03-11
.




^
 
Seabrook, John (2020). 
"Dressing for the Surveillance Age"
. 
The New Yorker
. Retrieved 
5 April
 2020
.




^ 
a
 
b
 
c
 
Heaven, Douglas (October 2019). "Why deep-learning AIs are so easy to fool". 
Nature
. 
574
 (7777): 163–166. 
Bibcode
:
2019Natur.574..163H
. 
doi
:
10.1038/d41586-019-03013-5
. 
PMID
 
31597977
. 
S2CID
 
203928744
.




^
 
Hutson, Matthew (10 May 2019). "AI can now defend itself against malicious messages hidden in speech". 
Nature
. 
doi
:
10.1038/d41586-019-01510-1
. 
PMID
 
32385365
. 
S2CID
 
189666088
.




^
 
Lepori, Michael A; Firestone, Chaz (2020-03-27). "Can you hear me now? Sensitive comparisons of human and machine perception". 
arXiv
:
2003.12362
 [
eess.AS
].




^
 
Vadillo, Jon; Santana, Roberto (2020-01-23). "On the human evaluation of audio adversarial examples". 
arXiv
:
2001.08444
 [
eess.AS
].




^
 
D. B. Skillicorn. "Adversarial knowledge discovery". IEEE Intelligent Systems, 24:54–61, 2009.




^ 
a
 
b
 
B. Biggio, G. Fumera, and F. Roli. "
Pattern recognition systems under attack: Design issues and research challenges
 
Archived
 2022-05-20 at the 
Wayback Machine
". Int'l J. Patt. Recogn. Artif. Intell., 28(7):1460002, 2014.




^ 
a
 
b
 
Barreno, Marco; Nelson, Blaine; Joseph, Anthony D.; Tygar, J. D. (2010). 
"The security of machine learning"
 
(PDF)
. 
Machine Learning
. 
81
 (2): 121–148. 
doi
:
10.1007/s10994-010-5188-5
. 
S2CID
 
2304759
.




^
 
Sikos, Leslie F. (2019). 
AI in Cybersecurity
. Intelligent Systems Reference Library. Vol. 151. Cham: Springer. p. 50. 
doi
:
10.1007/978-3-319-98842-9
. 
ISBN
 
978-3-319-98841-2
. 
S2CID
 
259216663
.




^ 
a
 
b
 
c
 
B. Biggio, G. Fumera, and F. Roli. "
Security evaluation of pattern classifiers under attack
 
Archived
 2018-05-18 at the 
Wayback Machine
". IEEE Transactions on Knowledge and Data Engineering, 26(4):984–996, 2014.




^ 
a
 
b
 
c
 
d
 
e
 
Biggio, Battista; Corona, Igino; Nelson, Blaine; Rubinstein, Benjamin I. P.; Maiorca, Davide; Fumera, Giorgio; Giacinto, Giorgio; Roli, Fabio (2014). "Security Evaluation of Support Vector Machines in Adversarial Environments". 
Support Vector Machines Applications
. Springer International Publishing. pp. 105–153. 
arXiv
:
1401.7727
. 
doi
:
10.1007/978-3-319-02300-7_4
. 
ISBN
 
978-3-319-02300-7
. 
S2CID
 
18666561
.




^
 
Heinrich, Kai; Graf, Johannes; Chen, Ji; Laurisch, Jakob; Zschech, Patrick (2020-06-15). 
"Fool Me Once, Shame On You, Fool Me Twice, Shame On Me: A Taxonomy of Attack and De-fense Patterns for AI Security"
. 
ECIS 2020 Research Papers
.




^
 
"Facebook removes 15 Billion fake accounts in two years"
. 
Tech Digest
. 2021-09-27
. Retrieved 
2022-06-08
.




^
 
"Facebook removed 3 billion fake accounts in just 6 months"
. 
New York Post
. Associated Press. 2019-05-23
. Retrieved 
2022-06-08
.




^
 
Schwarzschild, Avi; Goldblum, Micah; Gupta, Arjun; Dickerson, John P.; Goldstein, Tom (2021-07-01). 
"Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks"
. 
International Conference on Machine Learning
. PMLR: 9389–9398.




^
 
B. Biggio, B. Nelson, and P. Laskov. "
Support vector machines under adversarial label noise
 
Archived
 2020-08-03 at the 
Wayback Machine
". In Journal of Machine Learning Research – Proc. 3rd Asian Conf. Machine Learning, volume 20, pp. 97–112, 2011.




^
 
M. Kloft and P. Laskov. "
Security analysis of online centroid anomaly detection
". Journal of Machine Learning Research, 13:3647–3690, 2012.




^
 
Edwards, Benj (2023-10-25). 
"University of Chicago researchers seek to "poison" AI art generators with Nightshade"
. 
Ars Technica
. Retrieved 
2023-10-27
.




^
 
Rao, Rahul. 
"AI-Generated Data Can Poison Future AI Models"
. 
Scientific American
. Retrieved 
2024-06-22
.




^
 
Baruch, Gilad; Baruch, Moran; Goldberg, Yoav (2019). 
"A Little Is Enough: Circumventing Defenses For Distributed Learning"
. 
Advances in Neural Information Processing Systems
. 
32
. Curran Associates, Inc. 
arXiv
:
1902.06156
.




^
 
El-Mhamdi, El-Mahdi; Guerraoui, Rachid; Guirguis, Arsany; Hoang, Lê-Nguyên; Rouault, Sébastien (2022-05-26). 
"Genuinely distributed Byzantine machine learning"
. 
Distributed Computing
. 
35
 (4): 305–331. 
arXiv
:
1905.03853
. 
doi
:
10.1007/s00446-022-00427-9
. 
ISSN
 
1432-0452
. 
S2CID
 
249111966
.




^
 
Goldwasser, S.; Kim, Michael P.; Vaikuntanathan, V.; Zamir, Or (2022). "Planting Undetectable Backdoors in Machine Learning Models". 
arXiv
:
2204.06974
 [
cs.LG
].




^ 
a
 
b
 
Blanchard, Peva; El Mhamdi, El Mahdi; Guerraoui, Rachid; Stainer, Julien (2017). 
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
. 
Advances in Neural Information Processing Systems
. 
30
. Curran Associates, Inc.




^
 
Chen, Lingjiao; Wang, Hongyi; Charles, Zachary; Papailiopoulos, Dimitris (2018-07-03). 
"DRACO: Byzantine-resilient Distributed Training via Redundant Gradients"
. 
International Conference on Machine Learning
. PMLR: 903–912. 
arXiv
:
1803.09877
.




^
 
Mhamdi, El Mahdi El; Guerraoui, Rachid; Rouault, Sébastien (2018-07-03). 
"The Hidden Vulnerability of Distributed Learning in Byzantium"
. 
International Conference on Machine Learning
. PMLR: 3521–3530. 
arXiv
:
1802.07927
.




^
 
Allen-Zhu, Zeyuan; Ebrahimianghazani, Faeze; Li, Jerry; Alistarh, Dan (2020-09-28). "Byzantine-Resilient Non-Convex Stochastic Gradient Descent". 
arXiv
:
2012.14368
 [
cs.LG
].
 
Review




^
 
Mhamdi, El Mahdi El; Guerraoui, Rachid; Rouault, Sébastien (2020-09-28). 
Distributed Momentum for Byzantine-resilient Stochastic Gradient Descent
. 9th International Conference on Learning Representations (ICLR), May 4-8, 2021 (virtual conference)
. Retrieved 
2022-10-20
.
 
Review




^
 
Data, Deepesh; Diggavi, Suhas (2021-07-01). 
"Byzantine-Resilient High-Dimensional SGD with Local Iterations on Heterogeneous Data"
. 
International Conference on Machine Learning
. PMLR: 2478–2488.




^
 
Karimireddy, Sai Praneeth; He, Lie; Jaggi, Martin (2021-09-29). "Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing". 
arXiv
:
2006.09365
 [
cs.LG
].
 
Review




^
 
B. Nelson, B. I. Rubinstein, L. Huang, A. D. Joseph, S. J. Lee, S. Rao, and J. D. Tygar. "
Query strategies for evading convex-inducing classifiers
". J. Mach. Learn. Res., 13:1293–1332, 2012




^
 
"How to steal modern NLP systems with gibberish?"
. 
cleverhans-blog
. 2020-04-06
. Retrieved 
2020-10-15
.




^ 
a
 
b
 
c
 
d
 
e
 
f
 
g
 
h
 
Wang, Xinran; Xiang, Yu; Gao, Jun; Ding, Jie (2020-09-13). "Information Laundering for Model Privacy". 
arXiv
:
2009.06112
 [
cs.CR
].




^ 
a
 
b
 
Dickson, Ben (2021-04-23). 
"Machine learning: What are membership inference attacks?"
. 
TechTalks
. Retrieved 
2021-11-07
.




^
 
Goodfellow, Ian; Shlens, Jonathan; Szegedy, Christian (2015). "Explaining and Harnessing Adversarial Examples". 
International Conference on Learning Representations
. 
arXiv
:
1412.6572
.




^
 
Pieter, Huang; Papernot, Sandy; Goodfellow, Nicolas; Duan, Ian; Abbeel, Yan (2017-02-07). 
Adversarial Attacks on Neural Network Policies
. 
OCLC
 
1106256905
.




^
 
Korkmaz, Ezgi (2022). "Deep Reinforcement Learning Policies Learn Shared Adversarial Features Across MDPs". 
Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22)
. 
36
 (7): 7229–7238. 
arXiv
:
2112.09025
. 
doi
:
10.1609/aaai.v36i7.20684
. 
S2CID
 
245219157
.




^
 
Carlini, Nicholas; Wagner, David (2018). "Audio Adversarial Examples: Targeted Attacks on Speech-to-Text". 
2018 IEEE Security and Privacy Workshops (SPW)
. pp. 1–7. 
arXiv
:
1801.01944
. 
doi
:
10.1109/SPW.2018.00009
. 
ISBN
 
978-1-5386-8276-0
. 
S2CID
 
4475201
.




^
 
Goodfellow, Ian J.; Shlens, Jonathon; Szegedy, Christian (2015). 
Explaining and Harnessing Adversarial Examples
. International Conference on Learning Representations (ICLR).




^
 
Ribeiro, Antonio H.; Zachariah, Dave; Bach, Francis; Schön, Thomas B. (2023). 
Regularization properties of adversarially-trained linear regression
. Thirty-seventh Conference on Neural Information Processing Systems.




^
 
Tsipras, Dimitris; Santurkar, Shibani; Engstrom, Logan; Turner, Alexander; Ma, Aleksander (2019). 
Robustness May Be At Odds with Accuracy
. International Conference for Learning Representations.




^
 
Dan, C.; Wei, Y.; Ravikumar, P. (2020). 
Sharp statistical guarantees for adversarially robust Gaussian classification
. International Conference on Machine Learning.




^
 
Javanmard, A.; Soltanolkotabi, M.; Hassani, H. (2020). 
Precise tradeoffs in adversarial training for linear regression
. Conference on Learning Theory.




^
 
Ribeiro, A. H.; Schön, T. B. (2023). "Overparameterized Linear Regression under Adversarial Attacks". 
IEEE Transactions on Signal Processing
. 
71
: 601–614. 
arXiv
:
2204.06274
. 
Bibcode
:
2023ITSP...71..601R
. 
doi
:
10.1109/TSP.2023.3246228
.




^
 
Yin, D.; Kannan, R.; Bartlett, P. (2019). 
Rademacher Complexity for Adversarially Robust Generalization
. International Conference on Machine Learning.




^
 
Jagielski, Matthew; Oprea, Alina; Biggio, Battista; Liu, Chang; Nita-Rotaru, Cristina; Li, Bo (May 2018). "Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning". 
2018 IEEE Symposium on Security and Privacy (SP)
. IEEE. pp. 19–35. 
arXiv
:
1804.00308
. 
doi
:
10.1109/sp.2018.00057
. 
ISBN
 
978-1-5386-4353-2
. 
S2CID
 
4551073
.




^
 
"Attacking Machine Learning with Adversarial Examples"
. 
OpenAI
. 2017-02-24
. Retrieved 
2020-10-15
.




^
 
Gu, Tianyu; Dolan-Gavitt, Brendan; Garg, Siddharth (2019-03-11). "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain". 
arXiv
:
1708.06733
 [
cs.CR
].




^
 
Veale, Michael; Binns, Reuben; Edwards, Lilian (2018-11-28). 
"Algorithms that remember: model inversion attacks and data protection law"
. 
Philosophical Transactions. Series A, Mathematical, Physical, and Engineering Sciences
. 
376
 (2133). 
arXiv
:
1807.04644
. 
Bibcode
:
2018RSPTA.37680083V
. 
doi
:
10.1098/rsta.2018.0083
. 
ISSN
 
1364-503X
. 
PMC
 
6191664
. 
PMID
 
30322998
.




^
 
Shokri, Reza; Stronati, Marco; Song, Congzheng; Shmatikov, Vitaly (2017-03-31). "Membership Inference Attacks against Machine Learning Models". 
arXiv
:
1610.05820
 [
cs.CR
].




^ 
a
 
b
 
Goodfellow, Ian J.; Shlens, Jonathon; Szegedy, Christian (2015-03-20). "Explaining and Harnessing Adversarial Examples". 
arXiv
:
1412.6572
 [
stat.ML
].




^
 
Madry, Aleksander; Makelov, Aleksandar; Schmidt, Ludwig; Tsipras, Dimitris; Vladu, Adrian (2019-09-04). "Towards Deep Learning Models Resistant to Adversarial Attacks". 
arXiv
:
1706.06083
 [
stat.ML
].




^
 
Carlini, Nicholas; Wagner, David (2017-03-22). "Towards Evaluating the Robustness of Neural Networks". 
arXiv
:
1608.04644
 [
cs.CR
].




^
 
Brown, Tom B.; Mané, Dandelion; Roy, Aurko; Abadi, Martín; Gilmer, Justin (2018-05-16). "Adversarial Patch". 
arXiv
:
1712.09665
 [
cs.CV
].




^
 
Guo, Sensen; Zhao, Jinxiong; Li, Xiaoyu; Duan, Junhong; Mu, Dejun; Jing, Xiao (2021-04-24). 
"A Black-Box Attack Method against Machine-Learning-Based Anomaly Network Flow Detection Models"
. 
Security and Communication Networks
. 
2021
. e5578335. 
doi
:
10.1155/2021/5578335
. 
ISSN
 
1939-0114
.




^ 
a
 
b
 
Gomes, Joao (2018-01-17). 
"Adversarial Attacks and Defences for Convolutional Neural Networks"
. 
Onfido Tech
. Retrieved 
2021-10-23
.




^ 
a
 
b
 
c
 
d
 
e
 
Andriushchenko, Maksym; Croce, Francesco; Flammarion, Nicolas; Hein, Matthias (2020). 
"Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search"
. In Vedaldi, Andrea; Bischof, Horst; Brox, Thomas; Frahm, Jan-Michael (eds.). 
Computer Vision – ECCV 2020
. Lecture Notes in Computer Science. Vol. 12368. Cham: Springer International Publishing. pp. 484–501. 
arXiv
:
1912.00049
. 
doi
:
10.1007/978-3-030-58592-1_29
. 
ISBN
 
978-3-030-58592-1
. 
S2CID
 
208527215
.




^ 
a
 
b
 
c
 
d
 
e
 
f
 
g
 
Chen, Jianbo; Jordan, Michael I.; Wainwright, Martin J. (2019), 
HopSkipJumpAttack: A Query-Efficient Decision-Based Attack
, 
arXiv
:
1904.02144
, retrieved 
2021-10-25




^
 
Andriushchenko, Maksym; Croce, Francesco; Flammarion, Nicolas; Hein, Matthias (2020-07-29). "Square Attack: a query-efficient black-box adversarial attack via random search". 
arXiv
:
1912.00049
 [
cs.LG
].




^
 
"Black-box decision-based attacks on images"
. 
KejiTech
. 2020-06-21
. Retrieved 
2021-10-25
.




^ 
a
 
b
 
Goodfellow, Ian J.; Shlens, Jonathon; Szegedy, Christian (2015-03-20). "Explaining and Harnessing Adversarial Examples". 
arXiv
:
1412.6572
 [
stat.ML
].




^ 
a
 
b
 
"Adversarial example using FGSM | TensorFlow Core"
. 
TensorFlow
. Retrieved 
2021-10-24
.




^
 
Tsui, Ken (2018-08-22). 
"Perhaps the Simplest Introduction of Adversarial Examples Ever"
. 
Medium
. Retrieved 
2021-10-24
.




^
 
Corona-Figueroa, Abril; Bond-Taylor, Sam; Bhowmik, Neelanjan; Gaus, Yona Falinie A.; Breckon, Toby P.; Shum, Hubert P. H.; Willcocks, Chris G. (2023). 
Unaligned 2D to 3D Translation with Conditional Vector-Quantized Code Diffusion using Transformers
. IEEE/CVF. 
arXiv
:
2308.14152
.




^ 
a
 
b
 
Carlini, Nicholas; Wagner, David (2017-03-22). "Towards Evaluating the Robustness of Neural Networks". 
arXiv
:
1608.04644
 [
cs.CR
].




^
 
"carlini wagner attack"
. 
richardjordan.com
. Retrieved 
2021-10-23
.




^
 
Plotz, Mike (2018-11-26). 
"Paper Summary: Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods"
. 
Medium
. Retrieved 
2021-10-23
.




^
 
Kishor Datta Gupta; Akhtar, Zahid; Dasgupta, Dipankar (2021). "Determining Sequence of Image Processing Technique (IPT) to Detect Adversarial Attacks". 
SN Computer Science
. 
2
 (5): 383. 
arXiv
:
2007.00337
. 
doi
:
10.1007/s42979-021-00773-8
. 
ISSN
 
2662-995X
. 
S2CID
 
220281087
.




^
 
O. Dekel, O. Shamir, and L. Xiao. "
Learning to classify with missing and corrupted features
". Machine Learning, 81:149–178, 2010.




^
 
Liu, Wei; Chawla, Sanjay (2010). 
"Mining adversarial patterns via regularized loss minimization"
 
(PDF)
. 
Machine Learning
. 
81
: 69–83. 
doi
:
10.1007/s10994-010-5199-2
. 
S2CID
 
17497168
.




^
 
B. Biggio, G. Fumera, and F. Roli. "
Evade hard multiple classifier systems
 
Archived
 2015-01-15 at the 
Wayback Machine
". In O. Okun and G. Valentini, editors, Supervised and Unsupervised Ensemble Methods and Their Applications, volume 245 of Studies in Computational Intelligence, pages 15–38. Springer Berlin / Heidelberg, 2009.




^
 
B. I. P. Rubinstein, P. L. Bartlett, L. Huang, and N. Taft. "
Learning in a large function space: Privacy- preserving mechanisms for svm learning
". Journal of Privacy and Confidentiality, 4(1):65–100, 2012.




^
 
M. Kantarcioglu, B. Xi, C. Clifton. 
"Classifier Evaluation and Attribute Selection against Active Adversaries"
. Data Min. Knowl. Discov., 22:291–335, January 2011.




^
 
Chivukula, Aneesh; Yang, Xinghao; Liu, Wei; Zhu, Tianqing; Zhou, Wanlei (2020). 
"Game Theoretical Adversarial Deep Learning with Variational Adversaries"
. 
IEEE Transactions on Knowledge and Data Engineering
. 
33
 (11): 3568–3581. 
doi
:
10.1109/TKDE.2020.2972320
. 
hdl
:
10453/145751
. 
ISSN
 
1558-2191
. 
S2CID
 
213845560
.




^
 
Chivukula, Aneesh Sreevallabh; Liu, Wei (2019). 
"Adversarial Deep Learning Models with Multiple Adversaries"
. 
IEEE Transactions on Knowledge and Data Engineering
. 
31
 (6): 1066–1079. 
doi
:
10.1109/TKDE.2018.2851247
. 
hdl
:
10453/136227
. 
ISSN
 
1558-2191
. 
S2CID
 
67024195
.




^
 
"TrojAI"
. 
www.iarpa.gov
. Retrieved 
2020-10-14
.




^
 
Athalye, Anish; Carlini, Nicholas; Wagner, David (2018-02-01). "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Example". 
arXiv
:
1802.00420v1
 [
cs.LG
].




^
 
He, Warren; Wei, James; Chen, Xinyun; Carlini, Nicholas; Song, Dawn (2017-06-15). "Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong". 
arXiv
:
1706.04701
 [
cs.LG
].






External links
[
edit
]


MITRE ATLAS: Adversarial Threat Landscape for Artificial-Intelligence Systems


NIST 8269 Draft: A Taxonomy and Terminology of Adversarial Machine Learning


NIPS 2007 Workshop on 
Machine Learning in Adversarial Environments for Computer Security


AlfaSVMLib
 
Archived
 2020-09-24 at the 
Wayback Machine
 – Adversarial Label Flip Attacks against Support Vector Machines


Laskov, Pavel; Lippmann, Richard (2010). "Machine learning in adversarial environments". 
Machine Learning
. 
81
 (2): 115–119. 
doi
:
10.1007/s10994-010-5207-6
. 
S2CID
 
12567278
.


Dagstuhl Perspectives Workshop on "
Machine Learning Methods for Computer Security
"


Workshop on 
Artificial Intelligence and Security
, (AISec) Series


v
t
e
Differentiable computing
General


Differentiable programming


Information geometry


Statistical manifold


Automatic differentiation


Neuromorphic engineering


Pattern recognition


Tensor calculus


Computational learning theory


Inductive bias


Concepts


Gradient descent


SGD


Clustering


Regression


Overfitting


Hallucination


Adversary


Attention


Convolution


Loss functions


Backpropagation


Batchnorm


Activation


Softmax


Sigmoid


Rectifier


Regularization


Datasets


Augmentation


Diffusion


Autoregression


Applications


Machine learning


In-context learning


Artificial neural network


Deep learning


Scientific computing


Artificial Intelligence


Language model


Large language model


Hardware


IPU


TPU


VPU


Memristor


SpiNNaker


Software libraries


TensorFlow


PyTorch


Keras


Theano


JAX


Flux.jl


MindSpore


Implementations
Audio–visual


AlexNet


WaveNet


Human image synthesis


HWR


OCR


Speech synthesis


Speech recognition


Facial recognition


AlphaFold


Text-to-image models


DALL-E


Midjourney


Stable Diffusion


Text-to-video models


Sora


VideoPoet


Whisper


Verbal


Word2vec


Seq2seq


BERT


Gemini


LaMDA


Bard


NMT


Project Debater


IBM Watson


IBM Watsonx


Granite


GPT-1


GPT-2


GPT-3


GPT-4


ChatGPT


GPT-J


Chinchilla AI


PaLM


BLOOM


LLaMA


PanGu-Σ


Decisional


AlphaGo


AlphaZero


Q-learning


SARSA


OpenAI Five


Self-driving car


MuZero


Action selection


Auto-GPT


Robot control


People


Yoshua Bengio


Alex Graves


Ian Goodfellow


Stephen Grossberg


Demis Hassabis


Geoffrey Hinton


Yann LeCun


Fei-Fei Li


Andrew Ng


Jürgen Schmidhuber


David Silver


Ilya Sutskever


Organizations


Anthropic


EleutherAI


Google DeepMind


Hugging Face


OpenAI


Meta AI


Mila


MIT CSAIL


Huawei


Architectures


Neural Turing machine


Differentiable neural computer


Transformer


Recurrent neural network (RNN)


Long short-term memory (LSTM)


Gated recurrent unit (GRU)


Echo state network


Multilayer perceptron (MLP)


Convolutional neural network


Residual neural network


Mamba


Autoencoder


Variational autoencoder (VAE)


Generative adversarial network (GAN)


Graph neural network




 Portals

Computer programming


Technology


 Categories

Artificial neural networks


Machine learning












Retrieved from "
https://en.wikipedia.org/w/index.php?title=Adversarial_machine_learning&oldid=1240906035
"


Categories
: 
Machine learning
Computer security
AI safety
Hidden categories: 
All articles with dead external links
Articles with dead external links from February 2022
Articles with permanently dead external links
Webarchive template wayback links
Articles with short description
Short description is different from Wikidata




Read more
## Ian Goodfellow













Toggle the table of contents
















Ian Goodfellow








16 languages










Afrikaans
العربية
Deutsch
فارسی
Français
Galego
한국어
Italiano
עברית
مصرى
Nederlands
日本語
Türkçe
Українська
Tiếng Việt
中文




Edit links
























Article
Talk












English




































Read
Edit
View history
















Tools












Tools


move to sidebar


hide







		Actions
	






Read
Edit
View history











		General
	






What links here
Related changes
Upload file
Special pages
Permanent link
Page information
Cite this page
Get shortened URL
Download QR code
Wikidata item











		Print/export
	






Download as PDF
Printable version











		In other projects
	






Wikimedia Commons












































Appearance


move to sidebar


hide






















From Wikipedia, the free encyclopedia






American computer scientist






Ian Goodfellow
Born
November 18th, 1987
[
1
]
Nationality
American
Alma mater
Stanford University
Université de Montréal
Known for
Generative adversarial networks
, 
Adversarial examples
Scientific career
Fields
Computer science
Institutions
Apple Inc.
 
Google Brain
OpenAI
DeepMind
Google DeepMind
Thesis
Deep Learning of Representations and its Application to Computer Vision
 
(2014)
Doctoral advisor
Yoshua Bengio
Aaron Courville


Website
www
.iangoodfellow
.com


Ian J. Goodfellow
 (born 1987
[
1
]
) is an American 
computer scientist
, 
engineer
, and 
executive
, most noted for his work on 
artificial neural networks
 and 
deep learning
. He was previously employed as a research scientist at 
Google Brain
 and director of machine learning at 
Apple
 and has made several important contributions to the field of 
deep learning
 including the invention of the 
generative adversarial network
 (GAN). Goodfellow co-wrote, as the first author, the textbook 
Deep Learning
 (2016)
[
2
]
 and wrote the chapter on deep learning in the authoritative textbook of the field of artificial intelligence, 
Artificial Intelligence: A Modern Approach
[
3
]
[
4
]
 (used in more than 1,500 universities in 135 countries).
[
5
]
. He is currently a Research Scientist at Deepmind.
[
6
]






Education
[
edit
]


Goodfellow obtained his 
B.S.
 and 
M.S.
 in computer science from 
Stanford University
 under the supervision of 
Andrew Ng
 (co-founder and head of 
Google Brain
), and his Ph.D. in machine learning from the 
Université de Montréal
 in April 2014, under the supervision of 
Yoshua Bengio
 and Aaron Courville.
[
7
]
[
8
]
 Goodfellow's thesis is titled 
Deep learning of representations and its application to computer vision
.
[
9
]
[
10
]




Career
[
edit
]


After graduation, Goodfellow joined 
Google
 as part of the 
Google Brain
 research team.
[
11
]
 In March 2016 he left Google to join the newly founded 
OpenAI
 research laboratory.
[
12
]
  Barely 11 months later, in March 2017, Goodfellow returned to Google Research
[
13
]
 but left again in 2019.
[
14
]


In 2019 Goodfellow joined 
Apple
 as director of machine learning in the Special Projects Group.
[
14
]
 He resigned from Apple in April 2022 to protest Apple's plan to require in-person work for its employees.
[
15
]
 Goodfellow then joined 
DeepMind
 as a research scientist.
[
16
]




Research
[
edit
]


Goodfellow is best known for inventing 
generative adversarial networks
 (GAN), using deep learning to generate images. This approach uses two neural networks to competitively  improve an image's quality. A “generator” network creates a synthetic image based on an initial set of images such as a collection of faces. A “discriminator” network tries to detect whether or not the generator's output is real or fake. Then the generate-detect cycle is repeated. For each iteration, the generator and the discriminator use the other's feedback to improve or detect the generated images, until the discriminator can no longer distinguish between the fakes generated by its opponent and the real thing. The ability to create high quality generated imagery has increased rapidly. Unfortunately, so has its malicious use, to create 
deepfakes
 and generate video-based 
disinformation
.
[
17
]
[
18
]


At Google, Goodfellow developed a system enabling 
Google Maps
 to automatically transcribe addresses from photos taken by 
Street View cars
[
19
]
[
20
]
 and demonstrated security vulnerabilities of machine learning systems.
[
21
]
[
22
]




Recognition
[
edit
]


In 2017, Goodfellow was cited in 
MIT Technology Review
's 35 Innovators Under 35.
[
23
]
 In 2019, he was included in 
Foreign Policy
's list of 100 Global Thinkers.
[
24
]




References
[
edit
]






^ 
a
 
b
 
"Katalog der Deutschen Nationalbibliothek"
. 
portal.dnb.de
.




^
 
Goodfellow, Ian; Bengio, Yoshua; Courville, Aaron (2016). 
Deep Learning
. Cambridge, Massachusetts: MIT Press.




^
 
"Artificial Intelligence: A Modern Approach - The Definitive AI Book"
. 
How to Learn Machine Learning
. 2020
. Retrieved 
19 December
 2022
.




^
 
Goodfellow, Ian (28 April 2020). "Chapter 21: Deep Learning". 
Artificial intelligence : a modern approach
 
(PDF)
. By Russell, Stuart J.; Norvig, Peter (Fourth ed.). Hoboken, NJ: Pearson. 
ISBN
 
978-0134610993
.




^
 
"Nobel Week Dialogue"
. 
NobelPrize.org
. Retrieved 
19 December
 2022
.




^
 
Wayt, Theo. 
"Apple engineer who quit over return-to-office policy joins Google"
. 
New York Post
. New York Post
. Retrieved 
31 August
 2024
.




^
 
"Top 12 AI Leaders and Researchers you Should Know in 2022"
. 
Great Learning Blog: Free Resources what Matters to shape your Career!
. 9 May 2022
. Retrieved 
19 December
 2022
.




^
 
La Barbera, Steve (27 March 2019). 
"Montreal's Yoshua Bengio Honored with the 'Nobel Prize' of Computing"
. 
Montreal in Technology
. Archived from 
the original
 on 19 December 2022
. Retrieved 
19 December
 2022
.




^
 
Goodfellow, Ian (18 February 2015). 
Deep learning of representations and its application to computer vision
 (Thesis). 
hdl
:
1866/11674
.




^
 
"Ian Goodfellow PhD Defense Presentation"
. Universite de Montreal. 3 September 2014
. Retrieved 
27 October
 2020
.




^
 
Metz, Cade (15 February 2022). 
Genius Makers: The Mavericks Who Brought AI to Google, Facebook, and the World
. Penguin. pp. 203–213. 
ISBN
 
978-1-5247-4269-0
. Retrieved 
19 December
 2022
.




^
 
Metz, Cade (27 April 2016). 
"Inside OpenAI, Elon Musk's Wild Plan to Set Artificial Intelligence Free"
. 
Wired
. Retrieved 
31 July
 2016
.




^
 
Metz, Cade (19 April 2018). 
"A.I. Researchers Are Making More Than $1 Million, Even at a Nonprofit"
. 
The New York Times
. Retrieved 
19 December
 2022
.




^ 
a
 
b
 
Novet, Jordan (5 April 2019). 
"Apple hires AI expert Ian Goodfellow from Google"
. 
www.cnbc.com
. Retrieved 
5 April
 2019
.




^
 
"Apple's Director of Machine Learning Resigns Due to Return to Office Work"
. 
MacRumors
. 7 May 2022
. Retrieved 
7 May
 2022
.




^
 
Greene, Tristan (19 May 2022). 
"Losing Ian Goodfellow to DeepMind is the dumbest thing Apple's ever done"
. 
TNW | Neural
. Retrieved 
11 June
 2022
.




^
 
Waldrop, M. Mitchell (16 March 2020). 
"Synthetic media: The real trouble with deepfakes"
. 
Knowable Magazine
. Annual Reviews. 
doi
:
10.1146/knowable-031320-1
. 
S2CID
 
215882738
. Retrieved 
19 December
 2022
.




^
 
Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua (2014). "Generative Adversarial Networks". 
arXiv
:
1406.2661
 [
stat.ML
].




^
 
"How Google Cracked House Number Identification in Street View"
. 
MIT Technology Review
. 6 January 2014
. Retrieved 
31 July
 2016
.




^
 
Ibarz, Julian; Banerjee, Sujoy (3 May 2017). 
"Updating Google Maps with Deep Learning and Street View"
. 
Research Blog
. Retrieved 
4 May
 2017
.




^
 
Gershgorn, Dave (30 March 2016). 
"Fooling the Machine"
. 
Popular Science
. Retrieved 
31 July
 2016
.




^
 
Gershgorn, Dave (27 July 2016). 
"Researchers Have Successfully Tricked A.I. Into Seeing The Wrong Things"
. 
Popular Science
. Retrieved 
31 July
 2016
.




^
 
Knight, Will (16 August 2017). 
"Ian Goodfellow"
. 
MIT Technology Review
.




^
 
"A decade of Global Thinkers"
. 
Foreign Policy
. 2019.






v
t
e
Differentiable computing
General


Differentiable programming


Information geometry


Statistical manifold


Automatic differentiation


Neuromorphic engineering


Pattern recognition


Tensor calculus


Computational learning theory


Inductive bias


Concepts


Gradient descent


SGD


Clustering


Regression


Overfitting


Hallucination


Adversary


Attention


Convolution


Loss functions


Backpropagation


Batchnorm


Activation


Softmax


Sigmoid


Rectifier


Regularization


Datasets


Augmentation


Diffusion


Autoregression


Applications


Machine learning


In-context learning


Artificial neural network


Deep learning


Scientific computing


Artificial Intelligence


Language model


Large language model


Hardware


IPU


TPU


VPU


Memristor


SpiNNaker


Software libraries


TensorFlow


PyTorch


Keras


Theano


JAX


Flux.jl


MindSpore


Implementations
Audio–visual


AlexNet


WaveNet


Human image synthesis


HWR


OCR


Speech synthesis


Speech recognition


Facial recognition


AlphaFold


Text-to-image models


DALL-E


Midjourney


Stable Diffusion


Text-to-video models


Sora


VideoPoet


Whisper


Verbal


Word2vec


Seq2seq


BERT


Gemini


LaMDA


Bard


NMT


Project Debater


IBM Watson


IBM Watsonx


Granite


GPT-1


GPT-2


GPT-3


GPT-4


ChatGPT


GPT-J


Chinchilla AI


PaLM


BLOOM


LLaMA


PanGu-Σ


Decisional


AlphaGo


AlphaZero


Q-learning


SARSA


OpenAI Five


Self-driving car


MuZero


Action selection


Auto-GPT


Robot control


People


Yoshua Bengio


Alex Graves


Ian Goodfellow


Stephen Grossberg


Demis Hassabis


Geoffrey Hinton


Yann LeCun


Fei-Fei Li


Andrew Ng


Jürgen Schmidhuber


David Silver


Ilya Sutskever


Organizations


Anthropic


EleutherAI


Google DeepMind


Hugging Face


OpenAI


Meta AI


Mila


MIT CSAIL


Huawei


Architectures


Neural Turing machine


Differentiable neural computer


Transformer


Recurrent neural network (RNN)


Long short-term memory (LSTM)


Gated recurrent unit (GRU)


Echo state network


Multilayer perceptron (MLP)


Convolutional neural network


Residual neural network


Mamba


Autoencoder


Variational autoencoder (VAE)


Generative adversarial network (GAN)


Graph neural network




 Portals

Computer programming


Technology


 Categories

Artificial neural networks


Machine learning




Authority control databases
 
International
VIAF
WorldCat
National
Germany
Israel
United States
Czech Republic
Poland
Academics
Association for Computing Machinery
DBLP
Google Scholar
MathSciNet
Mathematics Genealogy Project
ORCID
Scopus
Other
IdRef










Retrieved from "
https://en.wikipedia.org/w/index.php?title=Ian_Goodfellow&oldid=1243274511
"


Categories
: 
American computer scientists
American artificial intelligence researchers
Google employees
Living people
Machine learning researchers
Scientists from San Francisco
Stanford University School of Engineering alumni
Université de Montréal alumni
Apple Inc. employees
1987 births
Hidden categories: 
Articles with short description
Short description matches Wikidata
Use dmy dates from December 2022
Articles with hCards
Place of birth missing (living people)




Read more
## Markov kernel













Toggle the table of contents
















Markov kernel








1 language










Català




Edit links
























Article
Talk












English




































Read
Edit
View history
















Tools












Tools


move to sidebar


hide







		Actions
	






Read
Edit
View history











		General
	






What links here
Related changes
Upload file
Special pages
Permanent link
Page information
Cite this page
Get shortened URL
Download QR code
Wikidata item











		Print/export
	






Download as PDF
Printable version












































Appearance


move to sidebar


hide






















From Wikipedia, the free encyclopedia






Concept in probability theory


In 
probability theory
, a 
Markov kernel
 (also known as a 
stochastic kernel
 or 
probability kernel
) is a map that in the general theory of 
Markov processes
 plays the role that the 
transition matrix
 does in the theory of Markov processes with a 
finite
 
state space
.
[
1
]






Formal definition
[
edit
]


Let 








(


X


,






A






)






{\displaystyle (X,{\mathcal {A}})}




 and 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




 be 
measurable spaces
. A 
Markov kernel
 with source 








(


X


,






A






)






{\displaystyle (X,{\mathcal {A}})}




 and target 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




, sometimes written as 








κ


:


(


X


,






A






)


→


(


Y


,






B






)






{\displaystyle \kappa :(X,{\mathcal {A}})\to (Y,{\mathcal {B}})}




, is a function 








κ


:






B






×


X


→


[


0


,


1


]






{\displaystyle \kappa :{\mathcal {B}}\times X\to [0,1]}




 with the following properties: 



For every (fixed) 










B




0






∈






B










{\displaystyle B_{0}\in {\mathcal {B}}}




, the map 








x


↦


κ


(




B




0






,


x


)






{\displaystyle x\mapsto \kappa (B_{0},x)}




 is 












A










{\displaystyle {\mathcal {A}}}




-
measurable


For every (fixed) 










x




0






∈


X






{\displaystyle x_{0}\in X}




, the map 








B


↦


κ


(


B


,




x




0






)






{\displaystyle B\mapsto \kappa (B,x_{0})}




 is a 
probability measure
 on 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}






In other words it associates to each point 








x


∈


X






{\displaystyle x\in X}




 a 
probability measure
 








κ


(


d


y




|




x


)


:


B


↦


κ


(


B


,


x


)






{\displaystyle \kappa (dy|x):B\mapsto \kappa (B,x)}




 on 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




 such that, for every measurable set 








B


∈






B










{\displaystyle B\in {\mathcal {B}}}




, the map 








x


↦


κ


(


B


,


x


)






{\displaystyle x\mapsto \kappa (B,x)}




 is measurable with respect to the 








σ






{\displaystyle \sigma }




-algebra 












A










{\displaystyle {\mathcal {A}}}




.
[
2
]




Examples
[
edit
]


Simple random walk
 on the integers
[
edit
]


Take 








X


=


Y


=




Z








{\displaystyle X=Y=\mathbb {Z} }




, and 












A






=






B






=






P






(




Z




)






{\displaystyle {\mathcal {A}}={\mathcal {B}}={\mathcal {P}}(\mathbb {Z} )}




 (the 
power set
 of 










Z








{\displaystyle \mathbb {Z} }




). Then a Markov kernel is fully determined by the probability it assigns to singletons 








{


m


}


,




m


∈


Y


=




Z








{\displaystyle \{m\},\,m\in Y=\mathbb {Z} }




 for each 








n


∈


X


=




Z








{\displaystyle n\in X=\mathbb {Z} }




:











κ


(


B




|




n


)


=




∑




m


∈


B






κ


(


{


m


}




|




n


)


,




∀


n


∈




Z




,




∀


B


∈






B










{\displaystyle \kappa (B|n)=\sum _{m\in B}\kappa (\{m\}|n),\qquad \forall n\in \mathbb {Z} ,\,\forall B\in {\mathcal {B}}}




.


Now the random walk  








κ






{\displaystyle \kappa }




  that goes to the right with probability 








p






{\displaystyle p}




  and to the left with probability 








1


−


p






{\displaystyle 1-p}




 is defined by 











κ


(


{


m


}




|




n


)


=


p




δ




m


,


n


+


1






+


(


1


−


p


)




δ




m


,


n


−


1






,




∀


n


,


m


∈




Z








{\displaystyle \kappa (\{m\}|n)=p\delta _{m,n+1}+(1-p)\delta _{m,n-1},\quad \forall n,m\in \mathbb {Z} }






where 








δ






{\displaystyle \delta }




 is the 
Kronecker delta
. The transition probabilities 








P


(


m




|




n


)


=


κ


(


{


m


}




|




n


)






{\displaystyle P(m|n)=\kappa (\{m\}|n)}




 for the random walk are equivalent to the Markov kernel.



General 
Markov processes
 with countable state space
[
edit
]


More generally take 








X






{\displaystyle X}




 and 








Y






{\displaystyle Y}




 both countable and 












A






=






P






(


X


)


,


 






B






=






P






(


Y


)






{\displaystyle {\mathcal {A}}={\mathcal {P}}(X),\ {\mathcal {B}}={\mathcal {P}}(Y)}




. 
Again a Markov kernel is defined by the probability it assigns to singleton sets for each 








i


∈


X






{\displaystyle i\in X}
















κ


(


B




|




i


)


=




∑




j


∈


B






κ


(


{


j


}




|




i


)


,




∀


i


∈


X


,




∀


B


∈






B










{\displaystyle \kappa (B|i)=\sum _{j\in B}\kappa (\{j\}|i),\qquad \forall i\in X,\,\forall B\in {\mathcal {B}}}




,


We define a Markov process by defining a transition probability 








P


(


j




|




i


)


=




K




j


i










{\displaystyle P(j|i)=K_{ji}}




 where the numbers 










K




j


i










{\displaystyle K_{ji}}




 define a (countable) 
stochastic matrix
 








(




K




j


i






)






{\displaystyle (K_{ji})}




 i.e. 





















K




j


i












≥


0


,








∀


(


j


,


i


)


∈


Y


×


X


,












∑




j


∈


Y








K




j


i












=


1


,








∀


i


∈


X


.














{\displaystyle {\begin{aligned}K_{ji}&\geq 0,\qquad &\forall (j,i)\in Y\times X,\\\sum _{j\in Y}K_{ji}&=1,\qquad &\forall i\in X.\\\end{aligned}}}






We then define 











κ


(


{


j


}




|




i


)


=




K




j


i






=


P


(


j




|




i


)


,




∀


i


∈


X


,




∀


B


∈






B










{\displaystyle \kappa (\{j\}|i)=K_{ji}=P(j|i),\qquad \forall i\in X,\quad \forall B\in {\mathcal {B}}}




.


Again the transition probability, the stochastic matrix and the Markov kernel are equivalent reformulations.



Markov kernel defined by a kernel function and a measure
[
edit
]


Let 








ν






{\displaystyle \nu }




 be a 
measure
 on 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




, and 








k


:


Y


×


X


→


[


0


,


∞


]






{\displaystyle k:Y\times X\to [0,\infty ]}




 a 
measurable function
 with respect to the 
product 








σ






{\displaystyle \sigma }




-algebra
 












A






⊗






B










{\displaystyle {\mathcal {A}}\otimes {\mathcal {B}}}




 such that 













∫




Y






k


(


y


,


x


)


ν


(




d




y


)


=


1


,




∀


x


∈


X






{\displaystyle \int _{Y}k(y,x)\nu (\mathrm {d} y)=1,\qquad \forall x\in X}




,


then 








κ


(


d


y




|




x


)


=


k


(


y


,


x


)


ν


(


d


y


)






{\displaystyle \kappa (dy|x)=k(y,x)\nu (dy)}




 i.e. the mapping 















{








κ


:






B






×


X


→


[


0


,


1


]










κ


(


B




|




x


)


=




∫




B






k


(


y


,


x


)


ν


(




d




y


)


















{\displaystyle {\begin{cases}\kappa :{\mathcal {B}}\times X\to [0,1]\\\kappa (B|x)=\int _{B}k(y,x)\nu (\mathrm {d} y)\end{cases}}}






defines a Markov kernel.
[
3
]
 This example generalises the countable Markov process example where 








ν






{\displaystyle \nu }




 was the 
counting measure
. Moreover it encompasses other important examples such as the convolution kernels, in particular the Markov kernels defined by the heat equation. The latter example includes the 
Gaussian kernel
 on 








X


=


Y


=




R








{\displaystyle X=Y=\mathbb {R} }




 with 








ν


(


d


x


)


=


d


x






{\displaystyle \nu (dx)=dx}




 standard Lebesgue measure and 













k




t






(


y


,


x


)


=






1








2


π






t










e




−


(


y


−


x




)




2








/




(


2




t




2






)










{\displaystyle k_{t}(y,x)={\frac {1}{{\sqrt {2\pi }}t}}e^{-(y-x)^{2}/(2t^{2})}}




.


Measurable functions
[
edit
]


Take 








(


X


,






A






)






{\displaystyle (X,{\mathcal {A}})}




 and 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




 arbitrary measurable spaces, and let 








f


:


X


→


Y






{\displaystyle f:X\to Y}




 be a measurable function. Now define 








κ


(


d


y




|




x


)


=




δ




f


(


x


)






(


d


y


)






{\displaystyle \kappa (dy|x)=\delta _{f(x)}(dy)}




 i.e. 











κ


(


B




|




x


)


=






1






B






(


f


(


x


)


)


=






1








f




−


1






(


B


)






(


x


)


=






{








1








if 




f


(


x


)


∈


B










0








otherwise




















{\displaystyle \kappa (B|x)=\mathbf {1} _{B}(f(x))=\mathbf {1} _{f^{-1}(B)}(x)={\begin{cases}1&{\text{if }}f(x)\in B\\0&{\text{otherwise}}\end{cases}}}




 for all 








B


∈






B










{\displaystyle B\in {\mathcal {B}}}




.


Note that the indicator function 












1








f




−


1






(


B


)










{\displaystyle \mathbf {1} _{f^{-1}(B)}}




 is 












A










{\displaystyle {\mathcal {A}}}




-measurable for all 








B


∈






B










{\displaystyle B\in {\mathcal {B}}}




 iff 








f






{\displaystyle f}




 is measurable.   

This example allows us to think of a Markov kernel as a generalised function with a (in general) random rather than certain value. That is, it is a 
multivalued function
 where the values are not equally weighted.



Galton–Watson process
[
edit
]


As a less obvious example, take 








X


=




N




,






A






=






P






(




N




)






{\displaystyle X=\mathbb {N} ,{\mathcal {A}}={\mathcal {P}}(\mathbb {N} )}




, and 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




 the real numbers 










R








{\displaystyle \mathbb {R} }




 with the standard sigma algebra of 
Borel sets
.  Then











κ


(


B




|




n


)


=






{












1






B






(


0


)






n


=


0










Pr


(




ξ




1






+


⋯


+




ξ




x






∈


B


)






n


≠


0


















{\displaystyle \kappa (B|n)={\begin{cases}\mathbf {1} _{B}(0)&n=0\\\Pr(\xi _{1}+\cdots +\xi _{x}\in B)&n\neq 0\\\end{cases}}}






where 








x






{\displaystyle x}




 is the number of element at the state 








n






{\displaystyle n}




, 










ξ




i










{\displaystyle \xi _{i}}




  are 
i.i.d.
 
random variables
 (usually with mean 0) and where 












1






B










{\displaystyle \mathbf {1} _{B}}




 is the indicator function. For the simple case of 
coin flips
 this models the different levels of a 
Galton board
.



Composition of Markov Kernels
[
edit
]


Given measurable spaces 








(


X


,






A






)






{\displaystyle (X,{\mathcal {A}})}




,  








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




 we consider a Markov kernel 








κ


:






B






×


X


→


[


0


,


1


]






{\displaystyle \kappa :{\mathcal {B}}\times X\to [0,1]}




 as a morphism 








κ


:


X


→


Y






{\displaystyle \kappa :X\to Y}




. Intuitively, rather than assigning to each 








x


∈


X






{\displaystyle x\in X}




 a sharply defined point 








y


∈


Y






{\displaystyle y\in Y}




 the kernel assigns a "fuzzy" point in 








Y






{\displaystyle Y}




 which is only known  with some level of uncertainty, much like actual physical measurements. If we have a third measurable space 








(


Z


,






C






)






{\displaystyle (Z,{\mathcal {C}})}




, and probability kernels 








κ


:


X


→


Y






{\displaystyle \kappa :X\to Y}




 and 








λ


:


Y


→


Z






{\displaystyle \lambda :Y\to Z}




, we can define a composition 








λ


∘


κ


:


X


→


Z






{\displaystyle \lambda \circ \kappa :X\to Z}




 by the 
Chapman-Kolmogorov equation












(


λ


∘


κ


)


(


d


z




|




x


)


=




∫




Y






λ


(


d


z




|




y


)


κ


(


d


y




|




x


)






{\displaystyle (\lambda \circ \kappa )(dz|x)=\int _{Y}\lambda (dz|y)\kappa (dy|x)}




.


The composition is associative by the Monotone Convergence Theorem and the identity function considered as a Markov kernel (i.e. the delta measure  










κ




1






(


d




x


′






|




x


)


=




δ




x






(


d




x


′




)






{\displaystyle \kappa _{1}(dx'|x)=\delta _{x}(dx')}




) is the unit for this composition. 

This composition defines the structure of a 
category
 on the measurable spaces with Markov kernels as morphisms, first defined by Lawvere,
[
4
]
 the 
category of Markov kernels
.



Probability Space defined by Probability Distribution and a Markov Kernel
[
edit
]


A composition of a probability space 








(


X


,






A






,




P




X






)






{\displaystyle (X,{\mathcal {A}},P_{X})}




 and a probability kernel 








κ


:


(


X


,






A






)


→


(


Y


,






B






)






{\displaystyle \kappa :(X,{\mathcal {A}})\to (Y,{\mathcal {B}})}




 defines a probability space 








(


Y


,






B






,




P




Y






=


κ


∘




P




X






)






{\displaystyle (Y,{\mathcal {B}},P_{Y}=\kappa \circ P_{X})}




, where the probability measure is given by













P




Y






(


B


)


=




∫




X








∫




B






κ


(


d


y




|




x


)




P




X






(


d


x


)


=




∫




X






κ


(


B




|




x


)




P




X






(


d


x


)


=






E








P




X










κ


(


B




|




⋅


)


.






{\displaystyle P_{Y}(B)=\int _{X}\int _{B}\kappa (dy|x)P_{X}(dx)=\int _{X}\kappa (B|x)P_{X}(dx)=\mathbb {E} _{P_{X}}\kappa (B|\cdot ).}






Properties
[
edit
]


Semidirect product
[
edit
]


Let 








(


X


,






A






,


P


)






{\displaystyle (X,{\mathcal {A}},P)}




 be a probability space and  








κ






{\displaystyle \kappa }




 a Markov kernel from  








(


X


,






A






)






{\displaystyle (X,{\mathcal {A}})}




 to some 








(


Y


,






B






)






{\displaystyle (Y,{\mathcal {B}})}




. Then there exists a unique measure  








Q






{\displaystyle Q}




 on  








(


X


×


Y


,






A






⊗






B






)






{\displaystyle (X\times Y,{\mathcal {A}}\otimes {\mathcal {B}})}




, such that:











Q


(


A


×


B


)


=




∫




A






κ


(


B




|




x


)




P


(


d


x


)


,




∀


A


∈






A






,




∀


B


∈






B






.






{\displaystyle Q(A\times B)=\int _{A}\kappa (B|x)\,P(dx),\quad \forall A\in {\mathcal {A}},\quad \forall B\in {\mathcal {B}}.}






Regular conditional distribution
[
edit
]


Let 








(


S


,


Y


)






{\displaystyle (S,Y)}




 be a 
Borel space
, 








X






{\displaystyle X}




 a 








(


S


,


Y


)






{\displaystyle (S,Y)}




-valued random variable on the measure space 








(


Ω


,






F






,


P


)






{\displaystyle (\Omega ,{\mathcal {F}},P)}




 and 












G






⊆






F










{\displaystyle {\mathcal {G}}\subseteq {\mathcal {F}}}




 a sub-








σ






{\displaystyle \sigma }




-algebra. Then there exists a Markov kernel 








κ






{\displaystyle \kappa }




 from 








(


Ω


,






G






)






{\displaystyle (\Omega ,{\mathcal {G}})}




 to 








(


S


,


Y


)






{\displaystyle (S,Y)}




,  such that 








κ


(


⋅


,


B


)






{\displaystyle \kappa (\cdot ,B)}




 is a version of the 
conditional expectation
 










E




[






1






{


X


∈


B


}






∣






G






]






{\displaystyle \mathbb {E} [\mathbf {1} _{\{X\in B\}}\mid {\mathcal {G}}]}




 for every 








B


∈


Y






{\displaystyle B\in Y}




, i.e.











P


(


X


∈


B


∣






G






)


=




E






[








1






{


X


∈


B


}






∣






G








]




=


κ


(


⋅


,


B


)


,




P




-a.s.








∀


B


∈






G






.






{\displaystyle P(X\in B\mid {\mathcal {G}})=\mathbb {E} \left[\mathbf {1} _{\{X\in B\}}\mid {\mathcal {G}}\right]=\kappa (\cdot ,B),\qquad P{\text{-a.s.}}\,\,\forall B\in {\mathcal {G}}.}






It is called regular conditional distribution of 








X






{\displaystyle X}




 given 












G










{\displaystyle {\mathcal {G}}}




 and is not uniquely defined.



Generalizations
[
edit
]


Transition kernels
 generalize Markov kernels in the sense that for all 








x


∈


X






{\displaystyle x\in X}




, the map











B


↦


κ


(


B




|




x


)






{\displaystyle B\mapsto \kappa (B|x)}






can be any type of (non negative) measure,  not necessarily a probability measure.



External links
[
edit
]


Markov kernel
 in 
nLab
.


References
[
edit
]






^
 
Reiss, R. D. (1993). 
A Course on Point Processes
. Springer Series in Statistics. 
doi
:
10.1007/978-1-4613-9308-5
. 
ISBN
 
978-1-4613-9310-8
.




^
 
Klenke, Achim (2014). 
Probability Theory: A Comprehensive Course
. Universitext (2 ed.). Springer. p. 180. 
doi
:
10.1007/978-1-4471-5361-0
. 
ISBN
 
978-1-4471-5360-3
.




^
 
Erhan, Cinlar (2011). 
Probability and Stochastics
. New York: Springer. pp. 37–38. 
ISBN
 
978-0-387-87858-4
.




^
 
F. W. Lawvere (1962). 
"The Category of Probabilistic Mappings"
 
(PDF)
.






Bauer, Heinz (1996), 
Probability Theory
, de Gruyter, 
ISBN
 
3-11-013935-9


§36. Kernels and semigroups of kernels


See also
[
edit
]


Category of Markov kernels










Retrieved from "
https://en.wikipedia.org/w/index.php?title=Markov_kernel&oldid=1236950192
"


Category
: 
Markov processes
Hidden categories: 
Articles with short description
Short description is different from Wikidata




Read more
## Generative Adversarial Network (GAN)























Open In App


























Share Your Experiences
Deep Learning Tutorial
Introduction to Deep Learning
Introduction to Deep Learning
Difference Between Artificial Intelligence vs Machine Learning vs Deep Learning
Basic Neural Network
Difference between ANN and BNN
Single Layer Perceptron in TensorFlow
Multi-Layer Perceptron Learning in Tensorflow
Deep Neural net with forward and back propagation from scratch - Python
Understanding Multi-Layer Feed Forward Networks
List of Deep Learning Layers
Activation Functions
Activation Functions
Types Of Activation Function in ANN
Activation Functions in Pytorch
Understanding Activation Functions in Depth
Artificial Neural Network
Artificial Neural Networks and its Applications
Gradient Descent Optimization in Tensorflow
Choose Optimal Number of Epochs to Train a Neural Network in Keras
Classification
Python | Classify Handwritten Digits with Tensorflow
Train a Deep Learning Model With Pytorch
Regression
Linear Regression using PyTorch
Linear Regression Using Tensorflow
Hyperparameter tuning
Hyperparameter tuning
Introduction to Convolution Neural Network
Introduction to Convolution Neural Network
Digital Image Processing Basics
Difference between Image Processing and Computer Vision
CNN | Introduction to Pooling Layer
CIFAR-10 Image Classification in TensorFlow
Implementation of a CNN based Image Classifier using PyTorch
Convolutional Neural Network (CNN) Architectures
Object Detection  vs Object Recognition vs Image Segmentation
YOLO v2 - Object Detection
Recurrent Neural Network
Natural Language Processing (NLP) Tutorial
Introduction to NLTK: Tokenization, Stemming, Lemmatization, POS Tagging
Word Embeddings in NLP
Introduction to Recurrent Neural Network
Recurrent Neural Networks Explanation
Sentiment Analysis with an Recurrent Neural Networks (RNN)
Short term Memory
What is LSTM - Long Short Term Memory?
Long Short Term Memory Networks Explanation
LSTM - Derivation of Back propagation through time
Text Generation using Recurrent Long Short Term Memory Network
Gated Recurrent Unit Networks
Gated Recurrent Unit Networks
ML | Text Generation using Gated Recurrent Unit Networks
Generative Learning
Autoencoders -Machine Learning
How Autoencoders works ?
Variational AutoEncoders
Contractive Autoencoder (CAE)
ML | AutoEncoder with TensorFlow 2.0
Implementing an Autoencoder in PyTorch
Generative adversarial networks
Basics of Generative Adversarial Networks (GANs)
Generative Adversarial Network (GAN)
Use Cases of Generative Adversarial Networks
Building a Generative Adversarial Network using Keras
Cycle Generative Adversarial Network (CycleGAN)
StyleGAN - Style Generative Adversarial Networks
Reinforcement Learning
Understanding Reinforcement Learning in-depth
Introduction to Thompson Sampling | Reinforcement Learning
Markov Decision Process
Bellman Equation
Meta-Learning in Machine Learning
Q-Learning in Python
Q-Learning
ML | Reinforcement Learning Algorithm : Python Implementation using Q-learning
Deep Q Learning
Deep Q-Learning
Implementing Deep Q-Learning using Tensorflow
AI Driven Snake Game using Deep Q Learning
Deep Learning Interview Questions
Machine Learning & Data Science 
Course
 




























Generative Adversarial Network (GAN)






Last Updated : 


09 Aug, 2024








 




Comments
















Improve














 








































Summarize
















Suggest changes






 






Like Article








Like


















Save


















Share
















Report
















Follow












GAN
(Generative Adversarial Network) represents a cutting-edge approach to generative modeling within deep learning, often leveraging architectures like 
convolutional neural networks
. The goal of generative modeling is to autonomously identify patterns in input data, enabling the model to produce new examples that feasibly resemble the original dataset.


This article covers everything you need to know about 
GAN, the Architecture of GAN, the Workings of GAN, and types of GAN Models, and so on.


Table of Content


What is a Generative Adversarial Network?
Types of GANs
Architecture of GANs
How does a GAN work?
Implementation of a GAN
Application Of Generative Adversarial Networks (GANs)
Advantages of GAN
Disadvantages of GAN
GAN(Generative Adversarial Network)- FAQs
What is a Generative Adversarial Network?
Generative Adversarial Networks (GANs) are a powerful class of neural networks that are used for an 
unsupervised learning
. GANs are made up of two 
neural networks
, 
a discriminator and a generator.
 They use adversarial training to produce artificial data that is identical to actual data. 


The Generator attempts to fool the Discriminator, which is tasked with accurately distinguishing between produced and genuine data, by producing random noise samples. 
Realistic, high-quality samples are produced as a result of this competitive interaction, which drives both networks toward advancement. 
GANs are proving to be highly versatile artificial intelligence tools, as evidenced by their extensive use in image synthesis, style transfer, and text-to-image synthesis. 
They have also revolutionized generative modeling.
Through adversarial training, these models engage in a competitive interplay until the generator becomes adept at creating realistic samples, fooling the discriminator approximately half the time.


Generative Adversarial Networks (GANs) can be broken down into three parts:


Generative:
 To learn a generative model, which describes how data is generated in terms of a probabilistic model.
Adversarial:
 The word adversarial refers to setting one thing up against another. This means that, in the context of GANs, the generative result is compared with the actual images in the data set. A mechanism known as a discriminator is used to apply a model that attempts to distinguish between real and fake images.
Networks:
 Use deep neural networks as artificial intelligence (AI) algorithms for training purposes.
Types of GANs
Vanilla GAN: 
This is the simplest type of GAN. Here, the Generator and the Discriminator are simple a basic 
multi-layer perceptrons
. In vanilla GAN, the algorithm is really simple, it tries to optimize the mathematical equation using 
stochastic gradient descent.
Conditional GAN (CGAN): 
CGAN
 can be described as a 
deep learning
 method in which 
some conditional parameters are put into place
. 
In CGAN, an additional parameter ‘y’ is added to the Generator for generating the corresponding data.
Labels are also put into the input to the Discriminator in order for the Discriminator to help distinguish the real data from the fake generated data.
Deep Convolutional GAN (DCGAN): 
DCGAN
 is one of the most popular and also the most successful implementations of GAN. It is composed of 
ConvNets
 in place of 
multi-layer perceptrons
. 
The ConvNets are implemented without max pooling, which is in fact replaced by convolutional stride.
 Also, the layers are not fully connected.
Laplacian Pyramid GAN (LAPGAN): 
The 
Laplacian pyramid
 is a linear invertible image representation consisting of a set of band-pass images, spaced an octave apart, plus a low-frequency residual.
This approach 
uses multiple numbers of Generator and Discriminator networks
 and different levels of the Laplacian Pyramid. 
This approach is mainly used because it produces very high-quality images. The image is down-sampled at first at each layer of the pyramid and then it is again up-scaled at each layer in a backward pass where the image acquires some noise from the Conditional GAN at these layers until it reaches its original size.
Super Resolution GAN (SRGAN): 
SRGAN
 as the name suggests is a way of designing a GAN in which a 
deep neural network
 is used along with an adversarial network in order to produce higher-resolution images. This type of GAN is particularly useful in optimally up-scaling native low-resolution images to enhance their details minimizing errors while doing so.
Architecture of GANs
A Generative Adversarial Network (GAN) is composed of two primary parts, which are the Generator and the Discriminator.


Generator Model
A key element responsible for creating fresh, accurate data in a Generative Adversarial Network (GAN) is the generator model. The generator takes random noise as input and converts it into complex data samples, such text or images. It is commonly depicted as a deep neural network. 


The training data’s underlying distribution is captured by layers of learnable parameters in its design through training. The generator adjusts its output to produce samples that closely mimic real data as it is being trained by using backpropagation to fine-tune its parameters.


The generator’s ability to generate high-quality, varied samples that can fool the discriminator is what makes it successful.


Generator Loss
The objective of the generator in a GAN is to produce synthetic samples that are realistic enough to fool the discriminator. The generator achieves this by minimizing its loss function 
[Tex]J_G[/Tex]
​. The loss is minimized when the log probability is maximized, i.e., when the discriminator is highly likely to classify the generated samples as real. The following equation is given below:


[Tex]J_{G} = -\frac{1}{m} \Sigma^m _{i=1} log D(G(z_{i}))






[/Tex]
Where, 


[Tex]J_G[/Tex]
 
measure how well the generator is fooling the discriminator.
log 
[Tex]D(G(z_i) )[/Tex]
represents log probability of the discriminator being correct for generated samples. 
The generator aims to minimize this loss, encouraging the production of samples that the discriminator classifies as real 
[Tex](log D(G(z_i))[/Tex]
, close to 1.
Discriminator Model
An artificial neural network called a discriminator model is used in Generative Adversarial Networks (GANs) to differentiate between generated and actual input. By evaluating input samples and allocating probability of authenticity, the discriminator functions as a binary classifier. 


Over time, the discriminator learns to differentiate between genuine data from the dataset and artificial samples created by the generator. This allows it to progressively hone its parameters and increase its level of proficiency. 


Convolutional layers
 or pertinent structures for other modalities are usually used in its architecture when dealing with picture data. Maximizing the discriminator’s capacity to accurately identify generated samples as fraudulent and real samples as authentic is the aim of the adversarial training procedure. The discriminator grows increasingly discriminating as a result of the generator and discriminator’s interaction, which helps the GAN produce extremely realistic-looking synthetic data overall.


Discriminator Loss
 
The discriminator reduces the negative log likelihood of correctly classifying both produced and real samples. This loss incentivizes the discriminator to accurately categorize generated samples as fake and real samples with the following equation:
[Tex]J_{D} = -\frac{1}{m} \Sigma_{i=1}^m log\; D(x_{i}) – \frac{1}{m}\Sigma_{i=1}^m log(1 – D(G(z_{i}))






[/Tex]


[Tex]J_D[/Tex]
 assesses the discriminator’s ability to discern between produced and actual samples.
The log likelihood that the discriminator will accurately categorize real data is represented by 
[Tex]logD(x_i)[/Tex]
.
The log chance that the discriminator would correctly categorize generated samples as fake is represented by 
[Tex]log⁡(1-D(G(z_i)))[/Tex]
.
The discriminator aims to reduce this loss by accurately identifying artificial and real samples.
MinMax Loss
In a Generative Adversarial Network (GAN), the minimax loss formula is provided by:


[Tex]min_{G}\;max_{D}(G,D) = [\mathbb{E}_{x∼p_{data}}[log\;D(x)] + \mathbb{E}_{z∼p_{z}(z)}[log(1 – D(g(z)))]



[/Tex]
Where,


G is generator network and is D is the discriminator network
Actual data samples obtained from the true data distribution 
[Tex]p_{data}(x)



[/Tex]
 are represented by x.
Random noise sampled from a previous distribution 
[Tex]p_z(z) [/Tex]
(usually a normal or uniform distribution) is represented by z.
D(x) represents the discriminator’s likelihood of correctly identifying actual data as real.
D(G(z)) is the likelihood that the discriminator will identify generated data coming from the generator as authentic.


How does a GAN work?
The steps involved in how a GAN works:


Initialization:
 Two neural networks are created: a Generator (G) and a Discriminator (D).
G is tasked with creating new data, like images or text, that closely resembles real data.
D acts as a critic, trying to distinguish between real data (from a training dataset) and the data generated by G.
Generator’s First Move:
 G takes a random noise vector as input. This noise vector contains random values and acts as the starting point for G’s creation process. Using its internal layers and learned patterns, G transforms the noise vector into a new data sample, like a generated image.
Discriminator’s Turn:
 D receives two kinds of inputs:
Real data samples from the training dataset.
The data samples generated by G in the previous step. D’s job is to analyze each input and determine whether it’s real data or something G cooked up. It outputs a probability score between 0 and 1. A score of 1 indicates the data is likely real, and 0 suggests it’s fake.
The Learning Process:
 Now, the adversarial part comes in:
If D correctly identifies real data as real (score close to 1) and generated data as fake (score close to 0), both G and D are rewarded to a small degree. This is because they’re both doing their jobs well.
However, the key is to continuously improve. If D consistently identifies everything correctly, it won’t learn much. So, the goal is for G to eventually trick D.
Generator’s Improvement:
When D mistakenly labels G’s creation as real (score close to 1), it’s a sign that G is on the right track. In this case, G receives a significant positive update, while D receives a penalty for being fooled. 
This feedback helps G improve its generation process to create more realistic data.
Discriminator’s Adaptation:
Conversely, if D correctly identifies G’s fake data (score close to 0), but G receives no reward, D is further strengthened in its discrimination abilities. 
This ongoing duel between G and D refines both networks over time.
As training progresses, G gets better at generating realistic data, making it harder for D to tell the difference. Ideally, G becomes so adept that D can’t reliably distinguish real from fake data. At this point, G is considered well-trained and can be used to generate new, realistic data samples.


Implementation of Generative Adversarial Network (GAN)
We will follow and understand the steps to understand how GAN is implemented:


Step1 : Importing the required libraries


Python




import
 
torch


import
 
torch.nn
 
as
 
nn


import
 
torch.optim
 
as
 
optim


import
 
torchvision


from
 
torchvision
 
import
 
datasets
,
 
transforms


import
 
matplotlib.pyplot
 
as
 
plt


import
 
numpy
 
as
 
np


# Set device


device
 
=
 
torch
.
device
(
'cuda'
 
if
 
torch
.
cuda
.
is_available
()
 
else
 
'cpu'
)




For training on the CIFAR-10 image dataset, this 
PyTorch
 module creates a Generative Adversarial Network (GAN), switching between generator and discriminator training. Visualization of the generated images occurs every tenth epoch, and the development of the GAN is tracked.


Step 2: Defining a Transform
The code uses PyTorch’s transforms to define a simple picture transforms.Compose. It normalizes and transforms photos into tensors.




Python




# Define a basic transform


transform
 
=
 
transforms
.
Compose
([


transforms
.
ToTensor
(),


transforms
.
Normalize
((
0.5
,
 
0.5
,
 
0.5
),
 
(
0.5
,
 
0.5
,
 
0.5
))


])




Step 3: Loading the Dataset
A 
CIFAR-10 dataset
 is created for training with below code, which also specifies a root directory, turns on train mode, downloads if needed, and applies the specified transform. Subsequently, it generates a 32-batch 
DataLoader
 and shuffles the training set of data.




Python




train_dataset
 
=
 
datasets
.
CIFAR10
(
root
=
'./data'
,
\
              
train
=
True
,
 
download
=
True
,
 
transform
=
transform
)


dataloader
 
=
 
torch
.
utils
.
data
.
DataLoader
(
train_dataset
,
 \
                                
batch_size
=
32
,
 
shuffle
=
True
)




 Step 4: Defining parameters to be used in later processes
A Generative Adversarial Network (GAN) is used with specified hyperparameters. 


The latent space’s dimensionality is represented by latent_dim. 
lr is the optimizer’s learning rate. 
The coefficients for the
 Adam optimizer
 are beta1 and beta2. To find the total number of training epochs, use num_epochs.


Python




# Hyperparameters


latent_dim
 
=
 
100


lr
 
=
 
0.0002


beta1
 
=
 
0.5


beta2
 
=
 
0.999


num_epochs
 
=
 
10




Step 5: Defining a Utility Class to Build the Generator
The generator architecture for a GAN in PyTorch is defined with below code. 


From 
nn.Module
, the Generator class inherits. It is comprised of a sequential model with Tanh, linear, convolutional, batch normalization, reshaping, and upsampling layers. 
The neural network synthesizes an image (img) from a latent vector (z), which is the generator’s output. 
The architecture uses a series of learned transformations to turn the initial random noise in the latent space into a meaningful image.




Python




# Define the generator


class
 
Generator
(
nn
.
Module
):


def
 
__init__
(
self
,
 
latent_dim
):


super
(
Generator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Linear
(
latent_dim
,
 
128
 
*
 
8
 
*
 
8
),


nn
.
ReLU
(),


nn
.
Unflatten
(
1
,
 
(
128
,
 
8
,
 
8
)),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
128
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
64
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Conv2d
(
64
,
 
3
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
Tanh
()


)


def
 
forward
(
self
,
 
z
):


img
 
=
 
self
.
model
(
z
)


return
 
img




Step 6: Defining a Utility Class to Build the Discriminator
The PyTorch code describes the discriminator architecture for a GAN. The class Discriminator is descended from nn.Module. It is composed of linear layers, batch normalization, 
dropout
, convolutional, 
LeakyReLU
, and sequential layers. 


An image (img) is the discriminator’s input, and its validity—the probability that the input image is real as opposed to artificial—is its output. 




Python




# Define the discriminator


class
 
Discriminator
(
nn
.
Module
):


def
 
__init__
(
self
):


super
(
Discriminator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Conv2d
(
3
,
 
32
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
32
,
 
64
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
ZeroPad2d
((
0
,
 
1
,
 
0
,
 
1
)),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
64
,
 
128
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
128
,
 
256
,
 
kernel_size
=
3
,
 
stride
=
1
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
256
,
 
momentum
=
0.8
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Flatten
(),


nn
.
Linear
(
256
 
*
 
5
 
*
 
5
,
 
1
),


nn
.
Sigmoid
()


)


def
 
forward
(
self
,
 
img
):


validity
 
=
 
self
.
model
(
img
)


return
 
validity




Step 7: Building the Generative Adversarial Network
The code snippet defines and initializes a discriminator (Discriminator) and a generator (Generator). 


The designated device (GPU if available) receives both models. 
Binary Cross Entropy Loss,
 which is frequently used for GANs, is selected as the loss function (adversarial_loss). 
For the generator (optimizer_G) and discriminator (optimizer_D), distinct Adam optimizers with predetermined learning rates and betas are also defined. 


Python




# Define the generator and discriminator


# Initialize generator and discriminator


generator
 
=
 
Generator
(
latent_dim
)
.
to
(
device
)


discriminator
 
=
 
Discriminator
()
.
to
(
device
)


# Loss function


adversarial_loss
 
=
 
nn
.
BCELoss
()


# Optimizers


optimizer_G
 
=
 
optim
.
Adam
(
generator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))


optimizer_D
 
=
 
optim
.
Adam
(
discriminator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))




Step 8: Training the Generative Adversarial Network
For a Generative Adversarial Network (GAN), the code implements the training loop. 


The training data batches are iterated through during each epoch. Whereas the generator (optimizer_G) is trained to generate realistic images that trick the discriminator, the discriminator (optimizer_D) is trained to distinguish between real and phony images. 
The generator and discriminator’s adversarial losses are computed. Model parameters are updated by means of Adam optimizers and the losses are backpropagated. 
Discriminator printing and generator losses are used to track progress. For a visual assessment of the training process, generated images are additionally saved and shown every 10 epochs.


Python




# Training loop


for
 
epoch
 
in
 
range
(
num_epochs
):


for
 
i
,
 
batch
 
in
 
enumerate
(
dataloader
):


# Convert list to tensor


real_images
 
=
 
batch
[
0
]
.
to
(
device
)


# Adversarial ground truths


valid
 
=
 
torch
.
ones
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


fake
 
=
 
torch
.
zeros
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


# Configure input


real_images
 
=
 
real_images
.
to
(
device
)


# ---------------------


#  Train Discriminator


# ---------------------


optimizer_D
.
zero_grad
()


# Sample noise as generator input


z
 
=
 
torch
.
randn
(
real_images
.
size
(
0
),
 
latent_dim
,
 
device
=
device
)


# Generate a batch of images


fake_images
 
=
 
generator
(
z
)


# Measure discriminator's ability 


# to classify real and fake images


real_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
real_images
),
 
valid
)


fake_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
fake_images
.
detach
()),
 
fake
)


d_loss
 
=
 
(
real_loss
 
+
 
fake_loss
)
 
/
 
2


# Backward pass and optimize


d_loss
.
backward
()


optimizer_D
.
step
()


# -----------------


#  Train Generator


# -----------------


optimizer_G
.
zero_grad
()


# Generate a batch of images


gen_images
 
=
 
generator
(
z
)


# Adversarial loss


g_loss
 
=
 
adversarial_loss
(
discriminator
(
gen_images
),
 
valid
)


# Backward pass and optimize


g_loss
.
backward
()


optimizer_G
.
step
()


# ---------------------


#  Progress Monitoring


# ---------------------


if
 
(
i
 
+
 
1
)
 
%
 
100
 
==
 
0
:


print
(


f
"Epoch [
{
epoch
+
1
}
/
{
num_epochs
}
]
\


                        Batch 
{
i
+
1
}
/
{
len
(
dataloader
)
}
 "


f
"Discriminator Loss: 
{
d_loss
.
item
()
:
.4f
}
 "


f
"Generator Loss: 
{
g_loss
.
item
()
:
.4f
}
"


)


# Save generated images for every epoch


if
 
(
epoch
 
+
 
1
)
 
%
 
10
 
==
 
0
:


with
 
torch
.
no_grad
():


z
 
=
 
torch
.
randn
(
16
,
 
latent_dim
,
 
device
=
device
)


generated
 
=
 
generator
(
z
)
.
detach
()
.
cpu
()


grid
 
=
 
torchvision
.
utils
.
make_grid
(
generated
,
\
                                        
nrow
=
4
,
 
normalize
=
True
)


plt
.
imshow
(
np
.
transpose
(
grid
,
 
(
1
,
 
2
,
 
0
)))


plt
.
axis
(
"off"
)


plt
.
show
()




Output:


Epoch [10/10]                        Batch 1300/1563 Discriminator Loss: 0.4473 Generator Loss: 0.9555
Epoch [10/10]                        Batch 1400/1563 Discriminator Loss: 0.6643 Generator Loss: 1.0215
Epoch [10/10]                        Batch 1500/1563 Discriminator Loss: 0.4720 Generator Loss: 2.5027
GAN Output
Application Of Generative Adversarial Networks (GANs)
GANs, or Generative Adversarial Networks, have many uses in many different fields. Here are some of the widely recognized uses of GANs:


Image Synthesis and Generation : GANs
 are often used for picture synthesis and generation tasks,  They may create fresh, lifelike pictures that mimic training data by learning the distribution that explains the dataset. The development of lifelike avatars, high-resolution photographs, and fresh artwork have all been facilitated by these types of generative networks.
Image-to-Image Translation : GANs
 may be used for problems involving image-to-image translation, where the objective is to convert an input picture from one domain to another while maintaining its key features. GANs may be used, for instance, to change pictures from day to night, transform drawings into realistic images, or change the creative style of an image.
Text-to-Image Synthesis : GANs
 have been used to create visuals from descriptions in text. GANs may produce pictures that translate to a description given a text input, such as a phrase or a caption. This application might have an impact on how realistic visual material is produced using text-based instructions.
Data Augmentation : GANs
 can augment present data and increase the robustness and generalizability of machine-learning models by creating synthetic data samples.
Data Generation for Training : GANs
 can enhance the resolution and quality of low-resolution images. By training on pairs of low-resolution and high-resolution images, GANs can generate high-resolution images from low-resolution inputs, enabling improved image quality in various applications such as medical imaging, satellite imaging, and video enhancement.
Advantages of GAN
The advantages of the GANs are as follows:


Synthetic data generation
: GANs can generate new, synthetic data that resembles some known data distribution, which can be useful for data augmentation, anomaly detection, or creative applications.
High-quality results
: GANs can produce high-quality, photorealistic results in image synthesis, video synthesis, music synthesis, and other tasks.
Unsupervised learning
: GANs can be trained without labeled data, making them suitable for unsupervised learning tasks, where labeled data is scarce or difficult to obtain.
Versatility
: GANs can be applied to a wide range of tasks, including image synthesis, text-to-image synthesis, image-to-image translation, 
anomaly detection
, 
data augmentation
, and others.
Disadvantages of GAN
The disadvantages of the GANs are as follows:


Training Instability
: GANs can be difficult to train, with the risk of instability, mode collapse, or failure to converge.
Computational Cost
: GANs can require a lot of computational resources and can be slow to train, especially for high-resolution images or large datasets.
Overfitting
: GANs can overfit the training data, producing synthetic data that is too similar to the training data and lacking diversity.
Bias and Fairness
: GANs can reflect the biases and unfairness present in the training data, leading to discriminatory or biased synthetic data.
Interpretability and Accountability
: GANs can be opaque and difficult to interpret or explain, making it challenging to ensure accountability, transparency, or fairness in their applications.
Generative Adversarial Network (GAN) – FAQs
What is a Generative Adversarial Network(GAN)?
An artificial intelligence model known as a GAN is made up of two neural networks—a discriminator and a generator—that were developed in tandem using adversarial training. The discriminator assesses the new data instances for authenticity, while the generator produces new ones.


What are the main applications of GAN?
Generating images and videos, transferring styles, enhancing data, translating images to other images, producing realistic synthetic data for machine learning model training, and super-resolution are just a few of the many uses for GANs.


What challenges do GAN face?
GANs encounter difficulties such training instability, mode collapse (when the generator generates a limited range of samples), and striking the correct balance between the discriminator and generator. It’s frequently necessary to carefully build the model architecture and tune the hyperparameters.


How are GAN evaluated?
The produced samples’ quality, diversity, and resemblance to real data are the main criteria used to assess GANs. For quantitative assessment, metrics like the Fréchet Inception Distance (FID) and Inception Score are frequently employed.


Can GAN be used for tasks other than image generation
?
Yes, different tasks can be assigned to GANs. Text, music, 3D models, and other things have all been generated with them. The usefulness of conditional GANs is expanded by enabling the creation of specific content under certain input conditions.


What are some famous architectures of GANs
?
A few well-known GAN architectures are Progressive GAN (PGAN), Wasserstein GAN (WGAN), Conditional GAN (cGAN), Deep Convolutional GAN (DCGAN), and Vanilla GAN. Each has special qualities and works best with particular kinds of data and tasks.




















R










 




Rahul_Roy
 












 
Follow
 




















 
















Improve


















Previous Article








Basics of Generative Adversarial Networks (GANs)










Next Article










Use Cases of Generative Adversarial Networks




















 
 
Please 
Login
 to comment...




















Read More








Similar Reads








What is so special about Generative Adversarial Network (GAN)


Fans are ecstatic for a variety of reasons, including the fact that GANs were the first generative algorithms to produce convincingly good results, as well as the fact that they have opened up many new research directions. In the last several years, GANs are considered to be the most prominent machine learning research, and since then, GANs have re








5 min read










Selection of GAN vs Adversarial Autoencoder models


In this article, we are going to see the selection of GAN vs Adversarial Autoencoder models. Generative Adversarial Network (GAN)The Generative Adversarial Network, or GAN, is one of the most prominent deep generative modeling methodologies right now. The primary distinction between GAN and VAE is that GAN seeks to match the pixel level distributio








6 min read










Building a Generative Adversarial Network using Keras


Prerequisites: Generative Adversarial Network This article will demonstrate how to build a Generative Adversarial Network using the Keras library. The dataset which is used is the CIFAR10 Image dataset which is preloaded into Keras. You can read about the dataset here. Step 1: Importing the required libraries import numpy as np import matplotlib.py








4 min read












Conditional Generative Adversarial Network


Imagine a situation where you can generate images of cats that match your ideal vision or a landscape that adheres to a specific artistic style. CGANs is a neural network that enables the generation of data that aligns with specific properties, which can be class labels, textual descriptions, or other traits, by harnessing the power of conditions.








13 min read










Generative Adversarial Networks (GANs) | An Introduction


Generative Adversarial Networks (GANs) was first introduced by Ian Goodfellow in 2014. GANs are a powerful class of neural networks that are used for unsupervised learning. GANs can create anything whatever you feed to them, as it Learn-Generate-Improve. To understand GANs first you must have little understanding of Convolutional Neural Networks. C








6 min read










Wasserstein Generative Adversarial Networks (WGANs) Convergence and Optimization


Wasserstein Generative Adversarial Network (WGANs) is a modification of Deep Learning GAN with few changes in the algorithm. GAN, or Generative Adversarial Network, is a way to build an accurate generative model. This network was introduced by Martin Arjovsky, Soumith Chintala, and Léon Bottou in 2017. It is widely used to generate realistic images








9 min read










Generative Adversarial Networks (GANs) in PyTorch


The aim of the article is to implement GANs architecture using PyTorch framework. The article provides comprehensive understanding of GANs in PyTorch along with in-depth explanation of the code. Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning. They consist of two neural








9 min read












Image Generation using Generative Adversarial Networks (GANs)


Generative Adversarial Networks (GANs) represent a revolutionary approach to, artificial intelligence, particularly for generating images. Introduced in 2014, GANs have significantly advanced the ability to create realistic and high-quality images from random noise. In this article, we are going to train GANs model on MNIST dataset for generating i








8 min read










Architecture of Super-Resolution Generative Adversarial Networks (SRGANs)


Super-Resolution Generative Adversarial Networks (SRGANs) are advanced deep learning models designed to upscale low-resolution images to high-resolution outputs with remarkable detail. This article aims to provide a comprehensive overview of SRGANs, focusing on their architecture, key components, and training techniques to enhance image quality. Ta








9 min read










Generative Adversarial Networks (GANs) with R


Generative Adversarial Networks (GANs) are a type of neural network architecture introduced by Ian Goodfellow and his colleagues in 2014. GANs are designed to generate new data samples that resemble a given dataset. They can produce high-quality synthetic data across various domains. Working of GANsGANs consist of two neural networks. first is the








15 min read










Generative Adversarial Networks (GANs) vs Diffusion Models


Generative Adversarial Networks (GANs) and Diffusion Models are powerful generative models designed to produce synthetic data that closely resembles real-world data. Each model has distinct architectures, strengths, and limitations, making them uniquely suited for various applications. This article aims to provide a comprehensive comparison between








5 min read










Mastering Adversarial Attacks: How One Pixel Can Fool a Neural Network


Neural networks are among the best tools for classification tasks. They power everything from image recognition to natural language processing, providing incredible accuracy and versatility. But what if I told you that you could completely undermine a neural network or trick it into making mistakes? Intrigued? Let's explore adversarial attacks and








5 min read










Deep Convolutional GAN with Keras


Deep Convolutional GAN (DCGAN) was proposed by a researcher from MIT and Facebook AI research. It is widely used in many convolution-based generation-based techniques. The focus of this paper was to make training GANs stable. Hence, they proposed some architectural changes in the computer vision problems. In this article, we will be using DCGAN on








9 min read












Understanding Auxiliary Classifier : GAN


Prerequisite: GANs(General Adversarial Networks) In this article, we will be discussing a special class conditional GAN or c-GAN known as Auxiliary Classifier GAN or AC-GAN. Before getting into that, it is important to understand what a class conditional GAN is. Class-Conditional GAN (c-GANs): c-GAN can be understood as a GAN with some conditional








4 min read










Building an Auxiliary GAN using Keras and Tensorflow


Prerequisites: Generative Adversarial Network This article will demonstrate how to build an Auxiliary Generative Adversarial Network using the Keras and TensorFlow libraries. The dataset which is used is the MNIST Image dataset pre-loaded into Keras. Step 1: Setting up the environment Step 1 : Open Anaconda prompt in Administrator mode. Step 2 : Cr








5 min read










Difference between GAN vs DCGAN.


Answer: GAN is a broader class of generative models, while DCGAN specifically refers to a type of GAN that utilizes deep convolutional neural networks for image generation.Below is a detailed comparison between GAN (Generative Adversarial Network) and DCGAN (Deep Convolutional Generative Adversarial Network): FeatureGANDCGANArchitectureGeneric arch








2 min read










Adversarial Search Algorithms


Adversarial search algorithms are the backbone of strategic decision-making in artificial intelligence, it enables the agents to navigate competitive scenarios effectively. This article offers concise yet comprehensive advantages of these algorithms from their foundational principles to practical applications. Let's uncover the strategies that driv








15+ min read












Alpha-Beta pruning in Adversarial Search Algorithms


In artificial intelligence, particularly in game playing and decision-making, adversarial search algorithms are used to model and solve problems where two or more players compete against each other. One of the most well-known techniques in this domain is alpha-beta pruning. This article explores the concept of alpha-beta pruning, its implementation








6 min read










Explain the role of minimax algorithm in adversarial search for optimal decision-making?


In the realm of artificial intelligence (AI), particularly in game theory and decision-making scenarios involving competition, the ability to predict and counteract an opponent's moves is paramount. This is where adversarial search algorithms come into play. Among the most prominent and foundational of these algorithms is the Minimax algorithm. It








11 min read










Pandas AI: The Generative AI Python Library


In the age of AI, many of our tasks have been automated especially after the launch of ChatGPT. One such tool that uses the power of ChatGPT to ease data manipulation task in Python is PandasAI. It leverages the power of ChatGPT to generate Python code and executes it. The output of the generated code is returned. Pandas AI helps performing tasks i








9 min read










The Difference Between Generative and Discriminative Machine Learning Algorithms


Machine learning algorithms allow computers to learn from data and make predictions or judgments, machine learning algorithms have revolutionized a number of sectors. Generic and discriminative algorithms are two essential strategies with various applications in the field of machine learning. We will examine the core distinctions between generative








6 min read










What is Language Revitalization in Generative AI?


Imagine a world where ancient tongues, on the brink of fading into silence, are reborn. Where stories whispered through generations find a digital echo and cultural knowledge carried in every syllable is amplified across the internet. This is the promise of language revitalization in generative AI, a revolutionary field that seeks to leverage the p








7 min read










Differences between Conversational AI and Generative AI


Artificial intelligence has evolved significantly in the past few years, making day-to-day tasks easy and efficient. Conversational AI and Generative AI are the two subsets of artificial intelligence that rapidly advancing the field of AI and have become prominent and transformative. Both technologies make use of machine learning and natural langua








8 min read












10 Best Generative AI Tools to Refine Your Content Strategy


Many of us struggle with content creation and strategy. We're good at the creative, artful side, like writing compelling stories. But the analytical, strategic part is harder. Even when we do get strategic, we spend lots of time on keyword research, topic selection, and tracking performance. AI content tools can give you an advantage on the science








9 min read










5 Top Generative AI Design Tools in 2024 [Free & Paid]


Are you ready to level up your design game? Gone are the days when designers had to sit and design creatives from scratch. With the rise of artificial intelligence and its integration with different domains, you can save a lot of time and still come up with quality output. You can use these tools in generating base designs and even assist the whole








9 min read










What is Generative AI?


Nowadays as we all know the power of Artificial Intelligence is developing day by day, and after the introduction of Generative AI is taking creativity to the next level Generative AI is a subset of Deep learning that is again a part of Artificial Intelligence.  In this article, we will explore,  What is Generative AI? Examples, Definition, Models








12 min read










What is the difference between Generative and Discriminative algorithm?


Answer: Generative algorithms model the joint probability distribution of input features and target labels, while discriminative algorithms directly learn the decision boundary between classes.Generative algorithms focus on modeling the joint probability distribution of both input features and target labels. By capturing statistical dependencies wi








2 min read












7 Best Generative AI Tools for Developers [2024]


In the rapidly evolving world of technology, generative Artificial intelligence (AI) tools for developers have become indispensable assets for innovation and efficiency. These cutting-edge tools harness the power of advanced algorithms and machine learning techniques to autonomously generate content, designs, and code, transforming the development








9 min read










Generative Modeling in TensorFlow


Generative modeling is the process of learning the underlying structure of a dataset to generate new samples that mimic the distribution of the original data. The article aims to provide a comprehensive overview of generative modelling along with the implementation leveraging the TensorFlow framework. Table of Content What are generative models and








14 min read










AI-Coustics: Fights Noisy Audio With Generative AI


Have you ever been troubled by noisy audio during a video call or interview? The constant hum of traffic, the rustle of wind, or even a bustling room can significantly degrade audio quality. For content creators, journalists, and anyone relying on clean audio recording and speech clarity in videos, these challenges can be a major source of frustrat








9 min read














Article Tags : 






AI-ML-DS






Deep Learning






Python






Python-Quizzes






Technical Scripter






python


 


+2 More






Practice Tags : 




python
python
 














Like














































































































285k+ interested Geeks 








Data Structures & Algorithms in Python - Self Paced 










Explore


































198k+ interested Geeks 








Python Full Course Online - Complete Beginner to Advanced 










Explore


































927k+ interested Geeks 








Complete Interview Preparation 










Explore














 










Explore More




























































Read more
## Generative Adversarial Network (GAN)























Open In App


























Share Your Experiences
Deep Learning Tutorial
Introduction to Deep Learning
Introduction to Deep Learning
Difference Between Artificial Intelligence vs Machine Learning vs Deep Learning
Basic Neural Network
Difference between ANN and BNN
Single Layer Perceptron in TensorFlow
Multi-Layer Perceptron Learning in Tensorflow
Deep Neural net with forward and back propagation from scratch - Python
Understanding Multi-Layer Feed Forward Networks
List of Deep Learning Layers
Activation Functions
Activation Functions
Types Of Activation Function in ANN
Activation Functions in Pytorch
Understanding Activation Functions in Depth
Artificial Neural Network
Artificial Neural Networks and its Applications
Gradient Descent Optimization in Tensorflow
Choose Optimal Number of Epochs to Train a Neural Network in Keras
Classification
Python | Classify Handwritten Digits with Tensorflow
Train a Deep Learning Model With Pytorch
Regression
Linear Regression using PyTorch
Linear Regression Using Tensorflow
Hyperparameter tuning
Hyperparameter tuning
Introduction to Convolution Neural Network
Introduction to Convolution Neural Network
Digital Image Processing Basics
Difference between Image Processing and Computer Vision
CNN | Introduction to Pooling Layer
CIFAR-10 Image Classification in TensorFlow
Implementation of a CNN based Image Classifier using PyTorch
Convolutional Neural Network (CNN) Architectures
Object Detection  vs Object Recognition vs Image Segmentation
YOLO v2 - Object Detection
Recurrent Neural Network
Natural Language Processing (NLP) Tutorial
Introduction to NLTK: Tokenization, Stemming, Lemmatization, POS Tagging
Word Embeddings in NLP
Introduction to Recurrent Neural Network
Recurrent Neural Networks Explanation
Sentiment Analysis with an Recurrent Neural Networks (RNN)
Short term Memory
What is LSTM - Long Short Term Memory?
Long Short Term Memory Networks Explanation
LSTM - Derivation of Back propagation through time
Text Generation using Recurrent Long Short Term Memory Network
Gated Recurrent Unit Networks
Gated Recurrent Unit Networks
ML | Text Generation using Gated Recurrent Unit Networks
Generative Learning
Autoencoders -Machine Learning
How Autoencoders works ?
Variational AutoEncoders
Contractive Autoencoder (CAE)
ML | AutoEncoder with TensorFlow 2.0
Implementing an Autoencoder in PyTorch
Generative adversarial networks
Basics of Generative Adversarial Networks (GANs)
Generative Adversarial Network (GAN)
Use Cases of Generative Adversarial Networks
Building a Generative Adversarial Network using Keras
Cycle Generative Adversarial Network (CycleGAN)
StyleGAN - Style Generative Adversarial Networks
Reinforcement Learning
Understanding Reinforcement Learning in-depth
Introduction to Thompson Sampling | Reinforcement Learning
Markov Decision Process
Bellman Equation
Meta-Learning in Machine Learning
Q-Learning in Python
Q-Learning
ML | Reinforcement Learning Algorithm : Python Implementation using Q-learning
Deep Q Learning
Deep Q-Learning
Implementing Deep Q-Learning using Tensorflow
AI Driven Snake Game using Deep Q Learning
Deep Learning Interview Questions
Machine Learning & Data Science 
Course
 




























Generative Adversarial Network (GAN)






Last Updated : 


09 Aug, 2024








 




Comments
















Improve














 








































Summarize
















Suggest changes






 






Like Article








Like


















Save


















Share
















Report
















Follow












GAN
(Generative Adversarial Network) represents a cutting-edge approach to generative modeling within deep learning, often leveraging architectures like 
convolutional neural networks
. The goal of generative modeling is to autonomously identify patterns in input data, enabling the model to produce new examples that feasibly resemble the original dataset.


This article covers everything you need to know about 
GAN, the Architecture of GAN, the Workings of GAN, and types of GAN Models, and so on.


Table of Content


What is a Generative Adversarial Network?
Types of GANs
Architecture of GANs
How does a GAN work?
Implementation of a GAN
Application Of Generative Adversarial Networks (GANs)
Advantages of GAN
Disadvantages of GAN
GAN(Generative Adversarial Network)- FAQs
What is a Generative Adversarial Network?
Generative Adversarial Networks (GANs) are a powerful class of neural networks that are used for an 
unsupervised learning
. GANs are made up of two 
neural networks
, 
a discriminator and a generator.
 They use adversarial training to produce artificial data that is identical to actual data. 


The Generator attempts to fool the Discriminator, which is tasked with accurately distinguishing between produced and genuine data, by producing random noise samples. 
Realistic, high-quality samples are produced as a result of this competitive interaction, which drives both networks toward advancement. 
GANs are proving to be highly versatile artificial intelligence tools, as evidenced by their extensive use in image synthesis, style transfer, and text-to-image synthesis. 
They have also revolutionized generative modeling.
Through adversarial training, these models engage in a competitive interplay until the generator becomes adept at creating realistic samples, fooling the discriminator approximately half the time.


Generative Adversarial Networks (GANs) can be broken down into three parts:


Generative:
 To learn a generative model, which describes how data is generated in terms of a probabilistic model.
Adversarial:
 The word adversarial refers to setting one thing up against another. This means that, in the context of GANs, the generative result is compared with the actual images in the data set. A mechanism known as a discriminator is used to apply a model that attempts to distinguish between real and fake images.
Networks:
 Use deep neural networks as artificial intelligence (AI) algorithms for training purposes.
Types of GANs
Vanilla GAN: 
This is the simplest type of GAN. Here, the Generator and the Discriminator are simple a basic 
multi-layer perceptrons
. In vanilla GAN, the algorithm is really simple, it tries to optimize the mathematical equation using 
stochastic gradient descent.
Conditional GAN (CGAN): 
CGAN
 can be described as a 
deep learning
 method in which 
some conditional parameters are put into place
. 
In CGAN, an additional parameter ‘y’ is added to the Generator for generating the corresponding data.
Labels are also put into the input to the Discriminator in order for the Discriminator to help distinguish the real data from the fake generated data.
Deep Convolutional GAN (DCGAN): 
DCGAN
 is one of the most popular and also the most successful implementations of GAN. It is composed of 
ConvNets
 in place of 
multi-layer perceptrons
. 
The ConvNets are implemented without max pooling, which is in fact replaced by convolutional stride.
 Also, the layers are not fully connected.
Laplacian Pyramid GAN (LAPGAN): 
The 
Laplacian pyramid
 is a linear invertible image representation consisting of a set of band-pass images, spaced an octave apart, plus a low-frequency residual.
This approach 
uses multiple numbers of Generator and Discriminator networks
 and different levels of the Laplacian Pyramid. 
This approach is mainly used because it produces very high-quality images. The image is down-sampled at first at each layer of the pyramid and then it is again up-scaled at each layer in a backward pass where the image acquires some noise from the Conditional GAN at these layers until it reaches its original size.
Super Resolution GAN (SRGAN): 
SRGAN
 as the name suggests is a way of designing a GAN in which a 
deep neural network
 is used along with an adversarial network in order to produce higher-resolution images. This type of GAN is particularly useful in optimally up-scaling native low-resolution images to enhance their details minimizing errors while doing so.
Architecture of GANs
A Generative Adversarial Network (GAN) is composed of two primary parts, which are the Generator and the Discriminator.


Generator Model
A key element responsible for creating fresh, accurate data in a Generative Adversarial Network (GAN) is the generator model. The generator takes random noise as input and converts it into complex data samples, such text or images. It is commonly depicted as a deep neural network. 


The training data’s underlying distribution is captured by layers of learnable parameters in its design through training. The generator adjusts its output to produce samples that closely mimic real data as it is being trained by using backpropagation to fine-tune its parameters.


The generator’s ability to generate high-quality, varied samples that can fool the discriminator is what makes it successful.


Generator Loss
The objective of the generator in a GAN is to produce synthetic samples that are realistic enough to fool the discriminator. The generator achieves this by minimizing its loss function 
[Tex]J_G[/Tex]
​. The loss is minimized when the log probability is maximized, i.e., when the discriminator is highly likely to classify the generated samples as real. The following equation is given below:


[Tex]J_{G} = -\frac{1}{m} \Sigma^m _{i=1} log D(G(z_{i}))






[/Tex]
Where, 


[Tex]J_G[/Tex]
 
measure how well the generator is fooling the discriminator.
log 
[Tex]D(G(z_i) )[/Tex]
represents log probability of the discriminator being correct for generated samples. 
The generator aims to minimize this loss, encouraging the production of samples that the discriminator classifies as real 
[Tex](log D(G(z_i))[/Tex]
, close to 1.
Discriminator Model
An artificial neural network called a discriminator model is used in Generative Adversarial Networks (GANs) to differentiate between generated and actual input. By evaluating input samples and allocating probability of authenticity, the discriminator functions as a binary classifier. 


Over time, the discriminator learns to differentiate between genuine data from the dataset and artificial samples created by the generator. This allows it to progressively hone its parameters and increase its level of proficiency. 


Convolutional layers
 or pertinent structures for other modalities are usually used in its architecture when dealing with picture data. Maximizing the discriminator’s capacity to accurately identify generated samples as fraudulent and real samples as authentic is the aim of the adversarial training procedure. The discriminator grows increasingly discriminating as a result of the generator and discriminator’s interaction, which helps the GAN produce extremely realistic-looking synthetic data overall.


Discriminator Loss
 
The discriminator reduces the negative log likelihood of correctly classifying both produced and real samples. This loss incentivizes the discriminator to accurately categorize generated samples as fake and real samples with the following equation:
[Tex]J_{D} = -\frac{1}{m} \Sigma_{i=1}^m log\; D(x_{i}) – \frac{1}{m}\Sigma_{i=1}^m log(1 – D(G(z_{i}))






[/Tex]


[Tex]J_D[/Tex]
 assesses the discriminator’s ability to discern between produced and actual samples.
The log likelihood that the discriminator will accurately categorize real data is represented by 
[Tex]logD(x_i)[/Tex]
.
The log chance that the discriminator would correctly categorize generated samples as fake is represented by 
[Tex]log⁡(1-D(G(z_i)))[/Tex]
.
The discriminator aims to reduce this loss by accurately identifying artificial and real samples.
MinMax Loss
In a Generative Adversarial Network (GAN), the minimax loss formula is provided by:


[Tex]min_{G}\;max_{D}(G,D) = [\mathbb{E}_{x∼p_{data}}[log\;D(x)] + \mathbb{E}_{z∼p_{z}(z)}[log(1 – D(g(z)))]



[/Tex]
Where,


G is generator network and is D is the discriminator network
Actual data samples obtained from the true data distribution 
[Tex]p_{data}(x)



[/Tex]
 are represented by x.
Random noise sampled from a previous distribution 
[Tex]p_z(z) [/Tex]
(usually a normal or uniform distribution) is represented by z.
D(x) represents the discriminator’s likelihood of correctly identifying actual data as real.
D(G(z)) is the likelihood that the discriminator will identify generated data coming from the generator as authentic.


How does a GAN work?
The steps involved in how a GAN works:


Initialization:
 Two neural networks are created: a Generator (G) and a Discriminator (D).
G is tasked with creating new data, like images or text, that closely resembles real data.
D acts as a critic, trying to distinguish between real data (from a training dataset) and the data generated by G.
Generator’s First Move:
 G takes a random noise vector as input. This noise vector contains random values and acts as the starting point for G’s creation process. Using its internal layers and learned patterns, G transforms the noise vector into a new data sample, like a generated image.
Discriminator’s Turn:
 D receives two kinds of inputs:
Real data samples from the training dataset.
The data samples generated by G in the previous step. D’s job is to analyze each input and determine whether it’s real data or something G cooked up. It outputs a probability score between 0 and 1. A score of 1 indicates the data is likely real, and 0 suggests it’s fake.
The Learning Process:
 Now, the adversarial part comes in:
If D correctly identifies real data as real (score close to 1) and generated data as fake (score close to 0), both G and D are rewarded to a small degree. This is because they’re both doing their jobs well.
However, the key is to continuously improve. If D consistently identifies everything correctly, it won’t learn much. So, the goal is for G to eventually trick D.
Generator’s Improvement:
When D mistakenly labels G’s creation as real (score close to 1), it’s a sign that G is on the right track. In this case, G receives a significant positive update, while D receives a penalty for being fooled. 
This feedback helps G improve its generation process to create more realistic data.
Discriminator’s Adaptation:
Conversely, if D correctly identifies G’s fake data (score close to 0), but G receives no reward, D is further strengthened in its discrimination abilities. 
This ongoing duel between G and D refines both networks over time.
As training progresses, G gets better at generating realistic data, making it harder for D to tell the difference. Ideally, G becomes so adept that D can’t reliably distinguish real from fake data. At this point, G is considered well-trained and can be used to generate new, realistic data samples.


Implementation of Generative Adversarial Network (GAN)
We will follow and understand the steps to understand how GAN is implemented:


Step1 : Importing the required libraries


Python




import
 
torch


import
 
torch.nn
 
as
 
nn


import
 
torch.optim
 
as
 
optim


import
 
torchvision


from
 
torchvision
 
import
 
datasets
,
 
transforms


import
 
matplotlib.pyplot
 
as
 
plt


import
 
numpy
 
as
 
np


# Set device


device
 
=
 
torch
.
device
(
'cuda'
 
if
 
torch
.
cuda
.
is_available
()
 
else
 
'cpu'
)




For training on the CIFAR-10 image dataset, this 
PyTorch
 module creates a Generative Adversarial Network (GAN), switching between generator and discriminator training. Visualization of the generated images occurs every tenth epoch, and the development of the GAN is tracked.


Step 2: Defining a Transform
The code uses PyTorch’s transforms to define a simple picture transforms.Compose. It normalizes and transforms photos into tensors.




Python




# Define a basic transform


transform
 
=
 
transforms
.
Compose
([


transforms
.
ToTensor
(),


transforms
.
Normalize
((
0.5
,
 
0.5
,
 
0.5
),
 
(
0.5
,
 
0.5
,
 
0.5
))


])




Step 3: Loading the Dataset
A 
CIFAR-10 dataset
 is created for training with below code, which also specifies a root directory, turns on train mode, downloads if needed, and applies the specified transform. Subsequently, it generates a 32-batch 
DataLoader
 and shuffles the training set of data.




Python




train_dataset
 
=
 
datasets
.
CIFAR10
(
root
=
'./data'
,
\
              
train
=
True
,
 
download
=
True
,
 
transform
=
transform
)


dataloader
 
=
 
torch
.
utils
.
data
.
DataLoader
(
train_dataset
,
 \
                                
batch_size
=
32
,
 
shuffle
=
True
)




 Step 4: Defining parameters to be used in later processes
A Generative Adversarial Network (GAN) is used with specified hyperparameters. 


The latent space’s dimensionality is represented by latent_dim. 
lr is the optimizer’s learning rate. 
The coefficients for the
 Adam optimizer
 are beta1 and beta2. To find the total number of training epochs, use num_epochs.


Python




# Hyperparameters


latent_dim
 
=
 
100


lr
 
=
 
0.0002


beta1
 
=
 
0.5


beta2
 
=
 
0.999


num_epochs
 
=
 
10




Step 5: Defining a Utility Class to Build the Generator
The generator architecture for a GAN in PyTorch is defined with below code. 


From 
nn.Module
, the Generator class inherits. It is comprised of a sequential model with Tanh, linear, convolutional, batch normalization, reshaping, and upsampling layers. 
The neural network synthesizes an image (img) from a latent vector (z), which is the generator’s output. 
The architecture uses a series of learned transformations to turn the initial random noise in the latent space into a meaningful image.




Python




# Define the generator


class
 
Generator
(
nn
.
Module
):


def
 
__init__
(
self
,
 
latent_dim
):


super
(
Generator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Linear
(
latent_dim
,
 
128
 
*
 
8
 
*
 
8
),


nn
.
ReLU
(),


nn
.
Unflatten
(
1
,
 
(
128
,
 
8
,
 
8
)),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
128
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
64
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Conv2d
(
64
,
 
3
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
Tanh
()


)


def
 
forward
(
self
,
 
z
):


img
 
=
 
self
.
model
(
z
)


return
 
img




Step 6: Defining a Utility Class to Build the Discriminator
The PyTorch code describes the discriminator architecture for a GAN. The class Discriminator is descended from nn.Module. It is composed of linear layers, batch normalization, 
dropout
, convolutional, 
LeakyReLU
, and sequential layers. 


An image (img) is the discriminator’s input, and its validity—the probability that the input image is real as opposed to artificial—is its output. 




Python




# Define the discriminator


class
 
Discriminator
(
nn
.
Module
):


def
 
__init__
(
self
):


super
(
Discriminator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Conv2d
(
3
,
 
32
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
32
,
 
64
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
ZeroPad2d
((
0
,
 
1
,
 
0
,
 
1
)),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
64
,
 
128
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
128
,
 
256
,
 
kernel_size
=
3
,
 
stride
=
1
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
256
,
 
momentum
=
0.8
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Flatten
(),


nn
.
Linear
(
256
 
*
 
5
 
*
 
5
,
 
1
),


nn
.
Sigmoid
()


)


def
 
forward
(
self
,
 
img
):


validity
 
=
 
self
.
model
(
img
)


return
 
validity




Step 7: Building the Generative Adversarial Network
The code snippet defines and initializes a discriminator (Discriminator) and a generator (Generator). 


The designated device (GPU if available) receives both models. 
Binary Cross Entropy Loss,
 which is frequently used for GANs, is selected as the loss function (adversarial_loss). 
For the generator (optimizer_G) and discriminator (optimizer_D), distinct Adam optimizers with predetermined learning rates and betas are also defined. 


Python




# Define the generator and discriminator


# Initialize generator and discriminator


generator
 
=
 
Generator
(
latent_dim
)
.
to
(
device
)


discriminator
 
=
 
Discriminator
()
.
to
(
device
)


# Loss function


adversarial_loss
 
=
 
nn
.
BCELoss
()


# Optimizers


optimizer_G
 
=
 
optim
.
Adam
(
generator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))


optimizer_D
 
=
 
optim
.
Adam
(
discriminator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))




Step 8: Training the Generative Adversarial Network
For a Generative Adversarial Network (GAN), the code implements the training loop. 


The training data batches are iterated through during each epoch. Whereas the generator (optimizer_G) is trained to generate realistic images that trick the discriminator, the discriminator (optimizer_D) is trained to distinguish between real and phony images. 
The generator and discriminator’s adversarial losses are computed. Model parameters are updated by means of Adam optimizers and the losses are backpropagated. 
Discriminator printing and generator losses are used to track progress. For a visual assessment of the training process, generated images are additionally saved and shown every 10 epochs.


Python




# Training loop


for
 
epoch
 
in
 
range
(
num_epochs
):


for
 
i
,
 
batch
 
in
 
enumerate
(
dataloader
):


# Convert list to tensor


real_images
 
=
 
batch
[
0
]
.
to
(
device
)


# Adversarial ground truths


valid
 
=
 
torch
.
ones
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


fake
 
=
 
torch
.
zeros
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


# Configure input


real_images
 
=
 
real_images
.
to
(
device
)


# ---------------------


#  Train Discriminator


# ---------------------


optimizer_D
.
zero_grad
()


# Sample noise as generator input


z
 
=
 
torch
.
randn
(
real_images
.
size
(
0
),
 
latent_dim
,
 
device
=
device
)


# Generate a batch of images


fake_images
 
=
 
generator
(
z
)


# Measure discriminator's ability 


# to classify real and fake images


real_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
real_images
),
 
valid
)


fake_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
fake_images
.
detach
()),
 
fake
)


d_loss
 
=
 
(
real_loss
 
+
 
fake_loss
)
 
/
 
2


# Backward pass and optimize


d_loss
.
backward
()


optimizer_D
.
step
()


# -----------------


#  Train Generator


# -----------------


optimizer_G
.
zero_grad
()


# Generate a batch of images


gen_images
 
=
 
generator
(
z
)


# Adversarial loss


g_loss
 
=
 
adversarial_loss
(
discriminator
(
gen_images
),
 
valid
)


# Backward pass and optimize


g_loss
.
backward
()


optimizer_G
.
step
()


# ---------------------


#  Progress Monitoring


# ---------------------


if
 
(
i
 
+
 
1
)
 
%
 
100
 
==
 
0
:


print
(


f
"Epoch [
{
epoch
+
1
}
/
{
num_epochs
}
]
\


                        Batch 
{
i
+
1
}
/
{
len
(
dataloader
)
}
 "


f
"Discriminator Loss: 
{
d_loss
.
item
()
:
.4f
}
 "


f
"Generator Loss: 
{
g_loss
.
item
()
:
.4f
}
"


)


# Save generated images for every epoch


if
 
(
epoch
 
+
 
1
)
 
%
 
10
 
==
 
0
:


with
 
torch
.
no_grad
():


z
 
=
 
torch
.
randn
(
16
,
 
latent_dim
,
 
device
=
device
)


generated
 
=
 
generator
(
z
)
.
detach
()
.
cpu
()


grid
 
=
 
torchvision
.
utils
.
make_grid
(
generated
,
\
                                        
nrow
=
4
,
 
normalize
=
True
)


plt
.
imshow
(
np
.
transpose
(
grid
,
 
(
1
,
 
2
,
 
0
)))


plt
.
axis
(
"off"
)


plt
.
show
()




Output:


Epoch [10/10]                        Batch 1300/1563 Discriminator Loss: 0.4473 Generator Loss: 0.9555
Epoch [10/10]                        Batch 1400/1563 Discriminator Loss: 0.6643 Generator Loss: 1.0215
Epoch [10/10]                        Batch 1500/1563 Discriminator Loss: 0.4720 Generator Loss: 2.5027
GAN Output
Application Of Generative Adversarial Networks (GANs)
GANs, or Generative Adversarial Networks, have many uses in many different fields. Here are some of the widely recognized uses of GANs:


Image Synthesis and Generation : GANs
 are often used for picture synthesis and generation tasks,  They may create fresh, lifelike pictures that mimic training data by learning the distribution that explains the dataset. The development of lifelike avatars, high-resolution photographs, and fresh artwork have all been facilitated by these types of generative networks.
Image-to-Image Translation : GANs
 may be used for problems involving image-to-image translation, where the objective is to convert an input picture from one domain to another while maintaining its key features. GANs may be used, for instance, to change pictures from day to night, transform drawings into realistic images, or change the creative style of an image.
Text-to-Image Synthesis : GANs
 have been used to create visuals from descriptions in text. GANs may produce pictures that translate to a description given a text input, such as a phrase or a caption. This application might have an impact on how realistic visual material is produced using text-based instructions.
Data Augmentation : GANs
 can augment present data and increase the robustness and generalizability of machine-learning models by creating synthetic data samples.
Data Generation for Training : GANs
 can enhance the resolution and quality of low-resolution images. By training on pairs of low-resolution and high-resolution images, GANs can generate high-resolution images from low-resolution inputs, enabling improved image quality in various applications such as medical imaging, satellite imaging, and video enhancement.
Advantages of GAN
The advantages of the GANs are as follows:


Synthetic data generation
: GANs can generate new, synthetic data that resembles some known data distribution, which can be useful for data augmentation, anomaly detection, or creative applications.
High-quality results
: GANs can produce high-quality, photorealistic results in image synthesis, video synthesis, music synthesis, and other tasks.
Unsupervised learning
: GANs can be trained without labeled data, making them suitable for unsupervised learning tasks, where labeled data is scarce or difficult to obtain.
Versatility
: GANs can be applied to a wide range of tasks, including image synthesis, text-to-image synthesis, image-to-image translation, 
anomaly detection
, 
data augmentation
, and others.
Disadvantages of GAN
The disadvantages of the GANs are as follows:


Training Instability
: GANs can be difficult to train, with the risk of instability, mode collapse, or failure to converge.
Computational Cost
: GANs can require a lot of computational resources and can be slow to train, especially for high-resolution images or large datasets.
Overfitting
: GANs can overfit the training data, producing synthetic data that is too similar to the training data and lacking diversity.
Bias and Fairness
: GANs can reflect the biases and unfairness present in the training data, leading to discriminatory or biased synthetic data.
Interpretability and Accountability
: GANs can be opaque and difficult to interpret or explain, making it challenging to ensure accountability, transparency, or fairness in their applications.
Generative Adversarial Network (GAN) – FAQs
What is a Generative Adversarial Network(GAN)?
An artificial intelligence model known as a GAN is made up of two neural networks—a discriminator and a generator—that were developed in tandem using adversarial training. The discriminator assesses the new data instances for authenticity, while the generator produces new ones.


What are the main applications of GAN?
Generating images and videos, transferring styles, enhancing data, translating images to other images, producing realistic synthetic data for machine learning model training, and super-resolution are just a few of the many uses for GANs.


What challenges do GAN face?
GANs encounter difficulties such training instability, mode collapse (when the generator generates a limited range of samples), and striking the correct balance between the discriminator and generator. It’s frequently necessary to carefully build the model architecture and tune the hyperparameters.


How are GAN evaluated?
The produced samples’ quality, diversity, and resemblance to real data are the main criteria used to assess GANs. For quantitative assessment, metrics like the Fréchet Inception Distance (FID) and Inception Score are frequently employed.


Can GAN be used for tasks other than image generation
?
Yes, different tasks can be assigned to GANs. Text, music, 3D models, and other things have all been generated with them. The usefulness of conditional GANs is expanded by enabling the creation of specific content under certain input conditions.


What are some famous architectures of GANs
?
A few well-known GAN architectures are Progressive GAN (PGAN), Wasserstein GAN (WGAN), Conditional GAN (cGAN), Deep Convolutional GAN (DCGAN), and Vanilla GAN. Each has special qualities and works best with particular kinds of data and tasks.




















R










 




Rahul_Roy
 












 
Follow
 




















 
















Improve


















Previous Article








Basics of Generative Adversarial Networks (GANs)










Next Article










Use Cases of Generative Adversarial Networks




















 
 
Please 
Login
 to comment...




















Read More








Similar Reads








What is so special about Generative Adversarial Network (GAN)


Fans are ecstatic for a variety of reasons, including the fact that GANs were the first generative algorithms to produce convincingly good results, as well as the fact that they have opened up many new research directions. In the last several years, GANs are considered to be the most prominent machine learning research, and since then, GANs have re








5 min read










Selection of GAN vs Adversarial Autoencoder models


In this article, we are going to see the selection of GAN vs Adversarial Autoencoder models. Generative Adversarial Network (GAN)The Generative Adversarial Network, or GAN, is one of the most prominent deep generative modeling methodologies right now. The primary distinction between GAN and VAE is that GAN seeks to match the pixel level distributio








6 min read










Building a Generative Adversarial Network using Keras


Prerequisites: Generative Adversarial Network This article will demonstrate how to build a Generative Adversarial Network using the Keras library. The dataset which is used is the CIFAR10 Image dataset which is preloaded into Keras. You can read about the dataset here. Step 1: Importing the required libraries import numpy as np import matplotlib.py








4 min read












Conditional Generative Adversarial Network


Imagine a situation where you can generate images of cats that match your ideal vision or a landscape that adheres to a specific artistic style. CGANs is a neural network that enables the generation of data that aligns with specific properties, which can be class labels, textual descriptions, or other traits, by harnessing the power of conditions.








13 min read










Generative Adversarial Networks (GANs) | An Introduction


Generative Adversarial Networks (GANs) was first introduced by Ian Goodfellow in 2014. GANs are a powerful class of neural networks that are used for unsupervised learning. GANs can create anything whatever you feed to them, as it Learn-Generate-Improve. To understand GANs first you must have little understanding of Convolutional Neural Networks. C








6 min read










Wasserstein Generative Adversarial Networks (WGANs) Convergence and Optimization


Wasserstein Generative Adversarial Network (WGANs) is a modification of Deep Learning GAN with few changes in the algorithm. GAN, or Generative Adversarial Network, is a way to build an accurate generative model. This network was introduced by Martin Arjovsky, Soumith Chintala, and Léon Bottou in 2017. It is widely used to generate realistic images








9 min read










Generative Adversarial Networks (GANs) in PyTorch


The aim of the article is to implement GANs architecture using PyTorch framework. The article provides comprehensive understanding of GANs in PyTorch along with in-depth explanation of the code. Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning. They consist of two neural








9 min read












Image Generation using Generative Adversarial Networks (GANs)


Generative Adversarial Networks (GANs) represent a revolutionary approach to, artificial intelligence, particularly for generating images. Introduced in 2014, GANs have significantly advanced the ability to create realistic and high-quality images from random noise. In this article, we are going to train GANs model on MNIST dataset for generating i








8 min read










Architecture of Super-Resolution Generative Adversarial Networks (SRGANs)


Super-Resolution Generative Adversarial Networks (SRGANs) are advanced deep learning models designed to upscale low-resolution images to high-resolution outputs with remarkable detail. This article aims to provide a comprehensive overview of SRGANs, focusing on their architecture, key components, and training techniques to enhance image quality. Ta








9 min read










Generative Adversarial Networks (GANs) with R


Generative Adversarial Networks (GANs) are a type of neural network architecture introduced by Ian Goodfellow and his colleagues in 2014. GANs are designed to generate new data samples that resemble a given dataset. They can produce high-quality synthetic data across various domains. Working of GANsGANs consist of two neural networks. first is the








15 min read










Generative Adversarial Networks (GANs) vs Diffusion Models


Generative Adversarial Networks (GANs) and Diffusion Models are powerful generative models designed to produce synthetic data that closely resembles real-world data. Each model has distinct architectures, strengths, and limitations, making them uniquely suited for various applications. This article aims to provide a comprehensive comparison between








5 min read










Mastering Adversarial Attacks: How One Pixel Can Fool a Neural Network


Neural networks are among the best tools for classification tasks. They power everything from image recognition to natural language processing, providing incredible accuracy and versatility. But what if I told you that you could completely undermine a neural network or trick it into making mistakes? Intrigued? Let's explore adversarial attacks and








5 min read










Deep Convolutional GAN with Keras


Deep Convolutional GAN (DCGAN) was proposed by a researcher from MIT and Facebook AI research. It is widely used in many convolution-based generation-based techniques. The focus of this paper was to make training GANs stable. Hence, they proposed some architectural changes in the computer vision problems. In this article, we will be using DCGAN on








9 min read












Understanding Auxiliary Classifier : GAN


Prerequisite: GANs(General Adversarial Networks) In this article, we will be discussing a special class conditional GAN or c-GAN known as Auxiliary Classifier GAN or AC-GAN. Before getting into that, it is important to understand what a class conditional GAN is. Class-Conditional GAN (c-GANs): c-GAN can be understood as a GAN with some conditional








4 min read










Building an Auxiliary GAN using Keras and Tensorflow


Prerequisites: Generative Adversarial Network This article will demonstrate how to build an Auxiliary Generative Adversarial Network using the Keras and TensorFlow libraries. The dataset which is used is the MNIST Image dataset pre-loaded into Keras. Step 1: Setting up the environment Step 1 : Open Anaconda prompt in Administrator mode. Step 2 : Cr








5 min read










Difference between GAN vs DCGAN.


Answer: GAN is a broader class of generative models, while DCGAN specifically refers to a type of GAN that utilizes deep convolutional neural networks for image generation.Below is a detailed comparison between GAN (Generative Adversarial Network) and DCGAN (Deep Convolutional Generative Adversarial Network): FeatureGANDCGANArchitectureGeneric arch








2 min read










Adversarial Search Algorithms


Adversarial search algorithms are the backbone of strategic decision-making in artificial intelligence, it enables the agents to navigate competitive scenarios effectively. This article offers concise yet comprehensive advantages of these algorithms from their foundational principles to practical applications. Let's uncover the strategies that driv








15+ min read












Alpha-Beta pruning in Adversarial Search Algorithms


In artificial intelligence, particularly in game playing and decision-making, adversarial search algorithms are used to model and solve problems where two or more players compete against each other. One of the most well-known techniques in this domain is alpha-beta pruning. This article explores the concept of alpha-beta pruning, its implementation








6 min read










Explain the role of minimax algorithm in adversarial search for optimal decision-making?


In the realm of artificial intelligence (AI), particularly in game theory and decision-making scenarios involving competition, the ability to predict and counteract an opponent's moves is paramount. This is where adversarial search algorithms come into play. Among the most prominent and foundational of these algorithms is the Minimax algorithm. It








11 min read










Pandas AI: The Generative AI Python Library


In the age of AI, many of our tasks have been automated especially after the launch of ChatGPT. One such tool that uses the power of ChatGPT to ease data manipulation task in Python is PandasAI. It leverages the power of ChatGPT to generate Python code and executes it. The output of the generated code is returned. Pandas AI helps performing tasks i








9 min read










The Difference Between Generative and Discriminative Machine Learning Algorithms


Machine learning algorithms allow computers to learn from data and make predictions or judgments, machine learning algorithms have revolutionized a number of sectors. Generic and discriminative algorithms are two essential strategies with various applications in the field of machine learning. We will examine the core distinctions between generative








6 min read










What is Language Revitalization in Generative AI?


Imagine a world where ancient tongues, on the brink of fading into silence, are reborn. Where stories whispered through generations find a digital echo and cultural knowledge carried in every syllable is amplified across the internet. This is the promise of language revitalization in generative AI, a revolutionary field that seeks to leverage the p








7 min read










Differences between Conversational AI and Generative AI


Artificial intelligence has evolved significantly in the past few years, making day-to-day tasks easy and efficient. Conversational AI and Generative AI are the two subsets of artificial intelligence that rapidly advancing the field of AI and have become prominent and transformative. Both technologies make use of machine learning and natural langua








8 min read












10 Best Generative AI Tools to Refine Your Content Strategy


Many of us struggle with content creation and strategy. We're good at the creative, artful side, like writing compelling stories. But the analytical, strategic part is harder. Even when we do get strategic, we spend lots of time on keyword research, topic selection, and tracking performance. AI content tools can give you an advantage on the science








9 min read










5 Top Generative AI Design Tools in 2024 [Free & Paid]


Are you ready to level up your design game? Gone are the days when designers had to sit and design creatives from scratch. With the rise of artificial intelligence and its integration with different domains, you can save a lot of time and still come up with quality output. You can use these tools in generating base designs and even assist the whole








9 min read










What is Generative AI?


Nowadays as we all know the power of Artificial Intelligence is developing day by day, and after the introduction of Generative AI is taking creativity to the next level Generative AI is a subset of Deep learning that is again a part of Artificial Intelligence.  In this article, we will explore,  What is Generative AI? Examples, Definition, Models








12 min read










What is the difference between Generative and Discriminative algorithm?


Answer: Generative algorithms model the joint probability distribution of input features and target labels, while discriminative algorithms directly learn the decision boundary between classes.Generative algorithms focus on modeling the joint probability distribution of both input features and target labels. By capturing statistical dependencies wi








2 min read












7 Best Generative AI Tools for Developers [2024]


In the rapidly evolving world of technology, generative Artificial intelligence (AI) tools for developers have become indispensable assets for innovation and efficiency. These cutting-edge tools harness the power of advanced algorithms and machine learning techniques to autonomously generate content, designs, and code, transforming the development








9 min read










Generative Modeling in TensorFlow


Generative modeling is the process of learning the underlying structure of a dataset to generate new samples that mimic the distribution of the original data. The article aims to provide a comprehensive overview of generative modelling along with the implementation leveraging the TensorFlow framework. Table of Content What are generative models and








14 min read










AI-Coustics: Fights Noisy Audio With Generative AI


Have you ever been troubled by noisy audio during a video call or interview? The constant hum of traffic, the rustle of wind, or even a bustling room can significantly degrade audio quality. For content creators, journalists, and anyone relying on clean audio recording and speech clarity in videos, these challenges can be a major source of frustrat








9 min read














Article Tags : 






AI-ML-DS






Deep Learning






Python






Python-Quizzes






Technical Scripter






python


 


+2 More






Practice Tags : 




python
python
 














Like














































































































285k+ interested Geeks 








Data Structures & Algorithms in Python - Self Paced 










Explore


































198k+ interested Geeks 








Python Full Course Online - Complete Beginner to Advanced 










Explore


































927k+ interested Geeks 








Complete Interview Preparation 










Explore














 










Explore More




























































Read more
## Generative Adversarial Network (GAN)























Open In App


























Share Your Experiences
Deep Learning Tutorial
Introduction to Deep Learning
Introduction to Deep Learning
Difference Between Artificial Intelligence vs Machine Learning vs Deep Learning
Basic Neural Network
Difference between ANN and BNN
Single Layer Perceptron in TensorFlow
Multi-Layer Perceptron Learning in Tensorflow
Deep Neural net with forward and back propagation from scratch - Python
Understanding Multi-Layer Feed Forward Networks
List of Deep Learning Layers
Activation Functions
Activation Functions
Types Of Activation Function in ANN
Activation Functions in Pytorch
Understanding Activation Functions in Depth
Artificial Neural Network
Artificial Neural Networks and its Applications
Gradient Descent Optimization in Tensorflow
Choose Optimal Number of Epochs to Train a Neural Network in Keras
Classification
Python | Classify Handwritten Digits with Tensorflow
Train a Deep Learning Model With Pytorch
Regression
Linear Regression using PyTorch
Linear Regression Using Tensorflow
Hyperparameter tuning
Hyperparameter tuning
Introduction to Convolution Neural Network
Introduction to Convolution Neural Network
Digital Image Processing Basics
Difference between Image Processing and Computer Vision
CNN | Introduction to Pooling Layer
CIFAR-10 Image Classification in TensorFlow
Implementation of a CNN based Image Classifier using PyTorch
Convolutional Neural Network (CNN) Architectures
Object Detection  vs Object Recognition vs Image Segmentation
YOLO v2 - Object Detection
Recurrent Neural Network
Natural Language Processing (NLP) Tutorial
Introduction to NLTK: Tokenization, Stemming, Lemmatization, POS Tagging
Word Embeddings in NLP
Introduction to Recurrent Neural Network
Recurrent Neural Networks Explanation
Sentiment Analysis with an Recurrent Neural Networks (RNN)
Short term Memory
What is LSTM - Long Short Term Memory?
Long Short Term Memory Networks Explanation
LSTM - Derivation of Back propagation through time
Text Generation using Recurrent Long Short Term Memory Network
Gated Recurrent Unit Networks
Gated Recurrent Unit Networks
ML | Text Generation using Gated Recurrent Unit Networks
Generative Learning
Autoencoders -Machine Learning
How Autoencoders works ?
Variational AutoEncoders
Contractive Autoencoder (CAE)
ML | AutoEncoder with TensorFlow 2.0
Implementing an Autoencoder in PyTorch
Generative adversarial networks
Basics of Generative Adversarial Networks (GANs)
Generative Adversarial Network (GAN)
Use Cases of Generative Adversarial Networks
Building a Generative Adversarial Network using Keras
Cycle Generative Adversarial Network (CycleGAN)
StyleGAN - Style Generative Adversarial Networks
Reinforcement Learning
Understanding Reinforcement Learning in-depth
Introduction to Thompson Sampling | Reinforcement Learning
Markov Decision Process
Bellman Equation
Meta-Learning in Machine Learning
Q-Learning in Python
Q-Learning
ML | Reinforcement Learning Algorithm : Python Implementation using Q-learning
Deep Q Learning
Deep Q-Learning
Implementing Deep Q-Learning using Tensorflow
AI Driven Snake Game using Deep Q Learning
Deep Learning Interview Questions
Machine Learning & Data Science 
Course
 




























Generative Adversarial Network (GAN)






Last Updated : 


09 Aug, 2024








 




Comments
















Improve














 








































Summarize
















Suggest changes






 






Like Article








Like


















Save


















Share
















Report
















Follow












GAN
(Generative Adversarial Network) represents a cutting-edge approach to generative modeling within deep learning, often leveraging architectures like 
convolutional neural networks
. The goal of generative modeling is to autonomously identify patterns in input data, enabling the model to produce new examples that feasibly resemble the original dataset.


This article covers everything you need to know about 
GAN, the Architecture of GAN, the Workings of GAN, and types of GAN Models, and so on.


Table of Content


What is a Generative Adversarial Network?
Types of GANs
Architecture of GANs
How does a GAN work?
Implementation of a GAN
Application Of Generative Adversarial Networks (GANs)
Advantages of GAN
Disadvantages of GAN
GAN(Generative Adversarial Network)- FAQs
What is a Generative Adversarial Network?
Generative Adversarial Networks (GANs) are a powerful class of neural networks that are used for an 
unsupervised learning
. GANs are made up of two 
neural networks
, 
a discriminator and a generator.
 They use adversarial training to produce artificial data that is identical to actual data. 


The Generator attempts to fool the Discriminator, which is tasked with accurately distinguishing between produced and genuine data, by producing random noise samples. 
Realistic, high-quality samples are produced as a result of this competitive interaction, which drives both networks toward advancement. 
GANs are proving to be highly versatile artificial intelligence tools, as evidenced by their extensive use in image synthesis, style transfer, and text-to-image synthesis. 
They have also revolutionized generative modeling.
Through adversarial training, these models engage in a competitive interplay until the generator becomes adept at creating realistic samples, fooling the discriminator approximately half the time.


Generative Adversarial Networks (GANs) can be broken down into three parts:


Generative:
 To learn a generative model, which describes how data is generated in terms of a probabilistic model.
Adversarial:
 The word adversarial refers to setting one thing up against another. This means that, in the context of GANs, the generative result is compared with the actual images in the data set. A mechanism known as a discriminator is used to apply a model that attempts to distinguish between real and fake images.
Networks:
 Use deep neural networks as artificial intelligence (AI) algorithms for training purposes.
Types of GANs
Vanilla GAN: 
This is the simplest type of GAN. Here, the Generator and the Discriminator are simple a basic 
multi-layer perceptrons
. In vanilla GAN, the algorithm is really simple, it tries to optimize the mathematical equation using 
stochastic gradient descent.
Conditional GAN (CGAN): 
CGAN
 can be described as a 
deep learning
 method in which 
some conditional parameters are put into place
. 
In CGAN, an additional parameter ‘y’ is added to the Generator for generating the corresponding data.
Labels are also put into the input to the Discriminator in order for the Discriminator to help distinguish the real data from the fake generated data.
Deep Convolutional GAN (DCGAN): 
DCGAN
 is one of the most popular and also the most successful implementations of GAN. It is composed of 
ConvNets
 in place of 
multi-layer perceptrons
. 
The ConvNets are implemented without max pooling, which is in fact replaced by convolutional stride.
 Also, the layers are not fully connected.
Laplacian Pyramid GAN (LAPGAN): 
The 
Laplacian pyramid
 is a linear invertible image representation consisting of a set of band-pass images, spaced an octave apart, plus a low-frequency residual.
This approach 
uses multiple numbers of Generator and Discriminator networks
 and different levels of the Laplacian Pyramid. 
This approach is mainly used because it produces very high-quality images. The image is down-sampled at first at each layer of the pyramid and then it is again up-scaled at each layer in a backward pass where the image acquires some noise from the Conditional GAN at these layers until it reaches its original size.
Super Resolution GAN (SRGAN): 
SRGAN
 as the name suggests is a way of designing a GAN in which a 
deep neural network
 is used along with an adversarial network in order to produce higher-resolution images. This type of GAN is particularly useful in optimally up-scaling native low-resolution images to enhance their details minimizing errors while doing so.
Architecture of GANs
A Generative Adversarial Network (GAN) is composed of two primary parts, which are the Generator and the Discriminator.


Generator Model
A key element responsible for creating fresh, accurate data in a Generative Adversarial Network (GAN) is the generator model. The generator takes random noise as input and converts it into complex data samples, such text or images. It is commonly depicted as a deep neural network. 


The training data’s underlying distribution is captured by layers of learnable parameters in its design through training. The generator adjusts its output to produce samples that closely mimic real data as it is being trained by using backpropagation to fine-tune its parameters.


The generator’s ability to generate high-quality, varied samples that can fool the discriminator is what makes it successful.


Generator Loss
The objective of the generator in a GAN is to produce synthetic samples that are realistic enough to fool the discriminator. The generator achieves this by minimizing its loss function 
[Tex]J_G[/Tex]
​. The loss is minimized when the log probability is maximized, i.e., when the discriminator is highly likely to classify the generated samples as real. The following equation is given below:


[Tex]J_{G} = -\frac{1}{m} \Sigma^m _{i=1} log D(G(z_{i}))






[/Tex]
Where, 


[Tex]J_G[/Tex]
 
measure how well the generator is fooling the discriminator.
log 
[Tex]D(G(z_i) )[/Tex]
represents log probability of the discriminator being correct for generated samples. 
The generator aims to minimize this loss, encouraging the production of samples that the discriminator classifies as real 
[Tex](log D(G(z_i))[/Tex]
, close to 1.
Discriminator Model
An artificial neural network called a discriminator model is used in Generative Adversarial Networks (GANs) to differentiate between generated and actual input. By evaluating input samples and allocating probability of authenticity, the discriminator functions as a binary classifier. 


Over time, the discriminator learns to differentiate between genuine data from the dataset and artificial samples created by the generator. This allows it to progressively hone its parameters and increase its level of proficiency. 


Convolutional layers
 or pertinent structures for other modalities are usually used in its architecture when dealing with picture data. Maximizing the discriminator’s capacity to accurately identify generated samples as fraudulent and real samples as authentic is the aim of the adversarial training procedure. The discriminator grows increasingly discriminating as a result of the generator and discriminator’s interaction, which helps the GAN produce extremely realistic-looking synthetic data overall.


Discriminator Loss
 
The discriminator reduces the negative log likelihood of correctly classifying both produced and real samples. This loss incentivizes the discriminator to accurately categorize generated samples as fake and real samples with the following equation:
[Tex]J_{D} = -\frac{1}{m} \Sigma_{i=1}^m log\; D(x_{i}) – \frac{1}{m}\Sigma_{i=1}^m log(1 – D(G(z_{i}))






[/Tex]


[Tex]J_D[/Tex]
 assesses the discriminator’s ability to discern between produced and actual samples.
The log likelihood that the discriminator will accurately categorize real data is represented by 
[Tex]logD(x_i)[/Tex]
.
The log chance that the discriminator would correctly categorize generated samples as fake is represented by 
[Tex]log⁡(1-D(G(z_i)))[/Tex]
.
The discriminator aims to reduce this loss by accurately identifying artificial and real samples.
MinMax Loss
In a Generative Adversarial Network (GAN), the minimax loss formula is provided by:


[Tex]min_{G}\;max_{D}(G,D) = [\mathbb{E}_{x∼p_{data}}[log\;D(x)] + \mathbb{E}_{z∼p_{z}(z)}[log(1 – D(g(z)))]



[/Tex]
Where,


G is generator network and is D is the discriminator network
Actual data samples obtained from the true data distribution 
[Tex]p_{data}(x)



[/Tex]
 are represented by x.
Random noise sampled from a previous distribution 
[Tex]p_z(z) [/Tex]
(usually a normal or uniform distribution) is represented by z.
D(x) represents the discriminator’s likelihood of correctly identifying actual data as real.
D(G(z)) is the likelihood that the discriminator will identify generated data coming from the generator as authentic.


How does a GAN work?
The steps involved in how a GAN works:


Initialization:
 Two neural networks are created: a Generator (G) and a Discriminator (D).
G is tasked with creating new data, like images or text, that closely resembles real data.
D acts as a critic, trying to distinguish between real data (from a training dataset) and the data generated by G.
Generator’s First Move:
 G takes a random noise vector as input. This noise vector contains random values and acts as the starting point for G’s creation process. Using its internal layers and learned patterns, G transforms the noise vector into a new data sample, like a generated image.
Discriminator’s Turn:
 D receives two kinds of inputs:
Real data samples from the training dataset.
The data samples generated by G in the previous step. D’s job is to analyze each input and determine whether it’s real data or something G cooked up. It outputs a probability score between 0 and 1. A score of 1 indicates the data is likely real, and 0 suggests it’s fake.
The Learning Process:
 Now, the adversarial part comes in:
If D correctly identifies real data as real (score close to 1) and generated data as fake (score close to 0), both G and D are rewarded to a small degree. This is because they’re both doing their jobs well.
However, the key is to continuously improve. If D consistently identifies everything correctly, it won’t learn much. So, the goal is for G to eventually trick D.
Generator’s Improvement:
When D mistakenly labels G’s creation as real (score close to 1), it’s a sign that G is on the right track. In this case, G receives a significant positive update, while D receives a penalty for being fooled. 
This feedback helps G improve its generation process to create more realistic data.
Discriminator’s Adaptation:
Conversely, if D correctly identifies G’s fake data (score close to 0), but G receives no reward, D is further strengthened in its discrimination abilities. 
This ongoing duel between G and D refines both networks over time.
As training progresses, G gets better at generating realistic data, making it harder for D to tell the difference. Ideally, G becomes so adept that D can’t reliably distinguish real from fake data. At this point, G is considered well-trained and can be used to generate new, realistic data samples.


Implementation of Generative Adversarial Network (GAN)
We will follow and understand the steps to understand how GAN is implemented:


Step1 : Importing the required libraries


Python




import
 
torch


import
 
torch.nn
 
as
 
nn


import
 
torch.optim
 
as
 
optim


import
 
torchvision


from
 
torchvision
 
import
 
datasets
,
 
transforms


import
 
matplotlib.pyplot
 
as
 
plt


import
 
numpy
 
as
 
np


# Set device


device
 
=
 
torch
.
device
(
'cuda'
 
if
 
torch
.
cuda
.
is_available
()
 
else
 
'cpu'
)




For training on the CIFAR-10 image dataset, this 
PyTorch
 module creates a Generative Adversarial Network (GAN), switching between generator and discriminator training. Visualization of the generated images occurs every tenth epoch, and the development of the GAN is tracked.


Step 2: Defining a Transform
The code uses PyTorch’s transforms to define a simple picture transforms.Compose. It normalizes and transforms photos into tensors.




Python




# Define a basic transform


transform
 
=
 
transforms
.
Compose
([


transforms
.
ToTensor
(),


transforms
.
Normalize
((
0.5
,
 
0.5
,
 
0.5
),
 
(
0.5
,
 
0.5
,
 
0.5
))


])




Step 3: Loading the Dataset
A 
CIFAR-10 dataset
 is created for training with below code, which also specifies a root directory, turns on train mode, downloads if needed, and applies the specified transform. Subsequently, it generates a 32-batch 
DataLoader
 and shuffles the training set of data.




Python




train_dataset
 
=
 
datasets
.
CIFAR10
(
root
=
'./data'
,
\
              
train
=
True
,
 
download
=
True
,
 
transform
=
transform
)


dataloader
 
=
 
torch
.
utils
.
data
.
DataLoader
(
train_dataset
,
 \
                                
batch_size
=
32
,
 
shuffle
=
True
)




 Step 4: Defining parameters to be used in later processes
A Generative Adversarial Network (GAN) is used with specified hyperparameters. 


The latent space’s dimensionality is represented by latent_dim. 
lr is the optimizer’s learning rate. 
The coefficients for the
 Adam optimizer
 are beta1 and beta2. To find the total number of training epochs, use num_epochs.


Python




# Hyperparameters


latent_dim
 
=
 
100


lr
 
=
 
0.0002


beta1
 
=
 
0.5


beta2
 
=
 
0.999


num_epochs
 
=
 
10




Step 5: Defining a Utility Class to Build the Generator
The generator architecture for a GAN in PyTorch is defined with below code. 


From 
nn.Module
, the Generator class inherits. It is comprised of a sequential model with Tanh, linear, convolutional, batch normalization, reshaping, and upsampling layers. 
The neural network synthesizes an image (img) from a latent vector (z), which is the generator’s output. 
The architecture uses a series of learned transformations to turn the initial random noise in the latent space into a meaningful image.




Python




# Define the generator


class
 
Generator
(
nn
.
Module
):


def
 
__init__
(
self
,
 
latent_dim
):


super
(
Generator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Linear
(
latent_dim
,
 
128
 
*
 
8
 
*
 
8
),


nn
.
ReLU
(),


nn
.
Unflatten
(
1
,
 
(
128
,
 
8
,
 
8
)),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
128
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
64
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Conv2d
(
64
,
 
3
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
Tanh
()


)


def
 
forward
(
self
,
 
z
):


img
 
=
 
self
.
model
(
z
)


return
 
img




Step 6: Defining a Utility Class to Build the Discriminator
The PyTorch code describes the discriminator architecture for a GAN. The class Discriminator is descended from nn.Module. It is composed of linear layers, batch normalization, 
dropout
, convolutional, 
LeakyReLU
, and sequential layers. 


An image (img) is the discriminator’s input, and its validity—the probability that the input image is real as opposed to artificial—is its output. 




Python




# Define the discriminator


class
 
Discriminator
(
nn
.
Module
):


def
 
__init__
(
self
):


super
(
Discriminator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Conv2d
(
3
,
 
32
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
32
,
 
64
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
ZeroPad2d
((
0
,
 
1
,
 
0
,
 
1
)),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
64
,
 
128
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
128
,
 
256
,
 
kernel_size
=
3
,
 
stride
=
1
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
256
,
 
momentum
=
0.8
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Flatten
(),


nn
.
Linear
(
256
 
*
 
5
 
*
 
5
,
 
1
),


nn
.
Sigmoid
()


)


def
 
forward
(
self
,
 
img
):


validity
 
=
 
self
.
model
(
img
)


return
 
validity




Step 7: Building the Generative Adversarial Network
The code snippet defines and initializes a discriminator (Discriminator) and a generator (Generator). 


The designated device (GPU if available) receives both models. 
Binary Cross Entropy Loss,
 which is frequently used for GANs, is selected as the loss function (adversarial_loss). 
For the generator (optimizer_G) and discriminator (optimizer_D), distinct Adam optimizers with predetermined learning rates and betas are also defined. 


Python




# Define the generator and discriminator


# Initialize generator and discriminator


generator
 
=
 
Generator
(
latent_dim
)
.
to
(
device
)


discriminator
 
=
 
Discriminator
()
.
to
(
device
)


# Loss function


adversarial_loss
 
=
 
nn
.
BCELoss
()


# Optimizers


optimizer_G
 
=
 
optim
.
Adam
(
generator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))


optimizer_D
 
=
 
optim
.
Adam
(
discriminator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))




Step 8: Training the Generative Adversarial Network
For a Generative Adversarial Network (GAN), the code implements the training loop. 


The training data batches are iterated through during each epoch. Whereas the generator (optimizer_G) is trained to generate realistic images that trick the discriminator, the discriminator (optimizer_D) is trained to distinguish between real and phony images. 
The generator and discriminator’s adversarial losses are computed. Model parameters are updated by means of Adam optimizers and the losses are backpropagated. 
Discriminator printing and generator losses are used to track progress. For a visual assessment of the training process, generated images are additionally saved and shown every 10 epochs.


Python




# Training loop


for
 
epoch
 
in
 
range
(
num_epochs
):


for
 
i
,
 
batch
 
in
 
enumerate
(
dataloader
):


# Convert list to tensor


real_images
 
=
 
batch
[
0
]
.
to
(
device
)


# Adversarial ground truths


valid
 
=
 
torch
.
ones
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


fake
 
=
 
torch
.
zeros
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


# Configure input


real_images
 
=
 
real_images
.
to
(
device
)


# ---------------------


#  Train Discriminator


# ---------------------


optimizer_D
.
zero_grad
()


# Sample noise as generator input


z
 
=
 
torch
.
randn
(
real_images
.
size
(
0
),
 
latent_dim
,
 
device
=
device
)


# Generate a batch of images


fake_images
 
=
 
generator
(
z
)


# Measure discriminator's ability 


# to classify real and fake images


real_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
real_images
),
 
valid
)


fake_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
fake_images
.
detach
()),
 
fake
)


d_loss
 
=
 
(
real_loss
 
+
 
fake_loss
)
 
/
 
2


# Backward pass and optimize


d_loss
.
backward
()


optimizer_D
.
step
()


# -----------------


#  Train Generator


# -----------------


optimizer_G
.
zero_grad
()


# Generate a batch of images


gen_images
 
=
 
generator
(
z
)


# Adversarial loss


g_loss
 
=
 
adversarial_loss
(
discriminator
(
gen_images
),
 
valid
)


# Backward pass and optimize


g_loss
.
backward
()


optimizer_G
.
step
()


# ---------------------


#  Progress Monitoring


# ---------------------


if
 
(
i
 
+
 
1
)
 
%
 
100
 
==
 
0
:


print
(


f
"Epoch [
{
epoch
+
1
}
/
{
num_epochs
}
]
\


                        Batch 
{
i
+
1
}
/
{
len
(
dataloader
)
}
 "


f
"Discriminator Loss: 
{
d_loss
.
item
()
:
.4f
}
 "


f
"Generator Loss: 
{
g_loss
.
item
()
:
.4f
}
"


)


# Save generated images for every epoch


if
 
(
epoch
 
+
 
1
)
 
%
 
10
 
==
 
0
:


with
 
torch
.
no_grad
():


z
 
=
 
torch
.
randn
(
16
,
 
latent_dim
,
 
device
=
device
)


generated
 
=
 
generator
(
z
)
.
detach
()
.
cpu
()


grid
 
=
 
torchvision
.
utils
.
make_grid
(
generated
,
\
                                        
nrow
=
4
,
 
normalize
=
True
)


plt
.
imshow
(
np
.
transpose
(
grid
,
 
(
1
,
 
2
,
 
0
)))


plt
.
axis
(
"off"
)


plt
.
show
()




Output:


Epoch [10/10]                        Batch 1300/1563 Discriminator Loss: 0.4473 Generator Loss: 0.9555
Epoch [10/10]                        Batch 1400/1563 Discriminator Loss: 0.6643 Generator Loss: 1.0215
Epoch [10/10]                        Batch 1500/1563 Discriminator Loss: 0.4720 Generator Loss: 2.5027
GAN Output
Application Of Generative Adversarial Networks (GANs)
GANs, or Generative Adversarial Networks, have many uses in many different fields. Here are some of the widely recognized uses of GANs:


Image Synthesis and Generation : GANs
 are often used for picture synthesis and generation tasks,  They may create fresh, lifelike pictures that mimic training data by learning the distribution that explains the dataset. The development of lifelike avatars, high-resolution photographs, and fresh artwork have all been facilitated by these types of generative networks.
Image-to-Image Translation : GANs
 may be used for problems involving image-to-image translation, where the objective is to convert an input picture from one domain to another while maintaining its key features. GANs may be used, for instance, to change pictures from day to night, transform drawings into realistic images, or change the creative style of an image.
Text-to-Image Synthesis : GANs
 have been used to create visuals from descriptions in text. GANs may produce pictures that translate to a description given a text input, such as a phrase or a caption. This application might have an impact on how realistic visual material is produced using text-based instructions.
Data Augmentation : GANs
 can augment present data and increase the robustness and generalizability of machine-learning models by creating synthetic data samples.
Data Generation for Training : GANs
 can enhance the resolution and quality of low-resolution images. By training on pairs of low-resolution and high-resolution images, GANs can generate high-resolution images from low-resolution inputs, enabling improved image quality in various applications such as medical imaging, satellite imaging, and video enhancement.
Advantages of GAN
The advantages of the GANs are as follows:


Synthetic data generation
: GANs can generate new, synthetic data that resembles some known data distribution, which can be useful for data augmentation, anomaly detection, or creative applications.
High-quality results
: GANs can produce high-quality, photorealistic results in image synthesis, video synthesis, music synthesis, and other tasks.
Unsupervised learning
: GANs can be trained without labeled data, making them suitable for unsupervised learning tasks, where labeled data is scarce or difficult to obtain.
Versatility
: GANs can be applied to a wide range of tasks, including image synthesis, text-to-image synthesis, image-to-image translation, 
anomaly detection
, 
data augmentation
, and others.
Disadvantages of GAN
The disadvantages of the GANs are as follows:


Training Instability
: GANs can be difficult to train, with the risk of instability, mode collapse, or failure to converge.
Computational Cost
: GANs can require a lot of computational resources and can be slow to train, especially for high-resolution images or large datasets.
Overfitting
: GANs can overfit the training data, producing synthetic data that is too similar to the training data and lacking diversity.
Bias and Fairness
: GANs can reflect the biases and unfairness present in the training data, leading to discriminatory or biased synthetic data.
Interpretability and Accountability
: GANs can be opaque and difficult to interpret or explain, making it challenging to ensure accountability, transparency, or fairness in their applications.
Generative Adversarial Network (GAN) – FAQs
What is a Generative Adversarial Network(GAN)?
An artificial intelligence model known as a GAN is made up of two neural networks—a discriminator and a generator—that were developed in tandem using adversarial training. The discriminator assesses the new data instances for authenticity, while the generator produces new ones.


What are the main applications of GAN?
Generating images and videos, transferring styles, enhancing data, translating images to other images, producing realistic synthetic data for machine learning model training, and super-resolution are just a few of the many uses for GANs.


What challenges do GAN face?
GANs encounter difficulties such training instability, mode collapse (when the generator generates a limited range of samples), and striking the correct balance between the discriminator and generator. It’s frequently necessary to carefully build the model architecture and tune the hyperparameters.


How are GAN evaluated?
The produced samples’ quality, diversity, and resemblance to real data are the main criteria used to assess GANs. For quantitative assessment, metrics like the Fréchet Inception Distance (FID) and Inception Score are frequently employed.


Can GAN be used for tasks other than image generation
?
Yes, different tasks can be assigned to GANs. Text, music, 3D models, and other things have all been generated with them. The usefulness of conditional GANs is expanded by enabling the creation of specific content under certain input conditions.


What are some famous architectures of GANs
?
A few well-known GAN architectures are Progressive GAN (PGAN), Wasserstein GAN (WGAN), Conditional GAN (cGAN), Deep Convolutional GAN (DCGAN), and Vanilla GAN. Each has special qualities and works best with particular kinds of data and tasks.




















R










 




Rahul_Roy
 












 
Follow
 




















 
















Improve


















Previous Article








Basics of Generative Adversarial Networks (GANs)










Next Article










Use Cases of Generative Adversarial Networks




















 
 
Please 
Login
 to comment...




















Read More








Similar Reads








What is so special about Generative Adversarial Network (GAN)


Fans are ecstatic for a variety of reasons, including the fact that GANs were the first generative algorithms to produce convincingly good results, as well as the fact that they have opened up many new research directions. In the last several years, GANs are considered to be the most prominent machine learning research, and since then, GANs have re








5 min read










Selection of GAN vs Adversarial Autoencoder models


In this article, we are going to see the selection of GAN vs Adversarial Autoencoder models. Generative Adversarial Network (GAN)The Generative Adversarial Network, or GAN, is one of the most prominent deep generative modeling methodologies right now. The primary distinction between GAN and VAE is that GAN seeks to match the pixel level distributio








6 min read










Building a Generative Adversarial Network using Keras


Prerequisites: Generative Adversarial Network This article will demonstrate how to build a Generative Adversarial Network using the Keras library. The dataset which is used is the CIFAR10 Image dataset which is preloaded into Keras. You can read about the dataset here. Step 1: Importing the required libraries import numpy as np import matplotlib.py








4 min read












Conditional Generative Adversarial Network


Imagine a situation where you can generate images of cats that match your ideal vision or a landscape that adheres to a specific artistic style. CGANs is a neural network that enables the generation of data that aligns with specific properties, which can be class labels, textual descriptions, or other traits, by harnessing the power of conditions.








13 min read










Generative Adversarial Networks (GANs) | An Introduction


Generative Adversarial Networks (GANs) was first introduced by Ian Goodfellow in 2014. GANs are a powerful class of neural networks that are used for unsupervised learning. GANs can create anything whatever you feed to them, as it Learn-Generate-Improve. To understand GANs first you must have little understanding of Convolutional Neural Networks. C








6 min read










Wasserstein Generative Adversarial Networks (WGANs) Convergence and Optimization


Wasserstein Generative Adversarial Network (WGANs) is a modification of Deep Learning GAN with few changes in the algorithm. GAN, or Generative Adversarial Network, is a way to build an accurate generative model. This network was introduced by Martin Arjovsky, Soumith Chintala, and Léon Bottou in 2017. It is widely used to generate realistic images








9 min read










Generative Adversarial Networks (GANs) in PyTorch


The aim of the article is to implement GANs architecture using PyTorch framework. The article provides comprehensive understanding of GANs in PyTorch along with in-depth explanation of the code. Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning. They consist of two neural








9 min read












Image Generation using Generative Adversarial Networks (GANs)


Generative Adversarial Networks (GANs) represent a revolutionary approach to, artificial intelligence, particularly for generating images. Introduced in 2014, GANs have significantly advanced the ability to create realistic and high-quality images from random noise. In this article, we are going to train GANs model on MNIST dataset for generating i








8 min read










Architecture of Super-Resolution Generative Adversarial Networks (SRGANs)


Super-Resolution Generative Adversarial Networks (SRGANs) are advanced deep learning models designed to upscale low-resolution images to high-resolution outputs with remarkable detail. This article aims to provide a comprehensive overview of SRGANs, focusing on their architecture, key components, and training techniques to enhance image quality. Ta








9 min read










Generative Adversarial Networks (GANs) with R


Generative Adversarial Networks (GANs) are a type of neural network architecture introduced by Ian Goodfellow and his colleagues in 2014. GANs are designed to generate new data samples that resemble a given dataset. They can produce high-quality synthetic data across various domains. Working of GANsGANs consist of two neural networks. first is the








15 min read










Generative Adversarial Networks (GANs) vs Diffusion Models


Generative Adversarial Networks (GANs) and Diffusion Models are powerful generative models designed to produce synthetic data that closely resembles real-world data. Each model has distinct architectures, strengths, and limitations, making them uniquely suited for various applications. This article aims to provide a comprehensive comparison between








5 min read










Mastering Adversarial Attacks: How One Pixel Can Fool a Neural Network


Neural networks are among the best tools for classification tasks. They power everything from image recognition to natural language processing, providing incredible accuracy and versatility. But what if I told you that you could completely undermine a neural network or trick it into making mistakes? Intrigued? Let's explore adversarial attacks and








5 min read










Deep Convolutional GAN with Keras


Deep Convolutional GAN (DCGAN) was proposed by a researcher from MIT and Facebook AI research. It is widely used in many convolution-based generation-based techniques. The focus of this paper was to make training GANs stable. Hence, they proposed some architectural changes in the computer vision problems. In this article, we will be using DCGAN on








9 min read












Understanding Auxiliary Classifier : GAN


Prerequisite: GANs(General Adversarial Networks) In this article, we will be discussing a special class conditional GAN or c-GAN known as Auxiliary Classifier GAN or AC-GAN. Before getting into that, it is important to understand what a class conditional GAN is. Class-Conditional GAN (c-GANs): c-GAN can be understood as a GAN with some conditional








4 min read










Building an Auxiliary GAN using Keras and Tensorflow


Prerequisites: Generative Adversarial Network This article will demonstrate how to build an Auxiliary Generative Adversarial Network using the Keras and TensorFlow libraries. The dataset which is used is the MNIST Image dataset pre-loaded into Keras. Step 1: Setting up the environment Step 1 : Open Anaconda prompt in Administrator mode. Step 2 : Cr








5 min read










Difference between GAN vs DCGAN.


Answer: GAN is a broader class of generative models, while DCGAN specifically refers to a type of GAN that utilizes deep convolutional neural networks for image generation.Below is a detailed comparison between GAN (Generative Adversarial Network) and DCGAN (Deep Convolutional Generative Adversarial Network): FeatureGANDCGANArchitectureGeneric arch








2 min read










Adversarial Search Algorithms


Adversarial search algorithms are the backbone of strategic decision-making in artificial intelligence, it enables the agents to navigate competitive scenarios effectively. This article offers concise yet comprehensive advantages of these algorithms from their foundational principles to practical applications. Let's uncover the strategies that driv








15+ min read












Alpha-Beta pruning in Adversarial Search Algorithms


In artificial intelligence, particularly in game playing and decision-making, adversarial search algorithms are used to model and solve problems where two or more players compete against each other. One of the most well-known techniques in this domain is alpha-beta pruning. This article explores the concept of alpha-beta pruning, its implementation








6 min read










Explain the role of minimax algorithm in adversarial search for optimal decision-making?


In the realm of artificial intelligence (AI), particularly in game theory and decision-making scenarios involving competition, the ability to predict and counteract an opponent's moves is paramount. This is where adversarial search algorithms come into play. Among the most prominent and foundational of these algorithms is the Minimax algorithm. It








11 min read










Pandas AI: The Generative AI Python Library


In the age of AI, many of our tasks have been automated especially after the launch of ChatGPT. One such tool that uses the power of ChatGPT to ease data manipulation task in Python is PandasAI. It leverages the power of ChatGPT to generate Python code and executes it. The output of the generated code is returned. Pandas AI helps performing tasks i








9 min read










The Difference Between Generative and Discriminative Machine Learning Algorithms


Machine learning algorithms allow computers to learn from data and make predictions or judgments, machine learning algorithms have revolutionized a number of sectors. Generic and discriminative algorithms are two essential strategies with various applications in the field of machine learning. We will examine the core distinctions between generative








6 min read










What is Language Revitalization in Generative AI?


Imagine a world where ancient tongues, on the brink of fading into silence, are reborn. Where stories whispered through generations find a digital echo and cultural knowledge carried in every syllable is amplified across the internet. This is the promise of language revitalization in generative AI, a revolutionary field that seeks to leverage the p








7 min read










Differences between Conversational AI and Generative AI


Artificial intelligence has evolved significantly in the past few years, making day-to-day tasks easy and efficient. Conversational AI and Generative AI are the two subsets of artificial intelligence that rapidly advancing the field of AI and have become prominent and transformative. Both technologies make use of machine learning and natural langua








8 min read












10 Best Generative AI Tools to Refine Your Content Strategy


Many of us struggle with content creation and strategy. We're good at the creative, artful side, like writing compelling stories. But the analytical, strategic part is harder. Even when we do get strategic, we spend lots of time on keyword research, topic selection, and tracking performance. AI content tools can give you an advantage on the science








9 min read










5 Top Generative AI Design Tools in 2024 [Free & Paid]


Are you ready to level up your design game? Gone are the days when designers had to sit and design creatives from scratch. With the rise of artificial intelligence and its integration with different domains, you can save a lot of time and still come up with quality output. You can use these tools in generating base designs and even assist the whole








9 min read










What is Generative AI?


Nowadays as we all know the power of Artificial Intelligence is developing day by day, and after the introduction of Generative AI is taking creativity to the next level Generative AI is a subset of Deep learning that is again a part of Artificial Intelligence.  In this article, we will explore,  What is Generative AI? Examples, Definition, Models








12 min read










What is the difference between Generative and Discriminative algorithm?


Answer: Generative algorithms model the joint probability distribution of input features and target labels, while discriminative algorithms directly learn the decision boundary between classes.Generative algorithms focus on modeling the joint probability distribution of both input features and target labels. By capturing statistical dependencies wi








2 min read












7 Best Generative AI Tools for Developers [2024]


In the rapidly evolving world of technology, generative Artificial intelligence (AI) tools for developers have become indispensable assets for innovation and efficiency. These cutting-edge tools harness the power of advanced algorithms and machine learning techniques to autonomously generate content, designs, and code, transforming the development








9 min read










Generative Modeling in TensorFlow


Generative modeling is the process of learning the underlying structure of a dataset to generate new samples that mimic the distribution of the original data. The article aims to provide a comprehensive overview of generative modelling along with the implementation leveraging the TensorFlow framework. Table of Content What are generative models and








14 min read










AI-Coustics: Fights Noisy Audio With Generative AI


Have you ever been troubled by noisy audio during a video call or interview? The constant hum of traffic, the rustle of wind, or even a bustling room can significantly degrade audio quality. For content creators, journalists, and anyone relying on clean audio recording and speech clarity in videos, these challenges can be a major source of frustrat








9 min read














Article Tags : 






AI-ML-DS






Deep Learning






Python






Python-Quizzes






Technical Scripter






python


 


+2 More






Practice Tags : 




python
python
 














Like














































































































285k+ interested Geeks 








Data Structures & Algorithms in Python - Self Paced 










Explore


































198k+ interested Geeks 








Python Full Course Online - Complete Beginner to Advanced 










Explore


































927k+ interested Geeks 








Complete Interview Preparation 










Explore














 










Explore More




























































Read more
## Generative Adversarial Network (GAN)























Open In App


























Share Your Experiences
Deep Learning Tutorial
Introduction to Deep Learning
Introduction to Deep Learning
Difference Between Artificial Intelligence vs Machine Learning vs Deep Learning
Basic Neural Network
Difference between ANN and BNN
Single Layer Perceptron in TensorFlow
Multi-Layer Perceptron Learning in Tensorflow
Deep Neural net with forward and back propagation from scratch - Python
Understanding Multi-Layer Feed Forward Networks
List of Deep Learning Layers
Activation Functions
Activation Functions
Types Of Activation Function in ANN
Activation Functions in Pytorch
Understanding Activation Functions in Depth
Artificial Neural Network
Artificial Neural Networks and its Applications
Gradient Descent Optimization in Tensorflow
Choose Optimal Number of Epochs to Train a Neural Network in Keras
Classification
Python | Classify Handwritten Digits with Tensorflow
Train a Deep Learning Model With Pytorch
Regression
Linear Regression using PyTorch
Linear Regression Using Tensorflow
Hyperparameter tuning
Hyperparameter tuning
Introduction to Convolution Neural Network
Introduction to Convolution Neural Network
Digital Image Processing Basics
Difference between Image Processing and Computer Vision
CNN | Introduction to Pooling Layer
CIFAR-10 Image Classification in TensorFlow
Implementation of a CNN based Image Classifier using PyTorch
Convolutional Neural Network (CNN) Architectures
Object Detection  vs Object Recognition vs Image Segmentation
YOLO v2 - Object Detection
Recurrent Neural Network
Natural Language Processing (NLP) Tutorial
Introduction to NLTK: Tokenization, Stemming, Lemmatization, POS Tagging
Word Embeddings in NLP
Introduction to Recurrent Neural Network
Recurrent Neural Networks Explanation
Sentiment Analysis with an Recurrent Neural Networks (RNN)
Short term Memory
What is LSTM - Long Short Term Memory?
Long Short Term Memory Networks Explanation
LSTM - Derivation of Back propagation through time
Text Generation using Recurrent Long Short Term Memory Network
Gated Recurrent Unit Networks
Gated Recurrent Unit Networks
ML | Text Generation using Gated Recurrent Unit Networks
Generative Learning
Autoencoders -Machine Learning
How Autoencoders works ?
Variational AutoEncoders
Contractive Autoencoder (CAE)
ML | AutoEncoder with TensorFlow 2.0
Implementing an Autoencoder in PyTorch
Generative adversarial networks
Basics of Generative Adversarial Networks (GANs)
Generative Adversarial Network (GAN)
Use Cases of Generative Adversarial Networks
Building a Generative Adversarial Network using Keras
Cycle Generative Adversarial Network (CycleGAN)
StyleGAN - Style Generative Adversarial Networks
Reinforcement Learning
Understanding Reinforcement Learning in-depth
Introduction to Thompson Sampling | Reinforcement Learning
Markov Decision Process
Bellman Equation
Meta-Learning in Machine Learning
Q-Learning in Python
Q-Learning
ML | Reinforcement Learning Algorithm : Python Implementation using Q-learning
Deep Q Learning
Deep Q-Learning
Implementing Deep Q-Learning using Tensorflow
AI Driven Snake Game using Deep Q Learning
Deep Learning Interview Questions
Machine Learning & Data Science 
Course
 




























Generative Adversarial Network (GAN)






Last Updated : 


09 Aug, 2024








 




Comments
















Improve














 








































Summarize
















Suggest changes






 






Like Article








Like


















Save


















Share
















Report
















Follow












GAN
(Generative Adversarial Network) represents a cutting-edge approach to generative modeling within deep learning, often leveraging architectures like 
convolutional neural networks
. The goal of generative modeling is to autonomously identify patterns in input data, enabling the model to produce new examples that feasibly resemble the original dataset.


This article covers everything you need to know about 
GAN, the Architecture of GAN, the Workings of GAN, and types of GAN Models, and so on.


Table of Content


What is a Generative Adversarial Network?
Types of GANs
Architecture of GANs
How does a GAN work?
Implementation of a GAN
Application Of Generative Adversarial Networks (GANs)
Advantages of GAN
Disadvantages of GAN
GAN(Generative Adversarial Network)- FAQs
What is a Generative Adversarial Network?
Generative Adversarial Networks (GANs) are a powerful class of neural networks that are used for an 
unsupervised learning
. GANs are made up of two 
neural networks
, 
a discriminator and a generator.
 They use adversarial training to produce artificial data that is identical to actual data. 


The Generator attempts to fool the Discriminator, which is tasked with accurately distinguishing between produced and genuine data, by producing random noise samples. 
Realistic, high-quality samples are produced as a result of this competitive interaction, which drives both networks toward advancement. 
GANs are proving to be highly versatile artificial intelligence tools, as evidenced by their extensive use in image synthesis, style transfer, and text-to-image synthesis. 
They have also revolutionized generative modeling.
Through adversarial training, these models engage in a competitive interplay until the generator becomes adept at creating realistic samples, fooling the discriminator approximately half the time.


Generative Adversarial Networks (GANs) can be broken down into three parts:


Generative:
 To learn a generative model, which describes how data is generated in terms of a probabilistic model.
Adversarial:
 The word adversarial refers to setting one thing up against another. This means that, in the context of GANs, the generative result is compared with the actual images in the data set. A mechanism known as a discriminator is used to apply a model that attempts to distinguish between real and fake images.
Networks:
 Use deep neural networks as artificial intelligence (AI) algorithms for training purposes.
Types of GANs
Vanilla GAN: 
This is the simplest type of GAN. Here, the Generator and the Discriminator are simple a basic 
multi-layer perceptrons
. In vanilla GAN, the algorithm is really simple, it tries to optimize the mathematical equation using 
stochastic gradient descent.
Conditional GAN (CGAN): 
CGAN
 can be described as a 
deep learning
 method in which 
some conditional parameters are put into place
. 
In CGAN, an additional parameter ‘y’ is added to the Generator for generating the corresponding data.
Labels are also put into the input to the Discriminator in order for the Discriminator to help distinguish the real data from the fake generated data.
Deep Convolutional GAN (DCGAN): 
DCGAN
 is one of the most popular and also the most successful implementations of GAN. It is composed of 
ConvNets
 in place of 
multi-layer perceptrons
. 
The ConvNets are implemented without max pooling, which is in fact replaced by convolutional stride.
 Also, the layers are not fully connected.
Laplacian Pyramid GAN (LAPGAN): 
The 
Laplacian pyramid
 is a linear invertible image representation consisting of a set of band-pass images, spaced an octave apart, plus a low-frequency residual.
This approach 
uses multiple numbers of Generator and Discriminator networks
 and different levels of the Laplacian Pyramid. 
This approach is mainly used because it produces very high-quality images. The image is down-sampled at first at each layer of the pyramid and then it is again up-scaled at each layer in a backward pass where the image acquires some noise from the Conditional GAN at these layers until it reaches its original size.
Super Resolution GAN (SRGAN): 
SRGAN
 as the name suggests is a way of designing a GAN in which a 
deep neural network
 is used along with an adversarial network in order to produce higher-resolution images. This type of GAN is particularly useful in optimally up-scaling native low-resolution images to enhance their details minimizing errors while doing so.
Architecture of GANs
A Generative Adversarial Network (GAN) is composed of two primary parts, which are the Generator and the Discriminator.


Generator Model
A key element responsible for creating fresh, accurate data in a Generative Adversarial Network (GAN) is the generator model. The generator takes random noise as input and converts it into complex data samples, such text or images. It is commonly depicted as a deep neural network. 


The training data’s underlying distribution is captured by layers of learnable parameters in its design through training. The generator adjusts its output to produce samples that closely mimic real data as it is being trained by using backpropagation to fine-tune its parameters.


The generator’s ability to generate high-quality, varied samples that can fool the discriminator is what makes it successful.


Generator Loss
The objective of the generator in a GAN is to produce synthetic samples that are realistic enough to fool the discriminator. The generator achieves this by minimizing its loss function 
[Tex]J_G[/Tex]
​. The loss is minimized when the log probability is maximized, i.e., when the discriminator is highly likely to classify the generated samples as real. The following equation is given below:


[Tex]J_{G} = -\frac{1}{m} \Sigma^m _{i=1} log D(G(z_{i}))






[/Tex]
Where, 


[Tex]J_G[/Tex]
 
measure how well the generator is fooling the discriminator.
log 
[Tex]D(G(z_i) )[/Tex]
represents log probability of the discriminator being correct for generated samples. 
The generator aims to minimize this loss, encouraging the production of samples that the discriminator classifies as real 
[Tex](log D(G(z_i))[/Tex]
, close to 1.
Discriminator Model
An artificial neural network called a discriminator model is used in Generative Adversarial Networks (GANs) to differentiate between generated and actual input. By evaluating input samples and allocating probability of authenticity, the discriminator functions as a binary classifier. 


Over time, the discriminator learns to differentiate between genuine data from the dataset and artificial samples created by the generator. This allows it to progressively hone its parameters and increase its level of proficiency. 


Convolutional layers
 or pertinent structures for other modalities are usually used in its architecture when dealing with picture data. Maximizing the discriminator’s capacity to accurately identify generated samples as fraudulent and real samples as authentic is the aim of the adversarial training procedure. The discriminator grows increasingly discriminating as a result of the generator and discriminator’s interaction, which helps the GAN produce extremely realistic-looking synthetic data overall.


Discriminator Loss
 
The discriminator reduces the negative log likelihood of correctly classifying both produced and real samples. This loss incentivizes the discriminator to accurately categorize generated samples as fake and real samples with the following equation:
[Tex]J_{D} = -\frac{1}{m} \Sigma_{i=1}^m log\; D(x_{i}) – \frac{1}{m}\Sigma_{i=1}^m log(1 – D(G(z_{i}))






[/Tex]


[Tex]J_D[/Tex]
 assesses the discriminator’s ability to discern between produced and actual samples.
The log likelihood that the discriminator will accurately categorize real data is represented by 
[Tex]logD(x_i)[/Tex]
.
The log chance that the discriminator would correctly categorize generated samples as fake is represented by 
[Tex]log⁡(1-D(G(z_i)))[/Tex]
.
The discriminator aims to reduce this loss by accurately identifying artificial and real samples.
MinMax Loss
In a Generative Adversarial Network (GAN), the minimax loss formula is provided by:


[Tex]min_{G}\;max_{D}(G,D) = [\mathbb{E}_{x∼p_{data}}[log\;D(x)] + \mathbb{E}_{z∼p_{z}(z)}[log(1 – D(g(z)))]



[/Tex]
Where,


G is generator network and is D is the discriminator network
Actual data samples obtained from the true data distribution 
[Tex]p_{data}(x)



[/Tex]
 are represented by x.
Random noise sampled from a previous distribution 
[Tex]p_z(z) [/Tex]
(usually a normal or uniform distribution) is represented by z.
D(x) represents the discriminator’s likelihood of correctly identifying actual data as real.
D(G(z)) is the likelihood that the discriminator will identify generated data coming from the generator as authentic.


How does a GAN work?
The steps involved in how a GAN works:


Initialization:
 Two neural networks are created: a Generator (G) and a Discriminator (D).
G is tasked with creating new data, like images or text, that closely resembles real data.
D acts as a critic, trying to distinguish between real data (from a training dataset) and the data generated by G.
Generator’s First Move:
 G takes a random noise vector as input. This noise vector contains random values and acts as the starting point for G’s creation process. Using its internal layers and learned patterns, G transforms the noise vector into a new data sample, like a generated image.
Discriminator’s Turn:
 D receives two kinds of inputs:
Real data samples from the training dataset.
The data samples generated by G in the previous step. D’s job is to analyze each input and determine whether it’s real data or something G cooked up. It outputs a probability score between 0 and 1. A score of 1 indicates the data is likely real, and 0 suggests it’s fake.
The Learning Process:
 Now, the adversarial part comes in:
If D correctly identifies real data as real (score close to 1) and generated data as fake (score close to 0), both G and D are rewarded to a small degree. This is because they’re both doing their jobs well.
However, the key is to continuously improve. If D consistently identifies everything correctly, it won’t learn much. So, the goal is for G to eventually trick D.
Generator’s Improvement:
When D mistakenly labels G’s creation as real (score close to 1), it’s a sign that G is on the right track. In this case, G receives a significant positive update, while D receives a penalty for being fooled. 
This feedback helps G improve its generation process to create more realistic data.
Discriminator’s Adaptation:
Conversely, if D correctly identifies G’s fake data (score close to 0), but G receives no reward, D is further strengthened in its discrimination abilities. 
This ongoing duel between G and D refines both networks over time.
As training progresses, G gets better at generating realistic data, making it harder for D to tell the difference. Ideally, G becomes so adept that D can’t reliably distinguish real from fake data. At this point, G is considered well-trained and can be used to generate new, realistic data samples.


Implementation of Generative Adversarial Network (GAN)
We will follow and understand the steps to understand how GAN is implemented:


Step1 : Importing the required libraries


Python




import
 
torch


import
 
torch.nn
 
as
 
nn


import
 
torch.optim
 
as
 
optim


import
 
torchvision


from
 
torchvision
 
import
 
datasets
,
 
transforms


import
 
matplotlib.pyplot
 
as
 
plt


import
 
numpy
 
as
 
np


# Set device


device
 
=
 
torch
.
device
(
'cuda'
 
if
 
torch
.
cuda
.
is_available
()
 
else
 
'cpu'
)




For training on the CIFAR-10 image dataset, this 
PyTorch
 module creates a Generative Adversarial Network (GAN), switching between generator and discriminator training. Visualization of the generated images occurs every tenth epoch, and the development of the GAN is tracked.


Step 2: Defining a Transform
The code uses PyTorch’s transforms to define a simple picture transforms.Compose. It normalizes and transforms photos into tensors.




Python




# Define a basic transform


transform
 
=
 
transforms
.
Compose
([


transforms
.
ToTensor
(),


transforms
.
Normalize
((
0.5
,
 
0.5
,
 
0.5
),
 
(
0.5
,
 
0.5
,
 
0.5
))


])




Step 3: Loading the Dataset
A 
CIFAR-10 dataset
 is created for training with below code, which also specifies a root directory, turns on train mode, downloads if needed, and applies the specified transform. Subsequently, it generates a 32-batch 
DataLoader
 and shuffles the training set of data.




Python




train_dataset
 
=
 
datasets
.
CIFAR10
(
root
=
'./data'
,
\
              
train
=
True
,
 
download
=
True
,
 
transform
=
transform
)


dataloader
 
=
 
torch
.
utils
.
data
.
DataLoader
(
train_dataset
,
 \
                                
batch_size
=
32
,
 
shuffle
=
True
)




 Step 4: Defining parameters to be used in later processes
A Generative Adversarial Network (GAN) is used with specified hyperparameters. 


The latent space’s dimensionality is represented by latent_dim. 
lr is the optimizer’s learning rate. 
The coefficients for the
 Adam optimizer
 are beta1 and beta2. To find the total number of training epochs, use num_epochs.


Python




# Hyperparameters


latent_dim
 
=
 
100


lr
 
=
 
0.0002


beta1
 
=
 
0.5


beta2
 
=
 
0.999


num_epochs
 
=
 
10




Step 5: Defining a Utility Class to Build the Generator
The generator architecture for a GAN in PyTorch is defined with below code. 


From 
nn.Module
, the Generator class inherits. It is comprised of a sequential model with Tanh, linear, convolutional, batch normalization, reshaping, and upsampling layers. 
The neural network synthesizes an image (img) from a latent vector (z), which is the generator’s output. 
The architecture uses a series of learned transformations to turn the initial random noise in the latent space into a meaningful image.




Python




# Define the generator


class
 
Generator
(
nn
.
Module
):


def
 
__init__
(
self
,
 
latent_dim
):


super
(
Generator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Linear
(
latent_dim
,
 
128
 
*
 
8
 
*
 
8
),


nn
.
ReLU
(),


nn
.
Unflatten
(
1
,
 
(
128
,
 
8
,
 
8
)),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
128
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Upsample
(
scale_factor
=
2
),


nn
.
Conv2d
(
128
,
 
64
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.78
),


nn
.
ReLU
(),


nn
.
Conv2d
(
64
,
 
3
,
 
kernel_size
=
3
,
 
padding
=
1
),


nn
.
Tanh
()


)


def
 
forward
(
self
,
 
z
):


img
 
=
 
self
.
model
(
z
)


return
 
img




Step 6: Defining a Utility Class to Build the Discriminator
The PyTorch code describes the discriminator architecture for a GAN. The class Discriminator is descended from nn.Module. It is composed of linear layers, batch normalization, 
dropout
, convolutional, 
LeakyReLU
, and sequential layers. 


An image (img) is the discriminator’s input, and its validity—the probability that the input image is real as opposed to artificial—is its output. 




Python




# Define the discriminator


class
 
Discriminator
(
nn
.
Module
):


def
 
__init__
(
self
):


super
(
Discriminator
,
 
self
)
.
__init__
()


self
.
model
 
=
 
nn
.
Sequential
(


nn
.
Conv2d
(
3
,
 
32
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
32
,
 
64
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
ZeroPad2d
((
0
,
 
1
,
 
0
,
 
1
)),


nn
.
BatchNorm2d
(
64
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
64
,
 
128
,
 
kernel_size
=
3
,
 
stride
=
2
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
128
,
 
momentum
=
0.82
),


nn
.
LeakyReLU
(
0.2
),


nn
.
Dropout
(
0.25
),


nn
.
Conv2d
(
128
,
 
256
,
 
kernel_size
=
3
,
 
stride
=
1
,
 
padding
=
1
),


nn
.
BatchNorm2d
(
256
,
 
momentum
=
0.8
),


nn
.
LeakyReLU
(
0.25
),


nn
.
Dropout
(
0.25
),


nn
.
Flatten
(),


nn
.
Linear
(
256
 
*
 
5
 
*
 
5
,
 
1
),


nn
.
Sigmoid
()


)


def
 
forward
(
self
,
 
img
):


validity
 
=
 
self
.
model
(
img
)


return
 
validity




Step 7: Building the Generative Adversarial Network
The code snippet defines and initializes a discriminator (Discriminator) and a generator (Generator). 


The designated device (GPU if available) receives both models. 
Binary Cross Entropy Loss,
 which is frequently used for GANs, is selected as the loss function (adversarial_loss). 
For the generator (optimizer_G) and discriminator (optimizer_D), distinct Adam optimizers with predetermined learning rates and betas are also defined. 


Python




# Define the generator and discriminator


# Initialize generator and discriminator


generator
 
=
 
Generator
(
latent_dim
)
.
to
(
device
)


discriminator
 
=
 
Discriminator
()
.
to
(
device
)


# Loss function


adversarial_loss
 
=
 
nn
.
BCELoss
()


# Optimizers


optimizer_G
 
=
 
optim
.
Adam
(
generator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))


optimizer_D
 
=
 
optim
.
Adam
(
discriminator
.
parameters
()
\
                         
,
 
lr
=
lr
,
 
betas
=
(
beta1
,
 
beta2
))




Step 8: Training the Generative Adversarial Network
For a Generative Adversarial Network (GAN), the code implements the training loop. 


The training data batches are iterated through during each epoch. Whereas the generator (optimizer_G) is trained to generate realistic images that trick the discriminator, the discriminator (optimizer_D) is trained to distinguish between real and phony images. 
The generator and discriminator’s adversarial losses are computed. Model parameters are updated by means of Adam optimizers and the losses are backpropagated. 
Discriminator printing and generator losses are used to track progress. For a visual assessment of the training process, generated images are additionally saved and shown every 10 epochs.


Python




# Training loop


for
 
epoch
 
in
 
range
(
num_epochs
):


for
 
i
,
 
batch
 
in
 
enumerate
(
dataloader
):


# Convert list to tensor


real_images
 
=
 
batch
[
0
]
.
to
(
device
)


# Adversarial ground truths


valid
 
=
 
torch
.
ones
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


fake
 
=
 
torch
.
zeros
(
real_images
.
size
(
0
),
 
1
,
 
device
=
device
)


# Configure input


real_images
 
=
 
real_images
.
to
(
device
)


# ---------------------


#  Train Discriminator


# ---------------------


optimizer_D
.
zero_grad
()


# Sample noise as generator input


z
 
=
 
torch
.
randn
(
real_images
.
size
(
0
),
 
latent_dim
,
 
device
=
device
)


# Generate a batch of images


fake_images
 
=
 
generator
(
z
)


# Measure discriminator's ability 


# to classify real and fake images


real_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
real_images
),
 
valid
)


fake_loss
 
=
 
adversarial_loss
(
discriminator
\
                                     
(
fake_images
.
detach
()),
 
fake
)


d_loss
 
=
 
(
real_loss
 
+
 
fake_loss
)
 
/
 
2


# Backward pass and optimize


d_loss
.
backward
()


optimizer_D
.
step
()


# -----------------


#  Train Generator


# -----------------


optimizer_G
.
zero_grad
()


# Generate a batch of images


gen_images
 
=
 
generator
(
z
)


# Adversarial loss


g_loss
 
=
 
adversarial_loss
(
discriminator
(
gen_images
),
 
valid
)


# Backward pass and optimize


g_loss
.
backward
()


optimizer_G
.
step
()


# ---------------------


#  Progress Monitoring


# ---------------------


if
 
(
i
 
+
 
1
)
 
%
 
100
 
==
 
0
:


print
(


f
"Epoch [
{
epoch
+
1
}
/
{
num_epochs
}
]
\


                        Batch 
{
i
+
1
}
/
{
len
(
dataloader
)
}
 "


f
"Discriminator Loss: 
{
d_loss
.
item
()
:
.4f
}
 "


f
"Generator Loss: 
{
g_loss
.
item
()
:
.4f
}
"


)


# Save generated images for every epoch


if
 
(
epoch
 
+
 
1
)
 
%
 
10
 
==
 
0
:


with
 
torch
.
no_grad
():


z
 
=
 
torch
.
randn
(
16
,
 
latent_dim
,
 
device
=
device
)


generated
 
=
 
generator
(
z
)
.
detach
()
.
cpu
()


grid
 
=
 
torchvision
.
utils
.
make_grid
(
generated
,
\
                                        
nrow
=
4
,
 
normalize
=
True
)


plt
.
imshow
(
np
.
transpose
(
grid
,
 
(
1
,
 
2
,
 
0
)))


plt
.
axis
(
"off"
)


plt
.
show
()




Output:


Epoch [10/10]                        Batch 1300/1563 Discriminator Loss: 0.4473 Generator Loss: 0.9555
Epoch [10/10]                        Batch 1400/1563 Discriminator Loss: 0.6643 Generator Loss: 1.0215
Epoch [10/10]                        Batch 1500/1563 Discriminator Loss: 0.4720 Generator Loss: 2.5027
GAN Output
Application Of Generative Adversarial Networks (GANs)
GANs, or Generative Adversarial Networks, have many uses in many different fields. Here are some of the widely recognized uses of GANs:


Image Synthesis and Generation : GANs
 are often used for picture synthesis and generation tasks,  They may create fresh, lifelike pictures that mimic training data by learning the distribution that explains the dataset. The development of lifelike avatars, high-resolution photographs, and fresh artwork have all been facilitated by these types of generative networks.
Image-to-Image Translation : GANs
 may be used for problems involving image-to-image translation, where the objective is to convert an input picture from one domain to another while maintaining its key features. GANs may be used, for instance, to change pictures from day to night, transform drawings into realistic images, or change the creative style of an image.
Text-to-Image Synthesis : GANs
 have been used to create visuals from descriptions in text. GANs may produce pictures that translate to a description given a text input, such as a phrase or a caption. This application might have an impact on how realistic visual material is produced using text-based instructions.
Data Augmentation : GANs
 can augment present data and increase the robustness and generalizability of machine-learning models by creating synthetic data samples.
Data Generation for Training : GANs
 can enhance the resolution and quality of low-resolution images. By training on pairs of low-resolution and high-resolution images, GANs can generate high-resolution images from low-resolution inputs, enabling improved image quality in various applications such as medical imaging, satellite imaging, and video enhancement.
Advantages of GAN
The advantages of the GANs are as follows:


Synthetic data generation
: GANs can generate new, synthetic data that resembles some known data distribution, which can be useful for data augmentation, anomaly detection, or creative applications.
High-quality results
: GANs can produce high-quality, photorealistic results in image synthesis, video synthesis, music synthesis, and other tasks.
Unsupervised learning
: GANs can be trained without labeled data, making them suitable for unsupervised learning tasks, where labeled data is scarce or difficult to obtain.
Versatility
: GANs can be applied to a wide range of tasks, including image synthesis, text-to-image synthesis, image-to-image translation, 
anomaly detection
, 
data augmentation
, and others.
Disadvantages of GAN
The disadvantages of the GANs are as follows:


Training Instability
: GANs can be difficult to train, with the risk of instability, mode collapse, or failure to converge.
Computational Cost
: GANs can require a lot of computational resources and can be slow to train, especially for high-resolution images or large datasets.
Overfitting
: GANs can overfit the training data, producing synthetic data that is too similar to the training data and lacking diversity.
Bias and Fairness
: GANs can reflect the biases and unfairness present in the training data, leading to discriminatory or biased synthetic data.
Interpretability and Accountability
: GANs can be opaque and difficult to interpret or explain, making it challenging to ensure accountability, transparency, or fairness in their applications.
Generative Adversarial Network (GAN) – FAQs
What is a Generative Adversarial Network(GAN)?
An artificial intelligence model known as a GAN is made up of two neural networks—a discriminator and a generator—that were developed in tandem using adversarial training. The discriminator assesses the new data instances for authenticity, while the generator produces new ones.


What are the main applications of GAN?
Generating images and videos, transferring styles, enhancing data, translating images to other images, producing realistic synthetic data for machine learning model training, and super-resolution are just a few of the many uses for GANs.


What challenges do GAN face?
GANs encounter difficulties such training instability, mode collapse (when the generator generates a limited range of samples), and striking the correct balance between the discriminator and generator. It’s frequently necessary to carefully build the model architecture and tune the hyperparameters.


How are GAN evaluated?
The produced samples’ quality, diversity, and resemblance to real data are the main criteria used to assess GANs. For quantitative assessment, metrics like the Fréchet Inception Distance (FID) and Inception Score are frequently employed.


Can GAN be used for tasks other than image generation
?
Yes, different tasks can be assigned to GANs. Text, music, 3D models, and other things have all been generated with them. The usefulness of conditional GANs is expanded by enabling the creation of specific content under certain input conditions.


What are some famous architectures of GANs
?
A few well-known GAN architectures are Progressive GAN (PGAN), Wasserstein GAN (WGAN), Conditional GAN (cGAN), Deep Convolutional GAN (DCGAN), and Vanilla GAN. Each has special qualities and works best with particular kinds of data and tasks.




















R










 




Rahul_Roy
 












 
Follow
 




















 
















Improve


















Previous Article








Basics of Generative Adversarial Networks (GANs)










Next Article










Use Cases of Generative Adversarial Networks




















 
 
Please 
Login
 to comment...




















Read More








Similar Reads








What is so special about Generative Adversarial Network (GAN)


Fans are ecstatic for a variety of reasons, including the fact that GANs were the first generative algorithms to produce convincingly good results, as well as the fact that they have opened up many new research directions. In the last several years, GANs are considered to be the most prominent machine learning research, and since then, GANs have re








5 min read










Selection of GAN vs Adversarial Autoencoder models


In this article, we are going to see the selection of GAN vs Adversarial Autoencoder models. Generative Adversarial Network (GAN)The Generative Adversarial Network, or GAN, is one of the most prominent deep generative modeling methodologies right now. The primary distinction between GAN and VAE is that GAN seeks to match the pixel level distributio








6 min read










Building a Generative Adversarial Network using Keras


Prerequisites: Generative Adversarial Network This article will demonstrate how to build a Generative Adversarial Network using the Keras library. The dataset which is used is the CIFAR10 Image dataset which is preloaded into Keras. You can read about the dataset here. Step 1: Importing the required libraries import numpy as np import matplotlib.py








4 min read












Conditional Generative Adversarial Network


Imagine a situation where you can generate images of cats that match your ideal vision or a landscape that adheres to a specific artistic style. CGANs is a neural network that enables the generation of data that aligns with specific properties, which can be class labels, textual descriptions, or other traits, by harnessing the power of conditions.








13 min read










Generative Adversarial Networks (GANs) | An Introduction


Generative Adversarial Networks (GANs) was first introduced by Ian Goodfellow in 2014. GANs are a powerful class of neural networks that are used for unsupervised learning. GANs can create anything whatever you feed to them, as it Learn-Generate-Improve. To understand GANs first you must have little understanding of Convolutional Neural Networks. C








6 min read










Wasserstein Generative Adversarial Networks (WGANs) Convergence and Optimization


Wasserstein Generative Adversarial Network (WGANs) is a modification of Deep Learning GAN with few changes in the algorithm. GAN, or Generative Adversarial Network, is a way to build an accurate generative model. This network was introduced by Martin Arjovsky, Soumith Chintala, and Léon Bottou in 2017. It is widely used to generate realistic images








9 min read










Generative Adversarial Networks (GANs) in PyTorch


The aim of the article is to implement GANs architecture using PyTorch framework. The article provides comprehensive understanding of GANs in PyTorch along with in-depth explanation of the code. Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning. They consist of two neural








9 min read












Image Generation using Generative Adversarial Networks (GANs)


Generative Adversarial Networks (GANs) represent a revolutionary approach to, artificial intelligence, particularly for generating images. Introduced in 2014, GANs have significantly advanced the ability to create realistic and high-quality images from random noise. In this article, we are going to train GANs model on MNIST dataset for generating i








8 min read










Architecture of Super-Resolution Generative Adversarial Networks (SRGANs)


Super-Resolution Generative Adversarial Networks (SRGANs) are advanced deep learning models designed to upscale low-resolution images to high-resolution outputs with remarkable detail. This article aims to provide a comprehensive overview of SRGANs, focusing on their architecture, key components, and training techniques to enhance image quality. Ta








9 min read










Generative Adversarial Networks (GANs) with R


Generative Adversarial Networks (GANs) are a type of neural network architecture introduced by Ian Goodfellow and his colleagues in 2014. GANs are designed to generate new data samples that resemble a given dataset. They can produce high-quality synthetic data across various domains. Working of GANsGANs consist of two neural networks. first is the








15 min read










Generative Adversarial Networks (GANs) vs Diffusion Models


Generative Adversarial Networks (GANs) and Diffusion Models are powerful generative models designed to produce synthetic data that closely resembles real-world data. Each model has distinct architectures, strengths, and limitations, making them uniquely suited for various applications. This article aims to provide a comprehensive comparison between








5 min read










Mastering Adversarial Attacks: How One Pixel Can Fool a Neural Network


Neural networks are among the best tools for classification tasks. They power everything from image recognition to natural language processing, providing incredible accuracy and versatility. But what if I told you that you could completely undermine a neural network or trick it into making mistakes? Intrigued? Let's explore adversarial attacks and








5 min read










Deep Convolutional GAN with Keras


Deep Convolutional GAN (DCGAN) was proposed by a researcher from MIT and Facebook AI research. It is widely used in many convolution-based generation-based techniques. The focus of this paper was to make training GANs stable. Hence, they proposed some architectural changes in the computer vision problems. In this article, we will be using DCGAN on








9 min read












Understanding Auxiliary Classifier : GAN


Prerequisite: GANs(General Adversarial Networks) In this article, we will be discussing a special class conditional GAN or c-GAN known as Auxiliary Classifier GAN or AC-GAN. Before getting into that, it is important to understand what a class conditional GAN is. Class-Conditional GAN (c-GANs): c-GAN can be understood as a GAN with some conditional








4 min read










Building an Auxiliary GAN using Keras and Tensorflow


Prerequisites: Generative Adversarial Network This article will demonstrate how to build an Auxiliary Generative Adversarial Network using the Keras and TensorFlow libraries. The dataset which is used is the MNIST Image dataset pre-loaded into Keras. Step 1: Setting up the environment Step 1 : Open Anaconda prompt in Administrator mode. Step 2 : Cr








5 min read










Difference between GAN vs DCGAN.


Answer: GAN is a broader class of generative models, while DCGAN specifically refers to a type of GAN that utilizes deep convolutional neural networks for image generation.Below is a detailed comparison between GAN (Generative Adversarial Network) and DCGAN (Deep Convolutional Generative Adversarial Network): FeatureGANDCGANArchitectureGeneric arch








2 min read










Adversarial Search Algorithms


Adversarial search algorithms are the backbone of strategic decision-making in artificial intelligence, it enables the agents to navigate competitive scenarios effectively. This article offers concise yet comprehensive advantages of these algorithms from their foundational principles to practical applications. Let's uncover the strategies that driv








15+ min read












Alpha-Beta pruning in Adversarial Search Algorithms


In artificial intelligence, particularly in game playing and decision-making, adversarial search algorithms are used to model and solve problems where two or more players compete against each other. One of the most well-known techniques in this domain is alpha-beta pruning. This article explores the concept of alpha-beta pruning, its implementation








6 min read










Explain the role of minimax algorithm in adversarial search for optimal decision-making?


In the realm of artificial intelligence (AI), particularly in game theory and decision-making scenarios involving competition, the ability to predict and counteract an opponent's moves is paramount. This is where adversarial search algorithms come into play. Among the most prominent and foundational of these algorithms is the Minimax algorithm. It








11 min read










Pandas AI: The Generative AI Python Library


In the age of AI, many of our tasks have been automated especially after the launch of ChatGPT. One such tool that uses the power of ChatGPT to ease data manipulation task in Python is PandasAI. It leverages the power of ChatGPT to generate Python code and executes it. The output of the generated code is returned. Pandas AI helps performing tasks i








9 min read










The Difference Between Generative and Discriminative Machine Learning Algorithms


Machine learning algorithms allow computers to learn from data and make predictions or judgments, machine learning algorithms have revolutionized a number of sectors. Generic and discriminative algorithms are two essential strategies with various applications in the field of machine learning. We will examine the core distinctions between generative








6 min read










What is Language Revitalization in Generative AI?


Imagine a world where ancient tongues, on the brink of fading into silence, are reborn. Where stories whispered through generations find a digital echo and cultural knowledge carried in every syllable is amplified across the internet. This is the promise of language revitalization in generative AI, a revolutionary field that seeks to leverage the p








7 min read










Differences between Conversational AI and Generative AI


Artificial intelligence has evolved significantly in the past few years, making day-to-day tasks easy and efficient. Conversational AI and Generative AI are the two subsets of artificial intelligence that rapidly advancing the field of AI and have become prominent and transformative. Both technologies make use of machine learning and natural langua








8 min read












10 Best Generative AI Tools to Refine Your Content Strategy


Many of us struggle with content creation and strategy. We're good at the creative, artful side, like writing compelling stories. But the analytical, strategic part is harder. Even when we do get strategic, we spend lots of time on keyword research, topic selection, and tracking performance. AI content tools can give you an advantage on the science








9 min read










5 Top Generative AI Design Tools in 2024 [Free & Paid]


Are you ready to level up your design game? Gone are the days when designers had to sit and design creatives from scratch. With the rise of artificial intelligence and its integration with different domains, you can save a lot of time and still come up with quality output. You can use these tools in generating base designs and even assist the whole








9 min read










What is Generative AI?


Nowadays as we all know the power of Artificial Intelligence is developing day by day, and after the introduction of Generative AI is taking creativity to the next level Generative AI is a subset of Deep learning that is again a part of Artificial Intelligence.  In this article, we will explore,  What is Generative AI? Examples, Definition, Models








12 min read










What is the difference between Generative and Discriminative algorithm?


Answer: Generative algorithms model the joint probability distribution of input features and target labels, while discriminative algorithms directly learn the decision boundary between classes.Generative algorithms focus on modeling the joint probability distribution of both input features and target labels. By capturing statistical dependencies wi








2 min read












7 Best Generative AI Tools for Developers [2024]


In the rapidly evolving world of technology, generative Artificial intelligence (AI) tools for developers have become indispensable assets for innovation and efficiency. These cutting-edge tools harness the power of advanced algorithms and machine learning techniques to autonomously generate content, designs, and code, transforming the development








9 min read










Generative Modeling in TensorFlow


Generative modeling is the process of learning the underlying structure of a dataset to generate new samples that mimic the distribution of the original data. The article aims to provide a comprehensive overview of generative modelling along with the implementation leveraging the TensorFlow framework. Table of Content What are generative models and








14 min read










AI-Coustics: Fights Noisy Audio With Generative AI


Have you ever been troubled by noisy audio during a video call or interview? The constant hum of traffic, the rustle of wind, or even a bustling room can significantly degrade audio quality. For content creators, journalists, and anyone relying on clean audio recording and speech clarity in videos, these challenges can be a major source of frustrat








9 min read














Article Tags : 






AI-ML-DS






Deep Learning






Python






Python-Quizzes






Technical Scripter






python


 


+2 More






Practice Tags : 




python
python
 














Like














































































































285k+ interested Geeks 








Data Structures & Algorithms in Python - Self Paced 










Explore


































198k+ interested Geeks 








Python Full Course Online - Complete Beginner to Advanced 










Explore


































927k+ interested Geeks 








Complete Interview Preparation 










Explore














 










Explore More




























































Read more
## 
      Overview of GAN Structure
      
    














    
        Home
      
  









    
        Products
      
  









    
        Machine Learning
      
  









    
        Advanced courses
      
  









    
        GAN
      
  















  
    
    Send feedback
  
  





      Overview of GAN Structure
      
    







      
      Stay organized with collections
    



      
      Save and categorize content based on your preferences.
    














A generative adversarial network (GAN) has two parts:




The 
generator
 learns to generate plausible data. The generated instances
become negative training examples for the discriminator.


The 
discriminator
 learns to distinguish the generator's fake data from
real data. The discriminator penalizes the generator for producing
implausible results.




When training begins, the generator produces obviously fake data, and the
discriminator quickly learns to tell that it's fake:




As training progresses, the generator gets closer to producing output that
can fool the discriminator:




Finally, if generator training goes well, the discriminator gets worse at
telling the difference between real and fake. It starts to classify fake data as
real, and its accuracy decreases.




Here's a picture of the whole system:




Both the generator and the discriminator are neural networks. The generator
output is connected directly to the discriminator input. Through

backpropagation
, the
discriminator's classification provides a signal that the generator uses to
update its weights.


Let's explain the pieces of this system in greater detail.










Previous



        arrow_back
      



        Generative Models
      










Next



        Discriminator
      



        arrow_forward
      



















  
    
    Send feedback
  
  










Except as otherwise noted, the content of this page is licensed under the 
Creative Commons Attribution 4.0 License
, and code samples are licensed under the 
Apache 2.0 License
. For details, see the 
Google Developers Site Policies
. Java is a registered trademark of Oracle and/or its affiliates.


Last updated 2022-07-18 UTC.






















Read more
## Statistics > Machine Learning











Statistics > Machine Learning






arXiv:1406.2661
 (stat)
    









  [Submitted on 10 Jun 2014]


Title:
Generative Adversarial Networks


Authors:
Ian J. Goodfellow
, 
Jean Pouget-Abadie
, 
Mehdi Mirza
, 
Bing Xu
, 
David Warde-Farley
, 
Sherjil Ozair
, 
Aaron Courville
, 
Yoshua Bengio
 
View a PDF of the paper titled Generative Adversarial Networks, by Ian J. Goodfellow and 7 other authors


View PDF




Abstract:
We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.
    








Subjects:




Machine Learning (stat.ML)
; Machine Learning (cs.LG)




Cite as:


arXiv:1406.2661
 [stat.ML]






 


(or 


arXiv:1406.2661v1
 [stat.ML]
 for this version)
          






 


 
https://doi.org/10.48550/arXiv.1406.2661






Focus to learn more








                  arXiv-issued DOI via DataCite
















Submission history
 From: Ian Goodfellow [
view email
]      
 
[v1]

        Tue, 10 Jun 2014 18:58:17 UTC (1,257 KB)








 




Full-text links:


Access Paper:





View a PDF of the paper titled Generative Adversarial Networks, by Ian J. Goodfellow and 7 other authors
View PDF
TeX Source
Other Formats


view license




 

    Current browse context: 
stat.ML






< prev




  |  
 


next >






new


 | 


recent


 | 
2014-06



    Change to browse by:
    


cs


cs.LG


stat










References & Citations




NASA ADS
Google Scholar


Semantic Scholar












 59 blog links
 (
what is this?
)
        






a


export BibTeX citation


Loading...










BibTeX formatted citation


×






loading...






Data provided by: 










Bookmark












 










Bibliographic Tools




Bibliographic and Citation Tools














Bibliographic Explorer Toggle








Bibliographic Explorer
 
(
What is the Explorer?
)
















Litmaps Toggle








Litmaps
 
(
What is Litmaps?
)
















scite.ai Toggle








scite Smart Citations
 
(
What are Smart Citations?
)


















Code, Data, Media




Code, Data and Media Associated with this Article














Links to Code Toggle








CatalyzeX Code Finder for Papers
 
(
What is CatalyzeX?
)
















DagsHub Toggle








DagsHub
 
(
What is DagsHub?
)
















GotitPub Toggle








Gotit.pub
 
(
What is GotitPub?
)
















Links to Code Toggle








Papers with Code
 
(
What is Papers with Code?
)
















ScienceCast Toggle








ScienceCast
 
(
What is ScienceCast?
)
























Demos




Demos














Replicate Toggle








Replicate
 
(
What is Replicate?
)
















Spaces Toggle








Hugging Face Spaces
 
(
What is Spaces?
)
















Spaces Toggle








TXYZ.AI
 
(
What is TXYZ.AI?
)


















Related Papers




Recommenders and Search Tools














Link to Influence Flower








Influence Flower
 
(
What are Influence Flowers?
)
















Connected Papers Toggle








Connected Papers
 
(
What is Connected Papers?
)
















Core recommender toggle








CORE Recommender
 
(
What is CORE?
)












Author


Venue


Institution


Topic





























        About arXivLabs
      








arXivLabs: experimental projects with community collaborators


arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.


Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.


Have an idea for a project that will add value for arXiv's community? 
Learn more about arXivLabs
.






















Which authors of this paper are endorsers?
 |
    
Disable MathJax
 (
What is MathJax?
)
    










Read more
## generative adversarial network (GAN) 

No Content
Read more
## No Title

No Content
Read more
## No Title

No Content
Read more
## No Title

No Content
Read more
## No Title

No Content
Read more
## A Gentle Introduction to Generative Adversarial Networks (GANs)

No Content
Read more
## What is a GAN?

























What is Cloud Computing?


Cloud Computing Concepts Hub


Generative AI






What is a GAN?






Create an AWS Account



















































             Explore Generative AI Services 
           



             Build, deploy, and run generative AI applications on AWS 
           
 























             Check out Generative AI on AWS 
           



             Innovate faster with the most comprehensive set of Generative AI services 
           
 























             Browse Generative AI Trainings 
           



             Get started on generative AI training with content built by AWS experts 
           
 























             Read Generative AI Blogs 
           



             Get the latest AWS generative AI product news and best practices 
           
 








































What is a GAN?


What are some use cases of generative adversarial networks?


How does a generative adversarial network work?


What are the types of generative adversarial networks?


How can AWS support your generative adversarial network requirements?
















What is a GAN?




A generative adversarial network (GAN) is a deep learning architecture. It trains two neural networks to compete against each other to generate more authentic new data from a given training dataset. For instance, you can generate new images from an existing image database or original music from a database of songs. A GAN is called 
adversarial
 because it trains two different networks and pits them against each other. One network generates new data by taking an input data sample and modifying it as much as possible. The other network tries to predict whether the generated data output belongs in the original dataset. In other words, the predicting network determines whether the generated data is fake or real. The system generates newer, improved versions of fake data values until the predicting network can no longer distinguish fake from original.
















What are some use cases of generative adversarial networks?




The GAN architecture has several applications across different industries. Next, we give some examples.


Generate images


Generative adversarial networks create realistic images through text-based prompts or by modifying existing images. They can help create realistic and immersive visual experiences in video games and digital entertainment.


GAN can also edit images—like converting a low-resolution image to a high resolution or turning a black-and-white image to color. It can also create realistic faces, characters, and animals for animation and video.


Generate training data for other models


In 
machine learning (ML)
, data augmentation artificially increases the training set by creating modified copies of a dataset using existing data.


You can use generative models for data augmentation to create synthetic data with all the attributes of real-world data. For instance, it can generate fraudulent transaction data that you then use to train another fraud-detection ML system. This data can teach the system to accurately distinguish between suspicious and genuine transactions.


Complete missing information


Sometimes, you may want the generative model to accurately guess and complete some missing information in a dataset.


For instance, you can train GAN to generate images of the surface below ground (sub-surface) by understanding the correlation between surface data and underground structures. By studying known sub-surface images, it can create new ones using terrain maps for energy applications like geothermal mapping or carbon capture and storage.


Generate 3D models from 2D data


GAN can generate 3D models from 2D photos or scanned images. For instance, in healthcare, GAN combines X-rays and other body scans to create realistic images of organs for surgical planning and simulation.


  
















How does a generative adversarial network work?




A generative adversarial network system comprises two deep neural networks—the 
generator network
 and the 
discriminator network
. Both networks train in an adversarial game, where one tries to generate new data and the other attempts to predict if the output is fake or real data.


Technically, the GAN works as follows. A complex mathematical equation forms the basis of the entire computing process, but this is a simplistic overview:




The generator neural network analyzes the training set and identifies data attributes


The discriminator neural network also analyzes the initial training data and distinguishes between the attributes independently


The generator modifies some data attributes by adding noise (or random changes) to certain attributes


The generator passes the modified data to the discriminator


The discriminator calculates the probability that the generated output belongs to the original dataset


The discriminator gives some guidance to the generator to reduce the noise vector randomization in the next cycle




The generator attempts to maximize the probability of mistake by the discriminator, but the discriminator attempts to minimize the probability of error. In training iterations, both the generator and discriminator evolve and confront each other continuously until they reach an equilibrium state. In the equilibrium state, the discriminator can no longer recognize synthesized data. At this point, the training process is over.


  


GAN training example


Let's contextualize the above with an example of the GAN model in image-to-image translation.


Consider that the input image is a human face that the GAN attempts to modify. For example, the attributes can be the shapes of eyes or ears. Let's say the generator changes the real images by adding sunglasses to them. The discriminator receives a set of images, some of real people with sunglasses and some generated images that were modified to include sunglasses.


If the discriminator can differentiate between fake and real, the generator updates its parameters to generate even better fake images. If the generator produces images that fool the discriminator, the discriminator updates its parameters. Competition improves both networks until equilibrium is reached.
















What are the types of generative adversarial networks?




There are different types of GAN models depending on the mathematical formulas used and the different ways the generator and discriminator interact with each other.


We give some commonly used models next, but the list is not comprehensive. There are numerous other GAN types—like StyleGAN, CycleGAN, and DiscoGAN—that solve different types of problems.


Vanilla GAN


This is the basic GAN model that generates data variation with little or no feedback from the discriminator network. A vanilla GAN typically requires enhancements for most real-world use cases.


Conditional GAN


A conditional GAN (cGAN) introduces the concept of conditionality, allowing for targeted data generation. The generator and discriminator receive additional information, typically as class labels or some other form of conditioning data.


For instance, if generating images, the condition could be a label that describes the image content. Conditioning allows the generator to produce data that meets specific conditions.


Deep xonvolutional GAN


Recognizing the power of convolutional neural networks (CNNs) in image processing, Deep convolutional GAN (DCGAN) integrates CNN architectures into GANs.


With DCGAN, the generator uses transposed convolutions to upscale data distribution, and the discriminator also uses convolutional layers to classify data. The DCGAN also introduces architectural guidelines to make training more stable.


Super-resolution GAN


Super-resolution GANS (SRGANs) focus on upscaling low-resolution images to high resolution. The goal is to enhance images to a higher resolution while maintaining image quality and details.


Laplacian Pyramid GANs (LAPGANs) address the challenge of generating high-resolution images by breaking down the problem into stages. They use a hierarchical approach, with multiple generators and discriminators working at different scales or resolutions of the image. The process begins with generating a low-resolution image that improves in quality over progressive GAN stages.
















How can AWS support your generative adversarial network requirements?




Amazon Web Services (AWS) offers many services to support your GAN requirements.


Amazon SageMaker
 is a fully managed service that you can use to prepare data and build, train, and deploy machine learning models. These models can be used in many scenarios, and SageMaker comes with fully managed infrastructure, tools, and workflows. It has a wide range of features to accelerate GAN development and training for any application.


Amazon Bedrock
 is a fully managed service. You can use it to access foundation models (FMs), or trained deep neural networks, from Amazon and leading artificial intelligence (AI) startups. These FMs are available through APIs—so you can choose from various options to find the best model for your needs. You can use these models in your own GAN applications. With Amazon Bedrock, you can more quickly develop and deploy scalable, reliable, and secure generative AI applications. And you don't have to manage infrastructure.


AWS DeepComposer
 gives you a creative way to get started with ML. You can get hands-on with a musical keyboard and the latest ML techniques designed to expand your ML skills. Regardless of their background in ML or music, your developers can get started with GANs. And they can train and optimize GAN models to create original music.


Get started with generative adversarial networks on AWS by 
creating an account 
today.
































 Next Steps on AWS

















           Check out additional product-related resources 
         


 Innovate faster with AWS generative AI services 

















           Sign up for a free account 
         




Instant get access to the AWS Free Tier.




 Sign up 

















           Start building in the console 
         




Get started building in the AWS management console.




 Sign in 












Read more
## Generative Adversarial Networks(GANs): End-to-End Introduction

No Content
Read more
## Please update your browser

No Content
Read more
## Please update your browser

No Content
Read more
## Understand & manage your location when you search on Google

No Content
Read more
## Sign in

No Content
Read more
## No Title

No Content
Read more
