var documenterSearchIndex = {"docs":
[{"location":"experiments/#Experiments","page":"Experiments","title":"Experiments","text":"","category":"section"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"The experiments/ directory contains a Makefile with which you can run the experiments for our publication On Multi-Class Extensions of Adjusted Classify and Count.","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"@InProceedings{bunse2022multiclass,\n  author    = {Mirko Bunse},\n  title     = {On Multi-Class Extensions of Adjusted Classify and Count},\n  booktitle = {Int. Worksh. on Learn. to Quantify: Meth. and Appl.},\n  year      = {2022},\n  pages     = {43--50},\n}","category":"page"},{"location":"experiments/#Running-the-experiments","page":"Experiments","title":"Running the experiments","text":"","category":"section"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"CAUTION: We have run these experiments on 40 cores with 48 GB of RAM; with this setup, the experiments took 67 h. If you just want to check whether the scripts work, you can call make -f lequa.mk tests to traverse the entire code path with just a few iterations; this test completes in a few minutes.","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"cd experiments/\nmake -f lequa.mk -n # inspect all steps of our experimentation without running them (dry-run)\n\nmake -f lequa.mk # run all experimentation (CAUTION: computation-heavy)","category":"page"},{"location":"experiments/#Docker-setup","page":"Experiments","title":"Docker setup","text":"","category":"section"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"We provide a Docker setup for those who prefer to run the experiments in an isolated environment, and possibly in a computing cluster.","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"cd experiments/docker/\nmake # build the Docker image\n./run.sh # start an interactive Docker container from the image","category":"page"},{"location":"api-reference/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"Below, you find a listing of all public methods of this package. Any other method you might find in the source code is not intended for direct usage.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"CurrentModule = QUnfold","category":"page"},{"location":"api-reference/#Common-interface","page":"API reference","title":"Common interface","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"TODO with an exemplary link to fit.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"fit\npredict\npredict_with_background","category":"page"},{"location":"api-reference/#QUnfold.fit","page":"API reference","title":"QUnfold.fit","text":"fit(m, X, y) -> FittedMethod\n\nReturn a copy of the QUnfold method m that is fitted to the data set (X, y).\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.predict","page":"API reference","title":"QUnfold.predict","text":"predict(m, X) -> Vector{Float64}\n\nPredict the class prevalences in the data set X with the fitted method m.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.predict_with_background","page":"API reference","title":"QUnfold.predict_with_background","text":"predict_with_background(m, X, X_b, α=1) -> Vector{Float64}\n\nPredict the class prevalences in the observed data set X with the fitted method m, taking into account a background measurement X_b that is scaled by α.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#Quantification-/-unfolding-methods","page":"API reference","title":"Quantification / unfolding methods","text":"","category":"section"},{"location":"api-reference/#CC","page":"API reference","title":"CC","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"CC","category":"page"},{"location":"api-reference/#QUnfold.CC","page":"API reference","title":"QUnfold.CC","text":"CC(classifier; kwargs...)\n\nThe Classify & Count method, which uses crisp classifier predictions without any adjustment. This weak baseline method is proposed by Forman, 2008: Quantifying counts and costs via classification.\n\nKeyword arguments\n\nfit_classifier = true whether or not to fit the given classifier.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#ACC","page":"API reference","title":"ACC","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"ACC","category":"page"},{"location":"api-reference/#QUnfold.ACC","page":"API reference","title":"QUnfold.ACC","text":"ACC(classifier; kwargs...)\n\nThe Adjusted Classify & Count method, which solves a least squares objective with crisp classifier predictions.\n\nA regularization strength τ > 0 yields the o-ACC method for ordinal quantification, which is proposed by Bunse et al., 2022: Ordinal Quantification through Regularization.\n\nKeyword arguments\n\nstrategy = :softmax is the solution strategy (see below).\nτ = 0.0 is the regularization strength for o-ACC.\na = Float64[] are the acceptance factors for unfolding analyses.\nfit_classifier = true whether or not to fit the given classifier.\n\nStrategies\n\nFor binary classification, ACC is proposed by Forman, 2008: Quantifying counts and costs via classification. In the multi-class setting, multiple extensions are available.\n\n:softmax (default; our method) improves :softmax_full_reg by setting one latent parameter to zero instead of introducing a technical regularization term.\n:constrained constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: A method of automated nonparametric content analysis for social science.\n:pinv computes a pseudo-inverse akin to a minimum-norm constraint, as discussed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:inv computes the true inverse (if existent) of the transfer matrix M, as proposed by Vucetic & Obradovic, 2001: Classification on data with biased class distribution.\n:ovr solves multiple binary one-versus-rest adjustments, as proposed by Forman (2008).\n:none yields the CC method without any adjustment.\n:softmax_full_reg (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:softmax_reg (our method) is a variant of :softmax, which sets one latent parameter to zero in addition to introducing a technical regularization term.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#PCC","page":"API reference","title":"PCC","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"PCC","category":"page"},{"location":"api-reference/#QUnfold.PCC","page":"API reference","title":"QUnfold.PCC","text":"PCC(classifier; kwargs...)\n\nThe Probabilistic Classify & Countmethod, which uses predictions of posterior probabilities without any adjustment. This method is proposed by Bella et al., 2010: Quantification via Probability Estimators.\n\nKeyword arguments\n\nfit_classifier = true whether or not to fit the given classifier.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#PACC","page":"API reference","title":"PACC","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"PACC","category":"page"},{"location":"api-reference/#QUnfold.PACC","page":"API reference","title":"QUnfold.PACC","text":"PACC(classifier; kwargs...)\n\nThe Probabilistic Adjusted Classify & Count method, which solves a least squares objective with predictions of posterior probabilities.\n\nA regularization strength τ > 0 yields the o-PACC method for ordinal quantification, which is proposed by Bunse et al., 2022: Ordinal Quantification through Regularization.\n\nKeyword arguments\n\nstrategy = :softmax is the solution strategy (see below).\nτ = 0.0 is the regularization strength for o-PACC.\na = Float64[] are the acceptance factors for unfolding analyses.\nfit_classifier = true whether or not to fit the given classifier.\n\nStrategies\n\nFor binary classification, PACC is proposed by Bella et al., 2010: Quantification via Probability Estimators. In the multi-class setting, multiple extensions are available.\n\n:softmax (default; our method) improves :softmax_full_reg by setting one latent parameter to zero instead of introducing a technical regularization term.\n:constrained constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: A method of automated nonparametric content analysis for social science.\n:pinv computes a pseudo-inverse akin to a minimum-norm constraint, as discussed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:inv computes the true inverse (if existent) of the transfer matrix M, as proposed by Vucetic & Obradovic, 2001: Classification on data with biased class distribution.\n:ovr solves multiple binary one-versus-rest adjustments, as proposed by Forman (2008).\n:none yields the CC method without any adjustment.\n:softmax_full_reg (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:softmax_reg (our method) is a variant of :softmax, which sets one latent parameter to zero in addition to introducing a technical regularization term.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#RUN","page":"API reference","title":"RUN","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"RUN","category":"page"},{"location":"api-reference/#QUnfold.RUN","page":"API reference","title":"QUnfold.RUN","text":"RUN(transformer; kwargs...)\n\nThe Regularized Unfolding method by Blobel, 1985: Unfolding methods in high-energy physics experiments.\n\nKeyword arguments\n\nstrategy = :softmax is the solution strategy (see below).\nτ = 1e-6 is the regularization strength for ordinal quantification.\nn_df = -1 (only used if strategy==:original) is the effective number of degrees of freedom, required to be 0 < n_df <= C where C is the number of classes.\na = Float64[] are the acceptance factors for unfolding analyses.\n\nStrategies\n\nBlobel's loss function, feature transformation, and regularization can be optimized with multiple strategies.\n\n:softmax (default; our method) improves :softmax_full_reg by setting one latent parameter to zero instead of introducing a technical regularization term.\n:original is the original, unconstrained Newton optimization proposed by Blobel (1985).\n:constrained constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: A method of automated nonparametric content analysis for social science.\n:softmax_full_reg (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:softmax_reg (our method) is a variant of :softmax, which sets one latent parameter to zero in addition to introducing a technical regularization term.\n:unconstrained (our method) is similar to :original, but uses a more generic solver.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#SVD","page":"API reference","title":"SVD","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"SVD","category":"page"},{"location":"api-reference/#QUnfold.SVD","page":"API reference","title":"QUnfold.SVD","text":"SVD(transformer; kwargs...)\n\nThe The Singular Value Decomposition-based unfolding method by Hoecker & Kartvelishvili, 1996: SVD approach to data unfolding.\n\nKeyword arguments\n\nstrategy = :softmax is the solution strategy (see below).\nτ = 1e-6 is the regularization strength for ordinal quantification.\nn_df = -1 (only used if strategy==:original) is the effective rank, required to be 0 < n_df < C where C is the number of classes.\na = Float64[] are the acceptance factors for unfolding analyses.\n\nStrategies\n\nHoecker & Kartvelishvili's loss function, feature transformation, and regularization can be optimized with multiple strategies.\n\n:softmax (default; our method) improves :softmax_full_reg by setting one latent parameter to zero instead of introducing a technical regularization term.\n:original is the original, analytic solution proposed by Hoecker & Kartvelishvili (1996).\n:constrained constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: A method of automated nonparametric content analysis for social science.\n:softmax_full_reg (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:softmax_reg (our method) is a variant of :softmax, which sets one latent parameter to zero in addition to introducing a technical regularization term.\n:unconstrained (our method) is similar to :original, but uses a more generic solver.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#HDx","page":"API reference","title":"HDx","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"HDx","category":"page"},{"location":"api-reference/#QUnfold.HDx","page":"API reference","title":"QUnfold.HDx","text":"HDx(n_bins; kwargs...)\n\nThe Hellinger Distance-based method on feature histograms by González-Castro et al., 2013: Class distribution estimation based on the Hellinger distance.\n\nThe parameter n_bins specifies the number of bins per feature. A regularization strength τ > 0 yields the o-HDx method for ordinal quantification, which is proposed by Bunse et al., 2022: Machine learning for acquiring knowledge in astro-particle physics.\n\nKeyword arguments\n\nstrategy = :softmax is the solution strategy (see below).\nτ = 0.0 is the regularization strength for o-HDx.\na = Float64[] are the acceptance factors for unfolding analyses.\n\nStrategies\n\nGonzález-Castro et al.'s loss function and feature transformation can be optimized with multiple strategies.\n\n:softmax (default; our method) improves :softmax_full_reg by setting one latent parameter to zero instead of introducing a technical regularization term.\n:constrained constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: A method of automated nonparametric content analysis for social science.\n:softmax_full_reg (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:softmax_reg (our method) is a variant of :softmax, which sets one latent parameter to zero in addition to introducing a technical regularization term.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#HDy","page":"API reference","title":"HDy","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"HDy","category":"page"},{"location":"api-reference/#QUnfold.HDy","page":"API reference","title":"QUnfold.HDy","text":"HDy(classifier, n_bins; kwargs...)\n\nThe Hellinger Distance-based method on prediction histograms by González-Castro et al., 2013: Class distribution estimation based on the Hellinger distance.\n\nThe parameter n_bins specifies the number of bins per class. A regularization strength τ > 0 yields the o-HDx method for ordinal quantification, which is proposed by Bunse et al., 2022: Machine learning for acquiring knowledge in astro-particle physics.\n\nKeyword arguments\n\nstrategy = :softmax is the solution strategy (see below).\nτ = 0.0 is the regularization strength for o-HDx.\na = Float64[] are the acceptance factors for unfolding analyses.\nfit_classifier = true whether or not to fit the given classifier.\n\nStrategies\n\nGonzález-Castro et al.'s loss function and feature transformation can be optimized with multiple strategies.\n\n:softmax (default; our method) improves :softmax_full_reg by setting one latent parameter to zero instead of introducing a technical regularization term.\n:constrained constrains the optimization to proper probability densities, as proposed by Hopkins & King, 2010: A method of automated nonparametric content analysis for social science.\n:softmax_full_reg (our method) introduces a soft-max layer, which makes contraints obsolete. This strategy employs a technical regularization term, as proposed by Bunse, 2022: On Multi-Class Extensions of Adjusted Classify and Count.\n:softmax_reg (our method) is a variant of :softmax, which sets one latent parameter to zero in addition to introducing a technical regularization term.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#IBU","page":"API reference","title":"IBU","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"IBU","category":"page"},{"location":"api-reference/#QUnfold.IBU","page":"API reference","title":"QUnfold.IBU","text":"IBU(transformer, n_bins; kwargs...)\n\nThe Iterative Bayesian Unfolding method by D'Agostini, 1995: A multidimensional unfolding method based on Bayes' theorem.\n\nKeyword arguments\n\no = 0 is the order of the polynomial for ordinal quantification.\nλ = 0.0 is the impact of the polynomial for ordinal quantification.\na = Float64[] are the acceptance factors for unfolding analyses.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#SLD","page":"API reference","title":"SLD","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"SLD","category":"page"},{"location":"api-reference/#QUnfold.SLD","page":"API reference","title":"QUnfold.SLD","text":"SLD(classifier; kwargs...)\n\nThe Saerens-Latinne-Decaestecker method, a.k.a. EMQ or Expectation Maximization-based Quantification by Saerens et al., 2002: Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure.\n\nA polynomial order o > 0 and regularization impact λ > 0 yield the o-SLD method for ordinal quantification, which is proposed by Bunse et al., 2022: Machine learning for acquiring knowledge in astro-particle physics.\n\nKeyword arguments\n\no = 0 is the order of the polynomial for o-SLD.\nλ = 0.0 is the impact of the polynomial for o-SLD.\na = Float64[] are the acceptance factors for unfolding analyses.\nfit_classifier = true whether or not to fit the given classifier.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#Feature-transformations","page":"API reference","title":"Feature transformations","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"The unfolding methods RUN, SVD, and IBU have the flexibility of choosing between different feature transformations.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"ClassTransformer\nTreeTransformer\nHistogramTransformer","category":"page"},{"location":"api-reference/#QUnfold.ClassTransformer","page":"API reference","title":"QUnfold.ClassTransformer","text":"ClassTransformer(classifier; kwargs...)\n\nThis transformer yields the classification-based feature transformation used in ACC, PACC, CC, PCC, and SLD.\n\nKeyword arguments\n\nis_probabilistic = false whether or not to use posterior predictions.\nfit_classifier = true whether or not to fit the given classifier.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#QUnfold.TreeTransformer","page":"API reference","title":"QUnfold.TreeTransformer","text":"TreeTransformer(tree; kwargs...)\n\nThis transformer yields a tree-induced partitioning, as proposed by Börner et al., 2017: Measurement/simulation mismatches and multivariate data discretization in the machine learning era.\n\nKeyword arguments\n\nfit_tree = 1. whether or not to fit the given tree. If fit_tree is false or 0., do not fit the tree and use all data for fitting M. If fit_tree is true or 1., fit both the tree and M with all data. If fit_tree is between 0 and 1, use a fraction of fit_tree for fitting the tree and the remaining fraction 1-fit_tree for fitting M.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#QUnfold.HistogramTransformer","page":"API reference","title":"QUnfold.HistogramTransformer","text":"HistogramTransformer(n_bins; kwargs...)\n\nThis transformer yields the histogram-based feature transformation used in HDx and HDy. The parameter n_bins specifies the number of bins per input feature.\n\nKeyword arguments\n\npreprocessor = nothing can be another AbstractTransformer that is called before this transformer.\n\n\n\n\n\n","category":"type"},{"location":"#Home","page":"Home","title":"QUnfold.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This Julia package implements our unified framework of quantification and unfolding algorithms.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"QUnfold.jl can be installed through the Julia package manager. From the Julia REPL, type ] to enter the Pkg mode of the REPL. Then run","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add QUnfold","category":"page"},{"location":"#Quick-start","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Each quantification / unfolding technique implements a fit and a predict function.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The fit function receives a training set (X, y) as an input. It returns a trained copy of the quantification / unfolding technique; no in-place training happens.\nThe predict function receives a single sample of multiple data items. It returns the estimated vector of class prevalences within this sample.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The underlying classifier of each technique must be a bagging classifier with oob_score=true, which implements the API of ScikitLearn.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using QUnfold, ScikitLearn\n@sk_import ensemble: RandomForestClassifier\n\nacc = ACC( # a scikit-learn bagging classifier with oob_score is needed\n    RandomForestClassifier(oob_score=true)\n)\n\n# X_trn, y_trn = my_training_data(...)\ntrained_acc = fit(acc, X_trn, y_trn) # fit returns a trained COPY\n\n# X_tst = my_testing_data(...)\np_est = predict(trained_acc, X_tst) # return a prevalence vector","category":"page"},{"location":"#Methods","page":"Home","title":"Methods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The following methods are implemented here:","category":"page"},{"location":"","page":"Home","title":"Home","text":"CC: The basic Classify & Count method.\nACC: The Adjusted Classify & Count method.\nPCC: The Probabilistic Classify & Count method.\nPACC: The Probabilistic Adjusted Classify & Count method.\nRUN: The Regularized Unfolding method.\nSVD: The Singular Value Decomposition-based unfolding method.\nHDx: The Hellinger Distance-based method on feature histograms.\nHDy: The Hellinger Distance-based method on prediction histograms.\nIBU: The Iterative Bayesian Unfolding method.\nSLD: The Saerens-Latinne-Decaestecker method, a.k.a. EMQ or Expectation Maximization-based Quantification.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Most of these methods support regularization towards smooth estimates, which is beneficial in ordinal quantification.","category":"page"},{"location":"#Citing","page":"Home","title":"Citing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This implementation is a part of my Ph.D. thesis.","category":"page"},{"location":"","page":"Home","title":"Home","text":"@PhdThesis{bunse2022machine,\n  author = {Bunse, Mirko},\n  school = {TU Dortmund University},\n  title  = {Machine Learning for Acquiring Knowledge in Astro-Particle Physics},\n  year   = {2022},\n  doi    = {10.17877/DE290R-23021},\n}","category":"page"}]
}
