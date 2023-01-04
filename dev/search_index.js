var documenterSearchIndex = {"docs":
[{"location":"experiments/#Experiments","page":"Experiments","title":"Experiments","text":"","category":"section"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"The experiments/ directory contains a Makefile with which you can run the experiments for our publication On Multi-Class Extensions of Adjusted Classify and Count.","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"@InProceedings{bunse2022multiclass,\n  author    = {Mirko Bunse},\n  title     = {On Multi-Class Extensions of Adjusted Classify and Count},\n  booktitle = {Int. Worksh. on Learn. to Quantify: Meth. and Appl.},\n  year      = {2022},\n  pages     = {43--50},\n}","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"CAUTION: We have run these experiments on 40 cores with 48 GB of RAM; with this setup, the experiments took 67 h. If you just want to check whether the scripts work, you can call make -f lequa.mk tests to traverse the entire code path with just a few iterations; this test completes in a few minutes.","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"cd experiments/\nmake -f lequa.mk -n # inspect all steps of our experimentation without running them (dry-run)\n\nmake -f lequa.mk # run all experimentation (CAUTION: computation-heavy)","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"We provide a Docker setup for those who prefer to run the experiments in an isolated environment, and possibly in a computing cluster.","category":"page"},{"location":"experiments/","page":"Experiments","title":"Experiments","text":"cd experiments/docker/\nmake # build the Docker image\n./run.sh # start an interactive Docker container from the image","category":"page"},{"location":"api-reference/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"Below, you find a listing of all public methods of this package. Any other method you might find in the source code is not intended for direct usage.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"CurrentModule = QUnfold","category":"page"},{"location":"api-reference/#Common-interface","page":"API reference","title":"Common interface","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"TODO with an exemplary link to fit.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"fit\npredict\npredict_with_background","category":"page"},{"location":"api-reference/#QUnfold.fit","page":"API reference","title":"QUnfold.fit","text":"fit(m, X, y) -> FittedMethod\n\nReturn a copy of the QUnfold method m that is fitted to the data set (X, y).\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.predict","page":"API reference","title":"QUnfold.predict","text":"predict(m, X) -> Vector{Float64}\n\nPredict the class prevalences in the data set X with the fitted method m.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.predict_with_background","page":"API reference","title":"QUnfold.predict_with_background","text":"predict_with_background(m, X, X_b, α=1) -> Vector{Float64}\n\nPredict the class prevalences in the observed data set X with the fitted method m, taking into account a background measurement X_b that is scaled by α.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#Quantification-/-unfolding-methods","page":"API reference","title":"Quantification / unfolding methods","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"CC\nACC\nPCC\nPACC\nRUN\nSVD\nHDx\nHDy\nIBU\nSLD","category":"page"},{"location":"api-reference/#QUnfold.CC","page":"API reference","title":"QUnfold.CC","text":"CC(classifier; kwargs...)\n\nThe Classify & Count method by Forman, 2008: Quantifying counts and costs via classification.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.ACC","page":"API reference","title":"QUnfold.ACC","text":"ACC(classifier; kwargs...)\n\nThe Adjusted Classify & Count method by Forman, 2008: Quantifying counts and costs via classification.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.PCC","page":"API reference","title":"QUnfold.PCC","text":"PCC(classifier; kwargs...)\n\nThe Probabilistic Classify & Count method by Bella et al., 2010: Quantification via Probability Estimators.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.PACC","page":"API reference","title":"QUnfold.PACC","text":"PACC(classifier; kwargs...)\n\nThe Probabilistic Adjusted Classify & Count method by Bella et al., 2010: Quantification via Probability Estimators.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.RUN","page":"API reference","title":"QUnfold.RUN","text":"RUN(transformer; kwargs...)\n\nThe Regularized Unfolding method by Blobel, 1985: Unfolding methods in high-energy physics experiments.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.SVD","page":"API reference","title":"QUnfold.SVD","text":"SVD(transformer; kwargs...)\n\nThe The Singular Value Decomposition-based unfolding method by Hoecker & Kartvelishvili, 1996: SVD approach to data unfolding.\n\n\n\n\n\n","category":"function"},{"location":"api-reference/#QUnfold.HDx","page":"API reference","title":"QUnfold.HDx","text":"HDx(n_bins; kwargs...)\n\nThe Hellinger Distance-based method on feature histograms by González-Castro et al., 2013: Class distribution estimation based on the Hellinger distance.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#QUnfold.HDy","page":"API reference","title":"QUnfold.HDy","text":"HDy(classifier, n_bins; kwargs...)\n\nThe Hellinger Distance-based method on prediction histograms by González-Castro et al., 2013: Class distribution estimation based on the Hellinger distance.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#QUnfold.IBU","page":"API reference","title":"QUnfold.IBU","text":"IBU(transformer, n_bins; kwargs...)\n\nThe Iterative Bayesian Unfolding method by D'Agostini, 1995: A multidimensional unfolding method based on Bayes' theorem.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#QUnfold.SLD","page":"API reference","title":"QUnfold.SLD","text":"SLD(classifier, n_bins; kwargs...)\n\nThe Saerens-Latinne-Decaestecker method, a.k.a. EMQ or Expectation Maximization-based Quantification by Saerens et al., 2002: Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure.\n\n\n\n\n\n","category":"type"},{"location":"api-reference/#Feature-transformations","page":"API reference","title":"Feature transformations","text":"","category":"section"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"TODO.","category":"page"},{"location":"api-reference/","page":"API reference","title":"API reference","text":"ClassTransformer\nTreeTransformer","category":"page"},{"location":"#Home","page":"Home","title":"QUnfold.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This Julia package implements our unified framework of quantification and unfolding algorithms.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"QUnfold.jl can be installed through the Julia package manager. From the Julia REPL, type ] to enter the Pkg mode of the REPL. Then run","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add QUnfold","category":"page"},{"location":"#Quick-start","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Each quantification / unfolding technique implements a fit and a predict function.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The fit function receives a training set (X, y) as an input. It returns a trained copy of the quantification / unfolding technique; no in-place training happens.\nThe predict function receives a single sample of multiple data items. It returns the estimated vector of class prevalences within this sample.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The underlying classifier of each technique must implement the API of ScikitLearn.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using QUnfold, ScikitLearn\n@sk_import linear_model: LogisticRegression\n\n# X_trn, y_trn = my_training_data(...)\n\nacc = ACC(LogisticRegression())\ntrained_acc = fit(acc, X_trn, y_trn) # fit returns a trained COPY\n\n# X_tst = my_testing_data(...)\n\np_est = predict(trained_acc, X_tst) # return a prevalence vector","category":"page"},{"location":"#Methods","page":"Home","title":"Methods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The following methods are implemented here:","category":"page"},{"location":"","page":"Home","title":"Home","text":"CC: The basic Classify & Count method (Forman, 2008).\nACC: The Adjusted Classify & Count method (Forman, 2008).\nPCC: The Probabilistic Classify & Count method (Bella et al., 2010).\nPACC: The Probabilistic Adjusted Classify & Count method (Bella et al., 2010).\nRUN: The Regularized Unfolding method (Blobel, 1985).\nSVD: The Singular Value Decomposition-based unfolding method (Hoecker & Kartvelishvili, 1996).\nHDx: The Hellinger Distance-based method on feature histograms (González-Castro et al., 2013).\nHDy: The Hellinger Distance-based method on prediction histograms (González-Castro et al., 2013).\nIBU: The Iterative Bayesian Unfolding method (D'Agostini, 1995).\nSLD: The Saerens-Latinne-Decaestecker method, a.k.a. EMQ or Expectation Maximization-based Quantification (Saerens et al., 2002).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Most of these methods support regularization towards smooth estimates, which is beneficial in ordinal quantification.","category":"page"},{"location":"#Citing","page":"Home","title":"Citing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This implementation is a part of my Ph.D. thesis.","category":"page"},{"location":"","page":"Home","title":"Home","text":"@PhdThesis{bunse2022machine,\n  author = {Bunse, Mirko},\n  school = {TU Dortmund University},\n  title  = {Machine Learning for Acquiring Knowledge in Astro-Particle Physics},\n  year   = {2022},\n  doi    = {10.17877/DE290R-23021},\n}","category":"page"}]
}
